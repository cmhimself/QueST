import torch_geometric.seed
import torch
import random
import logging
import anndata
import numpy as np
import scanpy as sc
import squidpy as sq
import networkx as nx
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph
from bench.bench_utils import visualize_niche, show_plot_with_timeout, visualize_graph, get_time_str
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def fix_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    #
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch_geometric.seed.seed_everything(seed)


def shuffle(adata, dataset, feature=None, fix_portion=0.02, min_k=1, max_k=3, plot=False, min_shuffle_ratio=0.5, max_shuffle_ratio=0.5):
    """graph should be constructed after shuffling"""
    logger.debug("Shuffling adata, using original adata to construct spatial graph")
    adata_shuffle = adata.copy() 
    if 'spatial_neighbors' in adata_shuffle.uns:
        del adata_shuffle.uns['spatial_neighbors']
    if 'spatial_connectivities' in adata_shuffle.obsp:
        del adata_shuffle.obsp['spatial_connectivities']
    if 'spatial_distances' in adata_shuffle.obsp:
        del adata_shuffle.obsp['spatial_distances']

    G_original = nx.from_scipy_sparse_array(adata.obsp['spatial_connectivities'])

    num_samples = int(fix_portion * adata_shuffle.n_obs)  # for example, 10% of cells
    fixed_center, fixed_nodes, k_list = set(), set(), []
    for _ in range(num_samples):
        while True:
            cell_idx = random.randint(0, adata_shuffle.n_obs - 1)
            if cell_idx not in fixed_center:
                break
        k = random.randint(min_k, max_k)
        try:
            subgraph = nx.ego_graph(G_original, cell_idx, radius=k)
        except:
            logger.debug("ignore isolated node for fix center")
            continue
        fixed_center.add(cell_idx)
        k_list.append(k)
        fixed_nodes.update(subgraph.nodes)
    logger.info(f"total nodes: {adata_shuffle.n_obs}, fixed center: {len(fixed_center)}, total fixed nodes: {len(fixed_nodes)}, k list: {Counter(k_list)}")

    all_indices = np.arange(adata.n_obs)
    shufflable_indices = list(set(range(adata_shuffle.n_obs)) - fixed_nodes)
    shuffled_indices = np.random.permutation(all_indices[shufflable_indices])
    all_indices[shufflable_indices] = shuffled_indices
    adata_shuffle = adata_shuffle[all_indices]

    if feature is not None:
        feature = feature[all_indices]

    adata_shuffle.obsm['spatial'] = adata.obsm['spatial'].copy()
    negative_subgraph_center = []
    ratio_list = []
    for ind in shufflable_indices:
        try:
            subgraph = nx.ego_graph(G_original, ind, radius=max_k)
        except:  # ignore isolated node
            logger.debug("ignore isolated node in calculating ")
            continue
        shuffle_ratio = np.sum([node in shufflable_indices for node in subgraph.nodes()]) / len(subgraph.nodes())
        ratio_list.append(shuffle_ratio)
        if min_shuffle_ratio <= shuffle_ratio <= max_shuffle_ratio:
            negative_subgraph_center.append(ind)
    logger.info(f"{len(negative_subgraph_center)} out of {len(shufflable_indices)} nodes with shuffle ratio in range [{min_shuffle_ratio}, {max_shuffle_ratio}] selected as negative samples")
    # plt.hist(ratio_list, bins='auto')
    # plt.show()
    if plot:
        if dataset == 'DLPFC':
            spot_size = 5
        elif dataset == 'MouseOlfactoryBulbTissue':
            if adata.uns['library_id'] == '10x':
                spot_size = 75
            elif adata.uns['library_id'] == 'slidev2':
                spot_size = 25
            elif adata.uns['library_id'] == 'stereoseq':
                spot_size = 1
            else:
                assert False, f"Unknown library id {adata.uns['library_id']} for dataset MouseOlfactoryBulbTissue!"
        else:
            assert False, f"Unknown dataset {dataset}!"
        sc.pl.spatial(adata_shuffle, spot_size=spot_size, color='cell_type', show=False)
        plt.gca().invert_yaxis()
        print("saving fig")
        plt.savefig("./results/fig/shuffle.pdf", dpi=300)
        plt.show()

    return adata_shuffle, np.array(list(fixed_center)), np.array(list(fixed_nodes)), np.array(negative_subgraph_center), feature


def load_adata(q_folder="./bench/adata_query/DLPFC", q_id=None,
               ref_folder="./data/DLPFC/adata_filtered", ref_id_list=None):
    logger.info("loading adata")
    q_path = f"{q_folder}/{q_id}.h5ad"
    adata_query = anndata.read_h5ad(q_path)
    adata_ref_list = [anndata.read_h5ad(f"{ref_folder}/{ref_id}.h5ad") for ref_id in ref_id_list]
    return adata_query, adata_ref_list


def build_graphs(adata_list, dataset="DLPFC"):
    logger.info(f"building graphs, time: {get_time_str()}")
    if dataset == "DLPFC":
        for adata in adata_list:
            sq.gr.spatial_neighbors(adata, coord_type='grid')
    elif dataset == 'MouseOlfactoryBulbTissue':
        for adata in adata_list:
            if adata.uns['library_id'] in ['10x', 'stereoseq']:
                sq.gr.spatial_neighbors(adata, coord_type='grid')

            elif adata.uns['library_id'] == 'slidev2':
                sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 100))
                # G = nx.from_scipy_sparse_array(adata.obsp['spatial_connectivities'])
                # visualize_graph(adata, G, title="delaunay radius 120"), exit()
    else:
        assert False, f"Unknown dataset {dataset}!"


def preprocess_adata(adata_list, param=None):
    logger.info(f"preprocessing adata: scale={param['scale']}, pca={param['pca']}")

    if param['min_count'] is not None:
        adata_ref_list = adata_list[1:]
        gene_total_counts = np.array([np.ravel(adata_ref.X.sum(axis=0)) for adata_ref in adata_ref_list])
        gene_mask = np.all(gene_total_counts > param['min_count'], axis=0)

        # only use reference data to select genes
        for i in range(len(adata_list)):
            adata_list[i] = adata_list[i][:, gene_mask]

    logger.info(f"{adata_list[0].n_vars} genes passed the filter with min count > {param['min_count']}, making adata copies")
    adata_raw_list = [adata.copy() for adata in adata_list]

    # normalize the data first
    # for adata in adata_list:
    #     logger.info(f"preprocessing {adata.uns[param['library_key']]}")
    #     sc.pp.normalize_total(adata, target_sum=1e4)
    #     sc.pp.log1p(adata)

    if param['hvg'] is not None:
        # assert not param['scale']
        hvg_list = []
        for adata in adata_list:
            logger.info(f"selecting hvg for {adata.uns['library_id']}")
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=param['hvg'])
            # sc.pp.highly_variable_genes(adata, n_top_genes=param['hvg'])
            hvg_list.append(adata.var[adata.var['highly_variable']].index)

        hvg_union = set().union(*hvg_list)
        hvg_union = list(sorted(hvg_union))
        if len(hvg_union) %2 != 0:
            hvg_union = hvg_union[1:]
        logger.info(f"{len(hvg_union)} union hvg genes selected")
        for i, adata in enumerate(adata_list):
            # adata = adata.raw[:, list(hvg_union)]
            adata_list[i] = adata_raw_list[i][:, hvg_union]
        # normalize the data again
        for adata in adata_list:
            logger.info(f"preprocessing {adata.uns['library_id']}")
            # sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)

    for adata in adata_list:
        if param['scale']:
            sc.pp.scale(adata)
        if param['pca']:
            assert param['scale']
            sc.tl.pca(adata, n_comps=param['n_pcs'])

    return adata_list


def extract_tensors_and_graphs(adatas):
    data_list = []
    for adata in adatas:
        x = torch.tensor(adata.X, dtype=torch.float)
        edge_index = torch.tensor(adata.obsp['connectivities'].nonzero(), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    return data_list


def prepare_graph_data(adata_ref_list, param):
    logger.info(f"constructing spatial graph, computing {param['model_k']}-hop subgraph and creating batch labels for each sample!")
    feature_list, edge_ind_list, batch_label_list = [], [], []
    sub_node_sample_list, sub_edge_ind_sample_list = [], [] 
    for i in range(len(adata_ref_list)):
        feature = get_feature(adata_ref_list[i], query=False, param=param, ref_id=adata_ref_list[i].uns['library_id'], device=param['device'])

        adj_mat = adata_ref_list[i].obsp['spatial_connectivities'].tocoo()
        edge_index = torch.tensor(np.vstack((adj_mat.row, adj_mat.col)), dtype=torch.int64).to(param['device']) 

        if isinstance(param['model_k'], int):
            k = param['model_k']
        elif isinstance(param['model_k'], list):
            if len(param['model_k']) < len(adata_ref_list):
                k = param['model_k'][0]
            else:
                k = param['model_k'][i]
        else:
            assert False, f"Unknown model k format {param['model_k']}!"
        logger.info(f"computing {k} hop subgraph for sample {adata_ref_list[i].uns[param['library_key']]}")
        sub_node_list, sub_edge_ind_list = [], []
        for node_ind in tqdm(range(adata_ref_list[i].n_obs)):
            sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(node_ind, k, edge_index, relabel_nodes=True)
            sub_node_list.append(sub_nodes)
            sub_edge_ind_list.append(sub_edge_index)

        batch_label = torch.zeros((adata_ref_list[i].n_obs, len(adata_ref_list))).to(param['device'])
        batch_label[:, i] = 1

        feature_list.append(feature)
        edge_ind_list.append(edge_index)
        sub_node_sample_list.append(sub_node_list)
        sub_edge_ind_sample_list.append(sub_edge_ind_list)
        batch_label_list.append(batch_label)

    return feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list, batch_label_list


def get_feature(adata, query=False, param=None, ref_id=None, device="cuda:0"):
    # if ref_id is None:
    #     ref_id = param['ref_id_list'][0]
    sample_id = adata.uns['library_id']
    if param['cca']:
        logger.info("using CCA embeddings")
        if query:
            feature = torch.load(f"{param['cca_folder']}/{sample_id}_{ref_id}.pt", map_location=device)
        else:
            feature = torch.load(f"{param['cca_folder']}/{sample_id}.pt", map_location=device)
    else:
        if not param['pca']:
            logger.info("using normalized count")
            feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
        else:
            logger.info("using PCA embeddings")
            feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
    return feature


def get_features(adata, adata_shf, param):
    device = param['device']
    if adata_shf is None:
        if not param['pca']:
            feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
            return feature
        else:
            feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
            return feature

    if not param['pca']:
        feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
        feature_shf = torch.tensor(adata_shf.X, dtype=torch.float32).to(device)
        return feature, feature_shf
    else:
        feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
        feature_shf = torch.tensor(adata_shf.obsm['X_pca'], dtype=torch.float32).to(device)
        return feature, feature_shf


def save_checkpoint(model, model_folder, epoch, ckpt_list=None, run=None, model_name=None, test_str=''):
    if model_name is None:
        ckpt_model_path = f"{model_folder}/{test_str}model_{epoch}.pth"
    else:
        ckpt_model_path = f"{model_folder}/{test_str}{model_name}_{epoch}.pth"
    if ckpt_list is not None:
        ckpt_list.append((epoch, ckpt_model_path))
    logger.info(f"saving model weights to {ckpt_model_path}")
    torch.save(model.state_dict(), ckpt_model_path)


def query(feature_q, feature_ref, edge_index_q, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref, model, niche_mask, method='cosine'):
    niche_z = model.encode_subgraph(feature_q, edge_index_q, niche_mask)
    # ref_z = model.encode_sample(feature_ref, edge_index_ref)
    ref_z = model.get_subgraph_rep(feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref)
    # ref_z, _ = model.get_latent_params(feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_r

    if method == 'cosine':
        logger.info("performing cosine similarity query")
        niche_z = torch.reshape(niche_z, (niche_z.shape[0], 1))
        ref_z_normed = F.normalize(ref_z, p=2, dim=1)
        niche_z_normed = F.normalize(niche_z, p=2, dim=0)
        similarities = torch.mm(ref_z_normed, niche_z_normed).squeeze(1)
    elif method == 'discriminator':
        logger.info("performing discriminator query")
        niche_z = torch.reshape(niche_z, (1, niche_z.shape[0]))
        niche_z = niche_z.expand(ref_z.shape[0], -1)
        similarities = model.contrastive_discriminator(niche_z, ref_z)
    else:
        assert False, f"Unknown query method {method}"

    return similarities


def visualize_query_result(fig_folder, ckpt_epoch, adata_ref, edge_index_ref, niche_prefix, param, subplot=False):
    logger.info("visualizing query result")
    ref_id = adata_ref.uns[param['library_key']]
    if not subplot:
        sc.pl.spatial(adata_ref, color=f"{param['query_method']}_{ckpt_epoch}_similarity", spot_size=5, show=False, title=f"{ref_id} epoch={ckpt_epoch} {param['query_method']} sim")
        if param['invert_y']:
            plt.gca().invert_yaxis()
        plt.savefig(f"{fig_folder}/{ref_id}_epoch={ckpt_epoch}_{param['query_method']}_value.png", dpi=300)
        # plt.show()
        show_plot_with_timeout(5)
        # print(ref_z_normed.shape, niche_z_normed.shape, similarities.shape)

        best_node = torch.argmax(torch.tensor(adata_ref.obs[f"{param['query_method']}_similarity"].to_list())).item()
        best_subgraph, _, _, _ = k_hop_subgraph(best_node, param['model_k'], edge_index_ref)
        best_subgraph = best_subgraph.cpu().numpy()
        visualize_niche(adata_ref, best_subgraph, niche_name=f'{niche_prefix}_query', invert_y=True,
                        title=f"{ref_id} epoch={ckpt_epoch} {niche_prefix} query",
                        save_path=f"{fig_folder}/{ref_id}_epoch={ckpt_epoch}_{param['query_method']}_query.png", show=False)
    # else:
    #     sc.pl.spatial(adata_ref, color=f"{param['query_method']}_similarity", spot_size=5, show=False, title=f"{ref_id} epoch={ckpt_epoch} {param['query_method']} sim", ax=ax)


def get_shuffle_param(param, current_ref_ind, total_ref_num):
    if isinstance(param['shuffle_min_k'], int):
        min_k = param['shuffle_min_k']
        max_k = param['shuffle_max_k']
        fix_portion = param['fix_portion']
    elif isinstance(param['shuffle_min_k'], list):
        if len(param['shuffle_min_k']) < total_ref_num:
            min_k = param['shuffle_min_k'][0]
            max_k = param['shuffle_max_k'][0]
            fix_portion = param['fix_portion'][0]
        else:
            min_k = param['shuffle_min_k'][current_ref_ind]
            max_k = param['shuffle_max_k'][current_ref_ind]
            fix_portion = param['fix_portion'][current_ref_ind]
    else:
        assert False, f"Unknown model k format {param['shuffle_min_k']}!"
    return min_k, max_k, fix_portion


def get_spot_size(dataset, ref_id):
    if dataset == 'DLPFC':
        spot_size = 5
    elif dataset == 'MouseOlfactoryBulbTissue':
        if ref_id == '10x':
            spot_size = 75
        elif ref_id == 'slidev2':
            spot_size = 25
        elif ref_id == 'stereoseq':
            spot_size = 1
        else:
            assert False, f"Unknown library id {ref_id} for dataset MouseOlfactoryBulbTissue!"
    else:
        assert False, f"Unknown dataset {dataset}!"
    return spot_size
