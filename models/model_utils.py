import os
import umap
import scipy
import torch
import random
import logging
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import networkx as nx
import torch.nn as nn
import seaborn as sns
import scipy.sparse as sp
import torch_geometric.seed
import matplotlib.pyplot as plt
import torch.nn.functional as F
import bench.bench_utils as bench_utils
import models.model_utils as model_utils
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from bench.bench_utils import visualize_niche, show_plot_with_timeout, get_time_str
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


def check_overall_distribution(adata_list, param, title=None, rna_counts=None):
    for adata in adata_list:
        if rna_counts is None:
            rna_counts = adata.obsm['X_pooled'] if param['pool_expr'] else adata.X
        if sp.issparse(rna_counts):
            data = rna_counts.A.flatten()
        else:
            data = rna_counts.flatten()

        # Plot histogram of gene expression values
        plt.figure(figsize=(8, 6))
        plt.hist(data[data > 0], bins=100, color='blue', alpha=0.7, label='Non-zero counts')  # Exclude zeros for log scale
        plt.yscale('log')  # Log-scale to highlight the tail of the distribution
        plt.xlabel('Gene Expression Value')
        plt.ylabel('Frequency (Log Scale)')
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'Distribution of Gene Expression Values Pooling={param["pool_expr"]}')
        plt.legend()
        plt.show()


def check_mean_variance(adata_list, param, title=None):
    for adata in adata_list:
        rna_counts = adata.obsm['X_pooled'] if param['pool_expr'] else adata.X
        rna_counts = rna_counts if not sp.issparse(rna_counts) else rna_counts.toarray()
        if sp.issparse(rna_counts):
            gene_means = np.array(rna_counts.mean(axis=0)).flatten()
            gene_vars = np.array(rna_counts.var(axis=0)).flatten()
        else:
            gene_means = rna_counts.mean(axis=0)
            gene_vars = rna_counts.var(axis=0)

        # Scatter plot of mean vs. variance
        plt.figure(figsize=(8, 6))
        plt.scatter(gene_means, gene_vars, alpha=0.6, color='purple', edgecolor='k', s=20)
        plt.plot([0, max(gene_means)], [0, max(gene_means)], color='red', linestyle='--', label='Variance = Mean')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Mean Gene Expression')
        plt.ylabel('Variance')
        if title is not None:
            plt.title(title)
        else:
            plt.title(f'Mean-Variance Relationship Pooling={param["pool_expr"]}')
        plt.legend()
        plt.show()


def shuffle(adata, dataset, feature=None, fix_portion=0.02, min_k=1, max_k=3, plot=False, min_shuffle_ratio=0.5, max_shuffle_ratio=0.5, sub_node_list=None):
    """graph should be constructed after shuffling"""
    """
    fix portion: DLPFC K=3: 0.02
    fix portion: DLPFC K=2: 0.05
    """
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
        if sub_node_list is not None:  # TODO: give sub_node_list means fixed subgraph size
            if len(sub_node_list[cell_idx]) == 1:
                logger.debug("ignore isolated node for fix center")
                continue
            fixed_nodes.update(sub_node_list[cell_idx].tolist())
        else:
            try:
                subgraph = nx.ego_graph(G_original, cell_idx, radius=k)
            except:
                logger.debug("ignore isolated node for fix center")
                continue
            fixed_nodes.update(subgraph.nodes)
        fixed_center.add(cell_idx)
        k_list.append(k)

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
    # ratio_list = []
    logger.info("selecting subgraphs")

    # print(sub_node_list)

    shufflable_tensor = torch.tensor(shufflable_indices, device=sub_node_list[0].device)  # torch.Size([2457])
    subgraph_padded = torch.nn.utils.rnn.pad_sequence([sub_node_list[i] for i in shufflable_indices], batch_first=True, padding_value=-1)  # torch.Size([4758, 37])
    is_in_shufflable = torch.isin(subgraph_padded, shufflable_tensor)  # torch.Size([4758, 37])
    valid_nodes = subgraph_padded != -1  # torch.Size([4758, 37])
    shuffle_ratios = (is_in_shufflable & valid_nodes).float().sum(dim=1) / valid_nodes.float().sum(dim=1)  # torch.Size([4758])
    mask = (shuffle_ratios >= min_shuffle_ratio) & (shuffle_ratios <= max_shuffle_ratio)
    negative_subgraph_center = shufflable_tensor[mask].cpu().numpy()

    logger.info(f"{len(negative_subgraph_center)} out of {len(shufflable_indices)} nodes with shuffle ratio in range [{min_shuffle_ratio}, {max_shuffle_ratio}] selected as negative samples")
    # plt.hist(ratio_list, bins='auto')
    # plt.show()
    if plot:
        sc.pl.spatial(adata_shuffle, spot_size=model_utils.get_spot_size(dataset, adata.uns['library_id']), color='cell_type', show=False)
        plt.gca().invert_yaxis()
        print("saving fig")
        plt.savefig("./results/fig/shuffle.pdf", dpi=300)
        plt.show()

    # return adata_shuffle, np.array(list(fixed_center)), np.array(list(fixed_nodes)), np.array(negative_subgraph_center), feature
    return adata_shuffle, np.array(list(fixed_center)), np.array(list(fixed_nodes)), negative_subgraph_center, feature


def load_adata(q_folder="./bench/adata_query/3.28/DLPFC", q_id=None,
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
            if adata.uns['library_id'] in ['10X', 'Stereo-seq']:
                sq.gr.spatial_neighbors(adata, coord_type='grid')
            elif adata.uns['library_id'] == 'Slide-seq V2':
                sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 100))
    elif dataset == "Simulation":
        for adata in adata_list:
            sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 5))
    elif dataset == "nsclc":
        for adata in adata_list:
            sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 300))
            # G = nx.from_scipy_sparse_array(adata.obsp['spatial_connectivities'])
            # visualize_graph(adata, G, title="delaunay radius 300"), exit()
    else:
        assert False, f"Unknown dataset {dataset}!"


def preprocess_adata(adata_list, param=None):
    logger.info(f"preprocessing adata, selecting common features")
    for adata in adata_list:
        adata.var_names_make_unique()
    logger.info(f"gene num before intersection: {[adata.shape[1] for adata in adata_list]}")
    gene_sets = [set(adata.var_names) for adata in adata_list]
    common_genes = list(sorted(set.intersection(*gene_sets)))
    logger.info(f"{len(common_genes)} common genes identified")
    for i in range(len(adata_list)):
        adata_list[i] = adata_list[i][:, common_genes]
    logger.info(f"gene num after intersection: {[adata.shape[1] for adata in adata_list]}")
    if param['min_count'] is not None:
        logger.info(f"filtering genes with min count={param['min_count']}")
        gene_total_counts = np.array([np.ravel(adata.X.sum(axis=0)) for adata in adata_list])
        gene_mask = np.all(gene_total_counts > param['min_count'], axis=0)
        for i in range(len(adata_list)):
            adata_list[i] = adata_list[i][:, gene_mask]
    logger.info(f"{adata_list[0].n_vars} genes passed the filter with min count > {param['min_count']}, making adata copies")
    adata_raw_list = [adata.copy() for adata in adata_list]

    if param['hvg'] is not None:
        hvg_list = []
        for adata in adata_list:
            logger.info(f"selecting hvg for {adata.uns['library_id']}, time: {bench_utils.get_time_str()}")
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=param['hvg'])
            hvg_list.append(adata.var[adata.var['highly_variable']].index)

        hvg_union = list(sorted(set().union(*hvg_list)))
        logger.info(f"{len(hvg_union)} union hvg genes selected")
        for i, adata in enumerate(adata_list):
            adata_list[i] = adata_raw_list[i][:, hvg_union]

    if param['normalize']:
        for adata in adata_list:
            logger.info(f"Normalizing {adata.uns['library_id']} count")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
    else:
        logger.info("Skip normalizing count")

    return adata_list


def extract_tensors_and_graphs(adatas):
    data_list = []
    for adata in adatas:
        x = torch.tensor(adata.X, dtype=torch.float)
        edge_index = torch.tensor(adata.obsp['connectivities'].nonzero(), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    return data_list


def prepare_graph_data(adata_ref_list, param, use_saved_subgraph=False):
    logger.info(f"Constructing spatial graph, computing {param['model_k']}-hop subgraph and creating batch labels for each sample! Batch num: {param['batch_num']}")
    feature_list, edge_ind_list, batch_label_list = [], [], []
    sub_node_sample_list, sub_edge_ind_sample_list = [], []  # 每个元素对应一张切片中的子图集合，每个子图集合对应里面的点和边
    for i in range(len(adata_ref_list)):
        feature = get_feature(adata_ref_list[i], query=False, param=param, ref_id=adata_ref_list[i].uns['library_id'])

        adj_mat = adata_ref_list[i].obsp['spatial_connectivities'].tocoo()
        edge_index = torch.tensor(np.vstack((adj_mat.row, adj_mat.col)), dtype=torch.int64).to(param['device'])  # 必须是int64不然下面dtype会不一致报错
        # edge_index = SparseTensor(np.vstack((adj_mat.row, adj_mat.col)).tocoo()).to(param['device'])  # 必须是int64不然下面dtype会不一致报错

        if isinstance(param['model_k'], int):
            k = param['model_k']
        elif isinstance(param['model_k'], list):
            if len(param['model_k']) < len(adata_ref_list):
                k = param['model_k'][0]
            else:
                k = param['model_k'][i]
            logger.info(f"using k={k} for sample {adata_ref_list[i].uns[param['library_key']]}")
        else:
            assert False, f"Unknown model k format {param['model_k']}!"

        if use_saved_subgraph:
            logger.info(f"loading saved {k} hop subgraph for sample {adata_ref_list[i].uns[param['library_key']]}, time: {get_time_str()}")
            if k == 3:
                sub_node_list = torch.load(f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_node_list.pt", map_location="cuda:0")
                sub_edge_ind_list = torch.load(f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_edge_index_list.pt", map_location="cuda:0")
            else:
                sub_node_list = torch.load(f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_node_list_k={k}.pt", map_location="cuda:0")
                sub_edge_ind_list = torch.load(f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_edge_ind_list_k={k}.pt", map_location="cuda:0")
        else:
            logger.info(f"computing {k} hop subgraph for sample {adata_ref_list[i].uns[param['library_key']]}, time: {get_time_str()}")
            sub_node_list, sub_edge_ind_list = [], []
            for node_ind in tqdm(range(adata_ref_list[i].n_obs)):
                sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(node_ind, k, edge_index, relabel_nodes=True)
                sub_node_list.append(sub_nodes)
                sub_edge_ind_list.append(sub_edge_index)
            if k != 3 and not os.path.exists(f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_node_list_k={k}.pt"):
            # if k != 3:
                print("saving sub node list and sub edge list")
                torch.save(sub_node_list, f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_node_list_k={k}.pt")
                torch.save(sub_edge_ind_list, f"./data/ccRCC/subgraph/{adata_ref_list[i].uns['library_id']}_sub_edge_ind_list_k={k}.pt")
        # batch_label = torch.zeros((adata_ref_list[i].n_obs, len(adata_ref_list))).to(param['device'])  # TODO: quest v2 still needs this
        batch_label = torch.zeros((adata_ref_list[i].n_obs, param['batch_num'])).to(param['device'])

        if adata_ref_list[i].uns['library_id'] in ["151507", "151508", "151509", "151510"]:
            batch_label[:, 0] = 1
        elif adata_ref_list[i].uns['library_id'] in ["151669", "151670", "151671", "151672"]:
            batch_label[:, 1] = 1
        elif adata_ref_list[i].uns['library_id'] in ["151673", "151674", "151675", "151676"]:
            batch_label[:, 2] = 1
        else:
            batch_label[:, i] = 1

        feature_list.append(feature)
        edge_ind_list.append(edge_index)
        sub_node_sample_list.append(sub_node_list)
        sub_edge_ind_sample_list.append(sub_edge_ind_list)
        batch_label_list.append(batch_label)

    return feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list, batch_label_list


def get_feature(adata, query=False, param=None, ref_id=None):
    device = "cuda:0"
    sample_id = adata.uns['library_id']
    if 'cca' in param.keys() and param['cca']:
        logger.info("getting feature: CCA embeddings")
        if query:
            feature = torch.load(f"{param['cca_folder']}/{sample_id}_{ref_id}.pt", map_location=device)
        else:
            feature = torch.load(f"{param['cca_folder']}/{sample_id}.pt", map_location=device)
    else:
        if 'pca' in param.keys() and param['pca']:
            logger.info("getting feature: PCA embeddings")
            feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)  # 这个确实是normed
        else:
            logger.info("getting feature: gene expression")
            if scipy.sparse.issparse(adata.X):
                cell_feature = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)  # 这个确实是normed
            else:
                cell_feature = torch.tensor(adata.X, dtype=torch.float32).to(device)  # 这个确实是normed
            if 'pool_expr' in param.keys() and param['pool_expr']:
                logger.info("using pooled normalized count as well")
                niche_feature = torch.tensor(adata.obsm['X_pooled'], dtype=torch.float32).to(device)
                feature = (cell_feature, niche_feature)
            else:
                feature = cell_feature
    return feature


def get_features(adata, adata_shf, param):
    device = param['device']
    if adata_shf is None:
        if not param['pca']:
            feature = torch.tensor(adata.X, dtype=torch.float32).to(device)  # 这个确实是normed
            return feature
        else:
            feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)  # 这个确实是normed
            return feature

    if not param['pca']:
        feature = torch.tensor(adata.X, dtype=torch.float32).to(device)  # 这个确实是normed
        feature_shf = torch.tensor(adata_shf.X, dtype=torch.float32).to(device)
        return feature, feature_shf
    else:
        feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)  # 这个确实是normed
        feature_shf = torch.tensor(adata_shf.obsm['X_pca'], dtype=torch.float32).to(device)
        return feature, feature_shf


def save_checkpoint(model, model_folder, epoch, ckpt_list, run, model_name=None, test_str=''):
    if model_name is None:
        ckpt_model_path = f"{model_folder}/{test_str}model_{epoch + 1}.pth"
    else:
        ckpt_model_path = f"{model_folder}/{test_str}{model_name}_{epoch + 1}.pth"
    ckpt_list.append((epoch + 1, ckpt_model_path))
    logger.info(f"saving model weights to {ckpt_model_path}")
    torch.save(model.state_dict(), ckpt_model_path)
    # artifact_model = wandb.Artifact("model_weight", type='model')
    # artifact_model.add_file(ckpt_model_path)
    # run.log_artifact(artifact_model)


def query(feature_q, feature_ref, edge_index_q, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref, model, niche_mask, method='cosine'):
    niche_z = model.encode_subgraph(feature_q, edge_index_q, niche_mask)
    # ref_z = model.encode_sample(feature_ref, edge_index_ref)
    ref_z = model.get_subgraph_rep(feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref)
    # _, ref_z = model.get_subgraph_rep(feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref)
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
        similarities = model.contrast_disc(niche_z, ref_z)
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


def get_shuffle_param(param, current_adata_ind, total_adata_num):
    if isinstance(param['shuffle_min_k'], int):
        min_k = param['shuffle_min_k']
        max_k = param['shuffle_max_k']
        fix_portion = param['fix_portion']
    elif isinstance(param['shuffle_min_k'], list):
        if len(param['shuffle_min_k']) < total_adata_num:
            min_k = param['shuffle_min_k'][0]
            max_k = param['shuffle_max_k'][0]
            fix_portion = param['fix_portion'][0]
        else:
            min_k = param['shuffle_min_k'][current_adata_ind]
            max_k = param['shuffle_max_k'][current_adata_ind]
            fix_portion = param['fix_portion'][current_adata_ind]
    else:
        assert False, f"Unknown model k format {param['shuffle_min_k']}!"
    return min_k, max_k, fix_portion


def get_spot_size(dataset, ref_id):
    if dataset == 'DLPFC':
        spot_size = 5
    elif dataset == 'MouseOlfactoryBulbTissue':
        if ref_id == '10X':
            spot_size = 75
        elif ref_id == 'Slide-seq V2':
            spot_size = 25
        elif ref_id == 'Stereo-seq':
            spot_size = 1
        else:
            assert False, f"Unknown library id {ref_id} for dataset MouseOlfactoryBulbTissue!"
    elif dataset == "Simulation":
        spot_size = 1
    elif dataset == "nsclc":
        spot_size = 75
    else:
        assert False, f"Unknown dataset {dataset}!"
    return spot_size


def check_embedding_batch(emb_list, batch_list, anno_list=None, pca=False):
    print("checking batch")
    # for emb, batch, anno in zip(emb_list, batch_list, anno_list):
    #     print(emb.shape, batch, anno.shape)
    embeddings = np.vstack(emb_list)

    batch_num = 0
    batch_ids = []
    for emb, batch in zip(emb_list, batch_list):
        batch_ids += [batch for _ in range(emb.shape[0])]
        batch_num += 1

    if pca:
        pca_model = PCA(n_components=50)  # Reduce to 50 dimensions or another reasonable choice
        embeddings = pca_model.fit_transform(embeddings)
        print(f"PCA reduced embeddings to {embeddings.shape[1]} dimensions.")

    umap_model = umap.UMAP(n_neighbors=100, min_dist=1.0, n_components=2, random_state=42, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)

    df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    df['Batch'] = pd.Categorical(batch_ids)

    if batch_num <= 10:
        palette = 'tab10'
    else:
        palette = sns.color_palette('tab20', 20) + sns.color_palette('Set1', 10)

    if anno_list is not None:
        annotations = np.concatenate(anno_list)
        df['Annotation'] = pd.Categorical(annotations)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        sns.scatterplot(ax=axes[0], data=df, x='UMAP1', y='UMAP2', hue='Batch', palette=palette, s=10, alpha=0.8)
        axes[0].set_title('UMAP Visualization by Batch')
        axes[0].legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')

        sns.scatterplot(ax=axes[1], data=df, x='UMAP1', y='UMAP2', hue='Annotation', palette=palette, s=10, alpha=0.8)
        axes[1].set_title('UMAP Visualization by Annotation')
        axes[1].legend(title='Annotation', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()
    else:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='Batch', palette=palette, s=10, alpha=0.8)
        plt.title('UMAP Visualization by Batch')
        plt.legend(title='Batch', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def chunked_top_k_cosine_sim(z_q, z_k, k, chunk_size=8192, device='cuda:0'):
    z_q_norm = F.normalize(z_q, p=2, dim=1)  # (N_q, d)
    z_k_norm = F.normalize(z_k, p=2, dim=1)  # (N_k, d)
    N_q = z_q_norm.size(0)
    top_k_indices = torch.empty((N_q, k), dtype=torch.long, device=device)  # (N_q, M) with the indices of top M neighbors in z_k

    start = 0
    while start < N_q:
        end = min(start + chunk_size, N_q)  # 实际上是在query sample上做chunk
        sim_chunk = z_q_norm[start: end] @ z_k_norm.T  # shape = (chunk_size, N_k)
        _, top_k_ind = torch.topk(sim_chunk, k=k, dim=1)  # top k over the second dimension -> shape (chunk_size, M)
        top_k_indices[start:end] = top_k_ind
        start = end

    return top_k_indices


def get_tls_embedding(adata_list, emb_list):
    tls_embedding_dict = {}
    for adata, emb in zip(adata_list, emb_list):
        tls_group_list = np.unique(adata.obs['tls_group'])[1:]  # NO_TLS, TLS_1, TLS_2, ...
        for tls_group in tls_group_list:
            group_indices = adata[adata.obs['tls_group'] == tls_group].obs_names
            group_emb = emb[adata.obs_names.isin(group_indices)]
            group_emb_mean = torch.mean(group_emb, dim=0).reshape(1, -1)
            tls_embedding_dict[tls_group] = group_emb_mean
    return tls_embedding_dict


def plot_tls_umap(tls_embedding_dict, tls_group_dict):
    tls_list, embedding_list = list(tls_embedding_dict.keys()), list(tls_embedding_dict.values())
    group_list = []
    for tls in tls_list:
        for group, group_tls in tls_group_dict.items():
            if tls in group_tls:
                group_list.append(group)
                break

    all_embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
    reducer = umap.UMAP(spread=1.0, min_dist=0.1, random_state=42, metric='cosine')
    embeddings_2d = reducer.fit_transform(all_embeddings)
    palette = sns.color_palette("tab10", n_colors=6)
    group2color = {group_name: palette[i] for i, group_name in enumerate(tls_group_dict.keys())}
    colors = [group2color[group] for group in group_list]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=80, alpha=0.8)
    for i, label in enumerate(tls_list):
        plt.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1], label, fontsize=8)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=group, markerfacecolor=color, markersize=10) 
               for group, color in group2color.items()]
    plt.legend(handles=handles, title="TLS Groups")
    plt.title("UMAP of TLS Points Colored by Group")
    plt.tight_layout()
    plt.show()
