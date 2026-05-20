import os
import umap
import time
import scipy
import torch
import random
import zipfile
import logging
import anndata
import threading
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
from tqdm import tqdm
from collections import Counter
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from torch_geometric.utils import k_hop_subgraph
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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
        sc.pl.spatial(adata_shuffle, spot_size=get_spot_size(dataset, adata.uns['library_id']), color='cell_type', show=False)
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
    else:
        logger.info(f"min_count=None, skipping gene-count filter; {adata_list[0].n_vars} genes kept, making adata copies")
    adata_raw_list = [adata.copy() for adata in adata_list]

    if param['hvg'] is not None:
        hvg_list = []
        for adata in adata_list:
            logger.info(f"selecting {param['hvg']} hvg for {adata.uns['library_id']}, time: {get_time_str()}")
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


def prepare_graph_data(adata_ref_list, param):
    logger.info(f"Constructing spatial graph, computing {param['model_k']}-hop subgraph and creating batch labels for each sample! Batch num: {param['batch_num']}")
    feature_list, edge_ind_list, batch_label_list = [], [], []
    sub_node_sample_list, sub_edge_ind_sample_list = [], []
    for i in range(len(adata_ref_list)):
        feature = get_feature(adata_ref_list[i], query=False, param=param, ref_id=adata_ref_list[i].uns['library_id'])

        adj_mat = adata_ref_list[i].obsp['spatial_connectivities'].tocoo()
        edge_index = torch.tensor(np.vstack((adj_mat.row, adj_mat.col)), dtype=torch.int64).to(param['device'])

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

        sample_id = adata_ref_list[i].uns[param['library_key']]
        sub_node_list, sub_edge_ind_list = [], []
        for node_ind in tqdm(range(adata_ref_list[i].n_obs),
                             desc=f"computing {k}-hop subgraph ({sample_id})"):
            sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(node_ind, k, edge_index, relabel_nodes=True)
            sub_node_list.append(sub_nodes)
            sub_edge_ind_list.append(sub_edge_index)
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
    device = param.get('device', 'cuda:0') if param is not None else 'cuda:0'
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
            feature = torch.tensor(adata.obsm['X_pca'], dtype=torch.float32).to(device)
        else:
            logger.info("getting feature: gene expression")
            if scipy.sparse.issparse(adata.X):
                cell_feature = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)
            else:
                cell_feature = torch.tensor(adata.X, dtype=torch.float32).to(device)
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

    umap_model = umap.UMAP(n_neighbors=100, min_dist=1.0, n_components=2, random_state=21, metric='cosine')
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
        end = min(start + chunk_size, N_q)  
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


def get_subgraph_embedding(model, adata_list, feature_list, edge_ind_list, group_key, exclude_values=None):
    """Mean-pool GIN node embeddings within each group defined by adata.obs[group_key].

    Consistent with encode_subgraph() / query niche embedding logic: average node-level
    embeddings (not pre-pooled subgraph embeddings) within each group mask. Used by
    Tutorial 4 (ccRCC, group_key='tls_group') and Tutorial 6 (Perturb-CAST,
    group_key='histo_annotation').

    Args:
        group_key: column name in adata.obs that defines the grouping (e.g. 'tls_group').
        exclude_values: list of group values to skip (e.g. ['NO_TLS'] or ['normal_tissue']).
    Returns:
        dict mapping group value to mean-pooled embedding tensor of shape [1, hidden_dim].
    """
    exclude = set(exclude_values or [])
    embedding_dict = {}
    model.eval()
    with torch.no_grad():
        for adata, feat, edge_ind in zip(adata_list, feature_list, edge_ind_list):
            z_node = model.encoder(feat, edge_ind)
            for group in np.unique(adata.obs[group_key]):
                if group in exclude:
                    continue
                mask = (adata.obs[group_key] == group).values
                embedding_dict[group] = z_node[mask].mean(dim=0, keepdim=True)
    return embedding_dict


def encode_subgraphs_with_genes(model, adata_list, genes, group_key,
                                exclude_values=None, device='cuda:0'):
    """Subset each adata to `genes`, normalize+log1p, build a grid graph, and
    pool node embeddings within each `obs[group_key]` value via model.encode_subgraph.
    Returns dict mapping group value -> tensor."""
    exclude = set(exclude_values or [])
    embedding_dict = {}
    model.eval()
    for adata in adata_list:
        if group_key not in adata.obs.columns: continue
        sub = adata[:, genes].copy()
        sc.pp.normalize_total(sub, target_sum=1e4); sc.pp.log1p(sub)
        if 'spatial_connectivities' not in sub.obsp:
            sq.gr.spatial_neighbors(sub, coord_type='grid')
        adj = sub.obsp['spatial_connectivities'].tocoo()
        edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.int64).to(device)
        X = sub.X.toarray() if hasattr(sub.X, 'toarray') else np.asarray(sub.X)
        feat = torch.tensor(X, dtype=torch.float32).to(device)
        labels = sub.obs[group_key]
        for g in labels.unique():
            if g is None or str(g) == 'nan' or g in exclude: continue
            mask = torch.tensor((labels == g).values, dtype=torch.bool).to(device)
            if mask.sum() == 0: continue
            with torch.no_grad():
                embedding_dict[g] = model.encode_subgraph(feat, edge_index, mask).cpu()
        del feat, edge_index
        torch.cuda.empty_cache()
    return embedding_dict


def plot_tls_umap(tls_embedding_dict, tls_group_dict):
    tls_list, embedding_list = list(tls_embedding_dict.keys()), list(tls_embedding_dict.values())
    group_list = []
    for tls in tls_list:
        for group, group_tls in tls_group_dict.items():
            if tls in group_tls:
                group_list.append(group)
                break

    all_embeddings = torch.cat(embedding_list, dim=0).cpu().numpy()
    reducer = umap.UMAP(spread=1.0, min_dist=0.1, random_state=21, metric='cosine')
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


# bench utils
color_simulation = {
    'B-cells': '#ff7f0e',
    'Cancer/Epithelial': '#d62728',
    'Endothelial': '#d3ede0',
    'Monocytes/Macrophages': '#ece0fc',
    'T-cells': '#4cb0d3',
    'T-edge': '#008080',
    'T-core': '#ff8744',
    'B-edge': '#89ca78',
    'B-edge-1': '#89ca78',
    'B-edge-2': '#ef596f',
    'B-edge-3': '#61afef',
    "Else": "#bebebe", 
}

color_palette_simu_celltype = {
    'B-cells': '#ff7f0e',
    'Cancer/Epithelial': '#d62728',
    'Endothelial': '#d3ede0',
    'Monocytes/Macrophages': '#ece0fc',
    'T-cells': '#4cb0d3',
}


color_dlpfc = {
    'Layer1': '#9467bd',
    'Layer2': '#8c564b',
    'Layer3': '#1f77b4',
    'Layer4': '#ff7f0e',
    'Layer5': '#2ca02c',
    'Layer6': '#d62728',
    'WM':     '#e377c2',
    'Else':   '#440256'
}

color_mobt_manual = {
    'ONL':  '#1f77b4',
    'GL':   '#ff7f0e',
    'GCL':  '#2ca02c',
    'EPL':  '#d62728',
    'MCL':  '#e377c2',
    'Else': '#440256'
}

color_mobt_auto = {
    'EPL':  '#1f77b4',
    'GCL':  '#ff7f0e',
    'GL':   '#2ca02c',
    'MCL':  '#d62728',
    'ONL':  '#9467bd',
    'Else': '#440256'
}

color_mobt = {
    'EPL':  '#1f77b4',
    'ONL':  '#ff7f0e',
    'GL':   '#2ca02c',
    'MCL':  '#d62728',
    'GCL':  '#9467bd',
    'Else': '#440256'
}





color_nsclc = {}


def check_raw_batch(adata_list, batch_key='batch'):
    combined_adata = adata_list[0].concatenate(*adata_list[1:], batch_key=batch_key)

    # Normalize and scale the combined data
    print("normalizing data")
    sc.pp.normalize_total(combined_adata)
    sc.pp.log1p(combined_adata)
    sc.pp.scale(combined_adata)

    print("performing pca")
    sc.tl.pca(combined_adata, n_comps=50)

    # Perform UMAP visualization
    print("finding neighbors")
    sc.pp.neighbors(combined_adata, use_rep='X_pca')
    print("computing umap")
    sc.tl.umap(combined_adata)
    sc.pl.umap(combined_adata, color=batch_key, title="UMAP of Raw Data")


def get_time_str():
    t = time.localtime()
    time_str = f"{t.tm_year}/{t.tm_mon:02d}/{t.tm_mday:02d} {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
    return time_str


def sc_preprocess(adata, top=None, subset=True):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=top, subset=subset)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50)
    return adata


def plot_all_samples(sample_names, adata_list, nrows=3, ncols=4, figsize=(20, 15), spot_size=5, key='cell_type', invert_y=True, spot_size_list=None):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)  # Adjust fig size as needed
    axs = axs.flatten()  # Flatten the array of axes for easy iteration
    if spot_size_list is None:
        spot_size_list = [spot_size for _ in range(len(sample_names))]

    spot_size_ind = 0
    for ax, adata, sample_id in zip(axs, adata_list, sample_names):
        sc.pl.spatial(adata, color=key, ax=ax, spot_size=spot_size_list[spot_size_ind], show=False)
        # sq.pl.spatial_scatter(adata, shape=None, color=key, ax=ax, size=spot_size)
        if invert_y:
            ax.invert_yaxis()
        ax.set_title(sample_id)
        spot_size_ind += 1

    plt.tight_layout()  # Adjust spacing between plots
    plt.show()


def plot_sample(adata, key='cell_type', spot_size=5, save_path=None, invert_y=True, library_key=None):
    # sc.pl.spatial(adata, color=key, spot_size=spot_size, show=False)
    axs = sq.pl.spatial_scatter(adata, color=key, size=spot_size, shape=None, library_key=library_key, return_ax=True)
    if invert_y:
        # plt.gca().invert_yaxis()
        for ax in axs:
            ax.invert_yaxis()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_spot(adata, spot_ind_list, spot_name='niche'):
    adata.obs[spot_name] = [spot_name if i in spot_ind_list else 'Else' for i in range(adata.shape[0])]
    sc.pl.spatial(adata, color=spot_name, spot_size=5, show=False)
    plt.gca().invert_yaxis()
    plt.show()


def visualize_all_sample_niche(adata, niche_name, ax):
    sc.pl.spatial(adata, color=niche_name, ax=ax, show=False)
    ax.set_title(niche_name)


def visualize_subgraph(adata, subgraph_nodes, spot_size=10, subgraph_key='subgraph', invert_y=False):
    adata.obs[subgraph_key] = ['subgraph' if i in subgraph_nodes else 'else' for i in range(adata.shape[0])]
    sq.pl.spatial_scatter(adata, color=subgraph_key, size=spot_size, shape=None)
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()


def visualize_niche(adata, niche_ind, niche_name, cell_type_key='cell_type', spot_size=5, invert_y=False, title=None, save_path=None, show=True, dataset=None):
    # if f'{niche_name}_cell_type' not in adata.obs.columns:
    #     adata.obs[f'{niche_name}_cell_type'] = pd.Categorical([adata.obs[cell_type_key][i] if i in niche_ind else 'Else' for i in range(adata.n_obs)])
    adata.obs[f'{niche_name}_cell_type'] = pd.Categorical([adata.obs[cell_type_key][i] if i in niche_ind else 'Else' for i in range(adata.n_obs)])
    if dataset is None:
        sc.pl.spatial(adata, color=f'{niche_name}_cell_type', spot_size=spot_size, show=False, title=niche_name if title is None else title)
    elif dataset == "DLPFC":
        palette = color_dlpfc
        sc.pl.spatial(adata, color=f'{niche_name}_cell_type', spot_size=spot_size, show=False, title=niche_name if title is None else title, palette=palette)
    else:
        assert False
    if invert_y:
        plt.gca().invert_yaxis()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
        show_plot_with_timeout(5)


def visualize_graph(adata, G, title='Spatial Graph'):
    spatial_coords = adata.obsm['spatial']
    connectivities = nx.adjacency_matrix(G)
    if scipy.sparse.issparse(connectivities):
        connectivities = connectivities.toarray()

    plt.figure(figsize=(10, 10))
    plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], s=1)  # s controls the size of the points

    # Draw lines between each point and its neighbors
    # Note: For large datasets, you might want to subsample or only plot a fraction of the edges for clarity
    n_cells = spatial_coords.shape[0]
    for i in range(n_cells):
        neighbors = connectivities[i].nonzero()[0]  # Indices of neighbors for cell i
        for neighbor in neighbors:
            plt.plot([spatial_coords[i, 0], spatial_coords[neighbor, 0]],
                     [spatial_coords[i, 1], spatial_coords[neighbor, 1]], c='gray', alpha=0.5)
    plt.title(title)
    plt.xlabel('Spatial coordinate 1')
    plt.ylabel('Spatial coordinate 2')
    plt.show()


def prune_subgraph_to_n(subgraph, start_node, n):
    bfs_levels = nx.single_source_shortest_path_length(subgraph, start_node)  # Get all nodes in the subgraph sorted by their BFS level from start_node
    # print("in pruning, checking subgraph size:", len(subgraph.nodes))
    # print("in pruning, checking bfs level size:", len(bfs_levels))

    subgraph_nodes_sorted_by_level = sorted(
        [(node, level) for node, level in bfs_levels.items() if node in subgraph.nodes()],
        key=lambda x: x[1],
    )  # sort subgraph nodes by descending bfs level, no reverse
    nodes_to_keep = set()
    for node, level in subgraph_nodes_sorted_by_level:
        if len(nodes_to_keep) < n:
            nodes_to_keep.add(node)
        else:
            break

    pruned_subgraph = subgraph.subgraph(nodes_to_keep)
    # if len(pruned_subgraph.nodes) < n:
    #     print("******", len(pruned_subgraph.nodes), n), exit()
    return pruned_subgraph


def prune_edges_by_distance(G, radius):
    edges_to_remove = []

    for edge in G.edges():
        node1, node2 = edge
        pos1 = G.nodes[node1]['pos']
        pos2 = G.nodes[node2]['pos']
        distance = np.linalg.norm(pos1 - pos2)
        if distance > radius:
            edges_to_remove.append(edge)

    G.remove_edges_from(edges_to_remove)


# Update the main function call to include G in prune_subgraph_to_n, as it requires full graph context
def find_subgraph(G, start_node_list, num_limit, cell_type_list, cell_type_key='cell_type', adata=None):
    logger.debug(f"finding subgraph of size {num_limit}")
    start_node = start_node_list[0]
    k = 1
    while True:
        subgraph = nx.ego_graph(G, start_node, radius=k, undirected=True, center=True)
        # visualize_subgraph(adata, subgraph.nodes, invert_y=True)
        nodes_to_remove = [n for n in subgraph.nodes if subgraph.nodes[n][cell_type_key] not in cell_type_list]
        subgraph.remove_nodes_from(nodes_to_remove)  # Filter nodes by type
        # visualize_subgraph(adata, subgraph.nodes, invert_y=True)
        if len(subgraph) > num_limit:
            subgraph = prune_subgraph_to_n(subgraph, start_node, num_limit)
            return subgraph
        elif len(subgraph) == num_limit:
            return subgraph
        k += 1


def find_junction(G, cell_type_list, cell_type_key='cell_type'):
    candidates = [n for n in G.nodes if G.nodes[n][cell_type_key] in cell_type_list]
    junction = []
    for n in candidates:
        neighbor_cell_types = [G.nodes[neighbor][cell_type_key] for neighbor in G.neighbors(n)]
        if all(cell_type in neighbor_cell_types for cell_type in cell_type_list):
            junction.append(n)
    return junction


def find_spots_in_ellipse(adata, anchor_list, cell_type_list, start, step, num_limit):
    ind1, ind2 = anchor_list
    pos = np.array(adata.obsm['spatial'])
    focus1, focus2 = pos[ind1], pos[ind2]
    focal_length = start
    while True:
        logger.debug(f"current focal length: {focal_length}")
        major_axis_length = 2 * focal_length
        ind_dist = []  # Temporary storage for indices and distances for sorting later

        for i, point in enumerate(pos):
            if adata.obs['cell_type'][i] in cell_type_list and np.linalg.norm(point - focus1) + np.linalg.norm(point - focus2) <= major_axis_length:
                distance_to_ind1 = np.linalg.norm(point - focus1)  # Distance from point to focus1 for pruning
                ind_dist.append((i, distance_to_ind1))
        logger.debug(f"current point in ellipse: {len(ind_dist)}")

        if len(ind_dist) > num_limit:
            logger.debug("exceeding num limit, pruning")
            ind_dist.sort(key=lambda x: x[1], reverse=False)  # Sort by distance to focus1 (ind1) for pruning
            ind_in_ellipse = [idx for idx, _ in ind_dist[:num_limit]]  # Prune to meet num_limit by removing points farthest from ind1
            return ind_in_ellipse
        elif len(ind_dist) == num_limit:
            ind_in_ellipse = [idx for idx, _ in ind_dist]  # Exactly matches num_limit
            return ind_in_ellipse

        focal_length += step  # Increment focal length for next iteration


def construct_adata_spatial_graph(adata, dataset, library_key=None):
    if dataset == "ccRCC-NSCLC":
        if adata.uns['library_id'].startswith('nsclc'):
            dataset = 'nsclc'
        else:
            dataset = "DLPFC"
    if dataset == "DLPFC":
        sq.gr.spatial_neighbors(adata, coord_type='grid', library_key=library_key)
    elif dataset == 'MouseOlfactoryBulbTissue':
        if adata.uns['library_id'] in ['10X', 'Stereo-seq']:
            sq.gr.spatial_neighbors(adata, coord_type='grid', library_key=library_key)
        elif adata.uns['library_id'] == 'Slide-seq V2':
            sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 100), library_key=library_key)
            # G = nx.from_scipy_sparse_array(adata.obsp['spatial_connectivities'])
            # visualize_graph(adata, G, title="delaunay radius 120"), exit()
    elif dataset == 'MERFISH':
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, library_key=library_key)  # TODO: select radius
    elif dataset == "Simulation":
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 5), library_key=library_key)
    elif dataset == "nsclc":
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=(0, 300), library_key=library_key)
        # G = nx.from_scipy_sparse_array(adata.obsp['spatial_connectivities'])
        # visualize_graph(adata, G, title="delaunay radius 5"), exit()
    else:
        assert False, f"Unknown dataset {dataset}!"


def construct_adata_list_spatial_graph(adata_list, dataset, library_key=None):
    for adata in adata_list:
        construct_adata_spatial_graph(adata, dataset, library_key=library_key)


def construct_merged_graph(adata, dataset, library_key='library_id', query_ref=None, adata_query_ref_list=None):
    if dataset == 'DLPFC':
        sq.gr.spatial_neighbors(adata, library_key=library_key, coord_type='grid')
    elif dataset == 'MERFISH':
        sq.gr.spatial_neighbors(adata, library_key=library_key, coord_type='generic', delaunay=True)
    elif dataset == 'MouseOlfactoryBulbTissue':
        assert len(query_ref) == 2 and len(adata_query_ref_list) == 2
        query_id, ref_id = query_ref
        assert query_id == 'Stereo-seq', query_id
        adata_q, adata_ref = adata_query_ref_list
        sq.gr.spatial_neighbors(adata_q, library_key=library_key, coord_type='grid')
        connectivities_q = adata_q.obsp['spatial_connectivities']
        if ref_id == '10X':
            sq.gr.spatial_neighbors(adata_ref, library_key=library_key, coord_type='grid')
        elif ref_id == 'Slide-seq V2':
            sq.gr.spatial_neighbors(adata_ref, library_key=library_key, coord_type='generic', delaunay=True)
        else:
            assert False, 'Unknown ref data!'
        connectivities_ref = adata_ref.obsp['spatial_connectivities']
        combined_connectivities = scipy.sparse.block_diag((connectivities_q, connectivities_ref))
        adata_merged = anndata.concat([adata_q, adata_ref], join='outer')
        adata_merged.obsp['spatial_connectivities'] = scipy.sparse.csr_matrix(combined_connectivities)  # metric only use this
        return adata_merged
    else:
        assert False, f'Unknown dataset {dataset} for graph constructing!'


def show_plot():
    plt.show()


def show_plot_with_timeout(timeout):
    thread = threading.Thread(target=show_plot)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print("plt.show() timed out!")
    plt.close()


def save_project_code(source_dir, output_zip, save_logger=None):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the source directory
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Write file to zip archive, using relative path
                    zipf.write(file_path, os.path.relpath(file_path, source_dir))

    if save_logger is not None:
        save_logger.info(f"saving source code to {output_zip}")


# =====================================================================
# Tutorial helpers: spot-level niche analysis (Perturb-CAST and similar)
# =====================================================================

def get_global_niche_adata(trainer, group_key='histo_annotation',
                           exclude_values=('normal_tissue',), min_count=3):
    """Extract per-spot niche embeddings from a trained QueSTTrainer and
    concatenate slices into one AnnData. Spots whose `obs[group_key]` is in
    `exclude_values`, or whose group has fewer than `min_count` members, are
    dropped. Niche embedding stored at `obsm['X_quest']`."""
    import torch
    import anndata as ad
    exclude = set(exclude_values or [])
    parts = []
    trainer.model.eval()
    with torch.no_grad():
        for adata, feat, edge, sub_node, sub_edge in zip(
                trainer.adata_list, trainer.feature_list, trainer.edge_ind_list,
                trainer.sub_node_sample_list, trainer.sub_edge_ind_sample_list):
            _, z_niche = trainer.model.get_subgraph_rep(feat, edge, sub_node, sub_edge)
            emb = z_niche.cpu().numpy()
            annots = adata.obs[group_key].values
            valid = np.zeros(len(adata), dtype=bool)
            for v in np.unique(annots):
                if v in exclude:
                    continue
                m = annots == v
                if m.sum() >= min_count:
                    valid |= m
            sub = adata[valid].copy()
            sub.obsm['X_quest'] = emb[valid]
            sub.obs['slice'] = adata.uns['library_id']
            parts.append(sub)
            logger.info(f"{adata.uns['library_id']}: {valid.sum()}/{len(adata)} spots kept")
    return ad.concat(parts, merge='same')


def load_pretrained_model(ckpt_path, in_dim, batch_num=1, model_k=3, device='cuda:0'):
    """Load a QueST model with the paper's canonical architecture.
    Returns `(model, model_param)`; pass `model_param` to `prepare_graph_data`."""
    from quest.model import QueST
    model_param = {
        'device': device, 'library_key': 'library_id', 'model_k': model_k,
        'enc_dims': [2048, 256, 32], 'dec_dims': [32],
        'batch_disc_dims': [32, 32, 16], 'dec_batch_dim': 2, 'batch_num': batch_num,
        'recon_loss': 'mse', 'residual': True,
        'norm': 'batchnorm', 'dropout': 0.1, 'activation': 'relu',
        'lr_gen': 0.001, 'lr_disc': 0.005, 'weight_decay': 5e-4,
        'lbd_positive': 1, 'lbd_negative': 1, 'lbd_gen': 1,
    }
    model = QueST(in_dim=in_dim, param=model_param, logger=logger).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, model_param


def wrap_subgraph_dict(emb_dict, group_dict=None):
    """Wrap a {key: emb} dict from `get_subgraph_embedding` into an AnnData with
    `obsm['X_quest']`. If `group_dict` (group_name -> list of keys) is given,
    add `obs['group']`."""
    import anndata as ad
    keys = list(emb_dict.keys())
    X = torch.cat([emb_dict[k] for k in keys], dim=0).cpu().numpy()
    obs = pd.DataFrame(index=[str(k) for k in keys])
    if group_dict is not None:
        m = {k: g for g, klist in group_dict.items() for k in klist}
        obs['group'] = [m.get(k, 'unassigned') for k in keys]
    adata = ad.AnnData(X=np.zeros((len(keys), 1), dtype='float32'), obs=obs)
    adata.obsm['X_quest'] = X
    return adata


def plot_niche_umap(adata, color, emb_key='X_quest', umap_key=None,
                    figsize=(6, 5), title=None, frameon=False,
                    edge_color=None, linewidths=None,
                    equal_aspect=False, hide_ticks=False, hide_spines=False,
                    invert_x=False, invert_y=False,
                    **plot_kwargs):
    """UMAP scatter colored by `color`. Use `umap_key` to plot pre-computed
    coordinates; otherwise compute UMAP from `adata.obsm[emb_key]`."""
    if umap_key is not None and umap_key in adata.obsm:
        basis = umap_key
    else:
        coords = umap.UMAP(spread=1.0, min_dist=0.1, random_state=42, metric='cosine').fit_transform(adata.obsm[emb_key])
        adata.obsm['X_umap'] = coords
        basis = 'X_umap'
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.embedding(adata, basis=basis, color=color, frameon=frameon,
                    title=title or color.capitalize(), ax=ax, show=False, **plot_kwargs)
    if edge_color is not None or linewidths is not None:
        for coll in ax.collections:
            if edge_color is not None: coll.set_edgecolor(edge_color)
            if linewidths is not None: coll.set_linewidths(linewidths)
    if equal_aspect: ax.set_aspect('equal')
    if hide_ticks: ax.set_xticks([]); ax.set_yticks([])
    if hide_spines:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if invert_x: ax.invert_xaxis()
    if invert_y: ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def get_niche_pattern(adata, n_neighbors=30, resolution=0.2, random_state=42,
                     emb_key='X_quest', key_added='leiden'):
    """L2-normalise `adata.obsm[emb_key]` into `obsm['X_niche']`, then run
    cosine `sc.pp.neighbors` and `sc.tl.leiden`. Cluster id at `obs[key_added]`."""
    from sklearn.preprocessing import normalize as l2norm
    adata.obsm['X_niche'] = l2norm(adata.obsm[emb_key])
    sc.pp.neighbors(adata, use_rep='X_niche', n_neighbors=n_neighbors, metric='cosine')
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, key_added=key_added)


def compute_npr_score(target_full, global_adata, target_slice,
                      leiden_key='leiden', min_local=10, high_threshold=3,
                      score_key='npr_score'):
    """Niche Prevalence Ratio: NPR(c) = (n_c_global/N_global) / (n_c_local/N_local).
    Clusters with fewer than `min_local` local spots get score 0. A spatial
    filter then drops isolated high-score spots. Non-tumor spots get 0.
    Writes the filtered scores to `target_full.obs[score_key]`; returns the
    per-cluster (unfiltered) NPR dict."""
    from scipy.spatial import cKDTree
    g = global_adata
    mask = g.obs['slice'] == target_slice
    key2leiden = dict(zip(zip(g.obs.loc[mask, 'array_row'],
                              g.obs.loc[mask, 'array_col']),
                          g.obs.loc[mask, leiden_key]))
    local_labels = np.array([key2leiden.get((r, c), '-1') for r, c
                              in zip(target_full.obs['array_row'],
                                     target_full.obs['array_col'])])
    is_tumor = local_labels != '-1'
    n_local, n_global = is_tumor.sum(), len(g)
    global_labels = g.obs[leiden_key].values
    cl_score = {}
    for cl in np.unique(local_labels):
        if cl == '-1':
            continue
        n_local_cl = (local_labels == cl).sum()
        n_global_cl = (global_labels == cl).sum()
        local_frac = n_local_cl / n_local
        global_frac = n_global_cl / n_global
        cl_score[cl] = (global_frac / local_frac) if (local_frac > 0 and n_local_cl >= min_local) else 0
    scores = np.array([cl_score.get(lab, 0) for lab in local_labels])
    coords = target_full.obsm['spatial']
    tree = cKDTree(coords)
    _, dists = tree.query(coords, k=2)
    radius = np.median(dists[:, 1]) * 1.5
    scores_filtered = scores.copy()
    for i in np.where(scores > high_threshold)[0]:
        nbrs = tree.query_ball_point(coords[i], radius)
        n_high = sum(scores[j] > high_threshold for j in nbrs if j != i)
        if n_high < 3:
            scores_filtered[i] = 0
    target_full.obs[score_key] = scores_filtered
    return cl_score


def score_query_by_groups(model, model_param, adata_list, query_embeddings,
                          group_dict, prefix='score_', device='cuda:0'):
    """For each adata, write `obs[prefix + group]` = mean cosine similarity of
    per-spot niche embeddings to that group's query embeddings (in-place)."""
    model.eval()
    for adata in adata_list:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
        feat_list, ei_list, sn_list, se_list, _ = prepare_graph_data([adata], model_param)
        with torch.no_grad():
            _, z_niche = model.get_subgraph_rep(feat_list[0], ei_list[0], sn_list[0], se_list[0])
        z_niche_norm = F.normalize(z_niche, p=2, dim=1)
        for grp, keys in group_dict.items():
            present = [k for k in keys if k in query_embeddings]
            if not present:
                adata.obs[f'{prefix}{grp}'] = 0.0
                continue
            stack = torch.cat([query_embeddings[k].unsqueeze(0) for k in present], dim=0).to(device)
            sim = z_niche_norm @ F.normalize(stack, p=2, dim=1).T
            adata.obs[f'{prefix}{grp}'] = sim.mean(dim=1).cpu().numpy()


def compute_queried_tls_region_labels(adata, score_cols, top_pct=10, key_added='queried_tls_region'):
    scores = adata.obs[score_cols].values
    agg_max = scores.max(axis=1)
    threshold = np.percentile(agg_max, 100 - top_pct)
    is_tls = agg_max >= threshold
    groups = [c.replace('score_', '') for c in score_cols]
    labels = np.array(['Else'] * len(adata), dtype=object)
    argmax_idx = scores.argmax(axis=1)
    for gi, g in enumerate(groups):
        labels[is_tls & (argmax_idx == gi)] = f'Type {g}'
    cats = ['Else'] + [f'Type {g}' for g in groups]
    adata.obs[key_added] = pd.Categorical(labels, categories=cats)


def plot_queried_tls_region_spatial(adata_list, key='queried_tls_region', palette=None, figsize=(7, 5)):
    if palette is None:
        palette = {'Else':    '#eeeeee',
                   'Type A':  '#d7263d',
                   'Type B1': '#39c6d6',
                   'Type B2': '#23ce6b'}
    fig, axes = plt.subplots(1, len(adata_list),
                             figsize=(figsize[0] * len(adata_list), figsize[1]))
    if len(adata_list) == 1:
        axes = [axes]
    for ax, adata in zip(axes, adata_list):
        sid = adata.uns.get('library_id', '')
        sc.pl.spatial(adata, color=key, palette=palette, ax=ax, show=False,
                      title=sid, frameon=False, spot_size=75)
    plt.tight_layout()
    plt.show()


def plot_npr_spatial(adata, score_key='npr_score', title=None, figsize=(6, 6)):
    """Spatial scatter of `obs[score_key]` with the canonical purple-yellow colormap."""
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'npr', ['#210c52', '#217b9c', '#22b5af', '#ebec23'])
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(adata, color=score_key, cmap=cmap,
                  vmin=0, vmax=max(adata.obs[score_key].max(), 1.0),
                  ax=ax, show=False, title=title or score_key, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_noi_query(adata_list, sample_ids, global_adata, noi_spots,
                    leiden_key='leiden', palette=None, figsize=(36, 6)):
    """Auto-infer the niche pattern that `noi_spots` mostly belong to, then                                                              
    project all spots of that pattern onto every slice spatially.          
                                                                                                                                        
    Each spot is categorized as Queried (in target niche pattern),
    Residual (sharing a nodule with a Queried spot), or Other."""                                                                        
    import pandas as pd                                                                                                                  
    from matplotlib.lines import Line2D                                                                                                  
                                                                                                                                        
    pattern_lookup = {(r['slice'], r['array_row'], r['array_col']): r[leiden_key]                                                        
                    for _, r in global_adata.obs.iterrows()}                                                                           
                                                            
    # Auto-infer target niche pattern: the one most NOI spots belong to                                                                  
    noi_patterns = [pattern_lookup.get(k) for k in noi_spots]                                                                            
    noi_patterns = [p for p in noi_patterns if p is not None]
    target_pattern = pd.Series(noi_patterns).value_counts().idxmax()                                                                     
                                                                                                                                        
    # All nodules (across every slice) that contain at least one Queried spot
    involved_nodules = set()                                                                                                             
    for sid, ad in zip(sample_ids, adata_list):
        for _, r in ad.obs.iterrows():                                                                                                   
            if pattern_lookup.get((sid, r['array_row'], r['array_col'])) == target_pattern:
                involved_nodules.add((sid, r['histo_annotation']))                                                                       
                                                                
    if palette is None:                                                                                                                  
        palette = {'Other': '#4d3d75', 'Residual': '#4ec4bf', 'Queried': '#eff04f'}
                                                                                                                                        
    fig, axes = plt.subplots(1, len(sample_ids), figsize=figsize)                                                                        
    for i, (sid, ad) in enumerate(zip(sample_ids, adata_list)):  
        status = []                                                                                                                      
        for _, r in ad.obs.iterrows():                                                                                                   
            p = pattern_lookup.get((sid, r['array_row'], r['array_col']))
            if p == target_pattern:                                                                                                      
                status.append('Queried')
            elif (sid, r['histo_annotation']) in involved_nodules:                                                                       
                status.append('Residual')                         
            else:                                                                                                                        
                status.append('Other')
        ad.obs['NOI'] = pd.Categorical(status, categories=['Other', 'Residual', 'Queried'])
        sc.pl.spatial(ad, color='NOI', palette=palette, ax=axes[i], show=False,                                                          
                    title=f'{sid}', frameon=False, legend_loc='none')
                                                                                                                                        
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)                                         
                for l, c in [(f'Queried (niche pattern {target_pattern})', palette['Queried']),                                           
                            ('Residual', palette['Residual']),                                                                           
                            ('Other', palette['Other'])]]                                                                                
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11, framealpha=0.8)                                                 
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])                                                                                            
    plt.show()                            
    