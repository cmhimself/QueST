import networkx as nx
import ot
import scipy
import logging
import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from graphcompass.tl.utils import _get_igraph
from sklearn.metrics.pairwise import laplacian_kernel
from wwl import WeisfeilerLehman, ContinuousWeisfeilerLehman

logger = logging.getLogger(__name__)


def w_dist(label_sequences, sinkhorn=False, categorical=False, sinkhorn_lambda=1e-2, processes=1):
    n = len(label_sequences)
    M = np.zeros((n, n))

    for graph_index_1, graph_1 in enumerate(label_sequences):  # Iterate over pairs of graphs
        labels_1 = label_sequences[graph_index_1]  # Only keep the embeddings for the first h iterations
        for graph_index_2, graph_2 in enumerate(label_sequences[graph_index_1:]):
            labels_2 = label_sequences[graph_index_2 + graph_index_1]
            ground_distance = 'hamming' if categorical else 'euclidean'  # Get cost matrix
            costs = ot.dist(labels_1, labels_2, metric=ground_distance)

            if sinkhorn:
                mat = ot.sinkhorn(np.ones(len(labels_1)) / len(labels_1), np.ones(len(labels_2)) / len(labels_2), costs, sinkhorn_lambda, numItermax=50)
                M[graph_index_1, graph_index_2 + graph_index_1] = np.sum(np.multiply(mat, costs))
            else:
                M[graph_index_1, graph_index_2 + graph_index_1] = ot.emd2([], [], costs, processes=processes)

    M = (M + M.T)
    return M


def pairwise_w_dist(X, node_features=None, num_iterations=3, sinkhorn=False, categorical=True, processes=1):
    logger.debug(f"calling pairwise_w_dist, checking categorical: {categorical}")
    if categorical:
        graph_with_attributes = []
        for i, graph in enumerate(X):
            feat = node_features[i]
            graph.vs['label'] = feat
            graph_with_attributes.append(graph)
        X = graph_with_attributes
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    pairwise_distances = w_dist(node_representations, sinkhorn=sinkhorn, categorical=categorical, sinkhorn_lambda=1e-2, processes=processes)
    return pairwise_distances


def w_wl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None, categorical=True):
    logger.debug("calling w_wl")
    D_W = pairwise_w_dist(X, node_features=node_features, num_iterations=num_iterations, sinkhorn=sinkhorn, categorical=categorical)
    wwl = laplacian_kernel(D_W, gamma=gamma)
    return wwl


def compare(adata: AnnData, library_key: str = "sample", cell_types_keys: list = None, num_iterations: int = 3, categorical=True, processes=1):
    samples = adata.obs[library_key].unique()
    graphs, node_features, cell_types = [], [], []
    adata.uns["wl_kernel"] = {}
    if categorical:
        assert cell_types_keys is not None, "Categorical WWL requires cell types keys!"
        logger.debug("cell_types_keys is not None, use categorical scheme")
        for cell_type_key in cell_types_keys:
            graphs, node_features, status, cell_types = [], [], [], []
            adata.uns["wl_kernel"] = {}
            adata.uns["wl_kernel"][cell_type_key] = {}
            for sample in samples:
                adata_sample = adata[adata.obs[library_key] == sample]
                status.append(adata_sample.obs[library_key][0])
                graphs.append(_get_igraph(adata_sample, cluster_key=None))

                node_features.append(np.array(adata_sample.obs[cell_type_key].values))
                cell_types.append(np.full(len(adata_sample.obs[cell_type_key]), cell_type_key))

            node_features = np.array(node_features)
            wasserstein_distance = pairwise_w_dist(graphs, node_features=node_features, num_iterations=num_iterations, categorical=categorical, processes=processes)

            adata.uns["wl_kernel"][cell_type_key]["wasserstein_distance"] = pd.DataFrame(wasserstein_distance, columns=samples, index=samples)
    else:
        logger.debug("use continuous scheme, defining node features...")
        for sample in samples:
            adata_sample = adata[adata.obs[library_key] == sample]
            graphs.append(_get_igraph(adata_sample, cluster_key=None))
            features = adata_sample.X
            if isinstance(features, scipy.sparse._csr.csr_matrix):
                features = features.toarray()
            node_features.append(np.array(features))

        node_features = np.array(node_features)


        logger.debug("Wasserstein distance between conditions...")
        wasserstein_distance = pairwise_w_dist(graphs, node_features=node_features, num_iterations=num_iterations, categorical=categorical, processes=processes)

        adata.uns["wl_kernel"]["wasserstein_distance"] = pd.DataFrame(wasserstein_distance, columns=samples, index=samples)

    logger.debug("Done!")


def cont_w_wl(adata_list, feature_list=None, num_iterations=3, processes=1):
    graphs, node_features= [], []
    for i, adata in enumerate(adata_list):
        graphs.append(_get_igraph(adata, cluster_key=None))
        feature = feature_list[i] if feature_list is not None else adata.X
        node_features.append(np.array(feature))  # should convert sparse to dense here
    node_features = np.array(node_features)
    wasserstein_distance = pairwise_w_dist(graphs, node_features=node_features, num_iterations=num_iterations, categorical=False, processes=processes)
    return wasserstein_distance


def wl_kernel_subgraph_sim(adata_q, adata_ref, ind_q, ind_ref, cell_type_key='cell_type', library_key='library_id', dataset='DLPFC', query_lib_id='query', ref_lib_id='ref', query_ref=None):
    adata_q_sub = adata_q[ind_q, :]
    adata_ref_sub = adata_ref[ind_ref, :]
    G_ref = nx.from_scipy_sparse_array(adata_ref.obsp['spatial_connectivities'])
    subgraph = G_ref.subgraph(ind_ref)
    adata_q_sub.obs[library_key] = pd.Categorical([query_lib_id] * adata_q_sub.n_obs)
    adata_ref_sub.obs[library_key] = pd.Categorical([ref_lib_id] * adata_ref_sub.n_obs)
    adata_merged = anndata.concat([adata_q_sub, adata_ref_sub], join='outer')
    merged_connectivities = scipy.sparse.block_diag((adata_q_sub.obsp['spatial_connectivities'], nx.adjacency_matrix(subgraph)))
    adata_merged.obsp['spatial_connectivities'] = scipy.sparse.csr_matrix(merged_connectivities)

    compare(adata=adata_merged, library_key=library_key, cell_types_keys=[cell_type_key])
    w_dist_mat = adata_merged.uns['wl_kernel'][cell_type_key]['wasserstein_distance']
    w_dist = w_dist_mat[ref_lib_id].loc[query_lib_id]
    return w_dist


def composition_div(adata_q, adata_ref, ind_q, ind_ref, cell_type_key='cell_type'):
    adata_q_sub = adata_q[ind_q, :]
    adata_ref_sub = adata_ref[ind_ref, :]
    cell_type_1 = np.array(adata_q_sub.obs[cell_type_key])
    cell_type_2 = np.array(adata_ref_sub.obs[cell_type_key])
    vocab = np.unique(np.concatenate((cell_type_1, cell_type_2))).tolist()
    distribution1 = np.zeros(len(vocab))
    distribution2 = np.zeros(len(vocab))
    for item in cell_type_1:
        distribution1[vocab.index(item)] += 1
    for item in cell_type_2:
        distribution2[vocab.index(item)] += 1
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)
    # kl_div = scipy.stats.entropy(prob_dist1, prob_dist2)  
    js_div = scipy.spatial.distance.jensenshannon(distribution1, distribution2, base=2) 
    return js_div


def pearson(sim_vec1, sim_vec2):
    correlation_matrix = np.corrcoef(sim_vec1, sim_vec2)
    pearson_correlation = correlation_matrix[0, 1]
    return pearson_correlation


def sample_sim_wdist(adata1, adata2, sim1, sim2, processes=1):
    wdist_mat = cont_w_wl(adata_list=[adata1, adata2], feature_list=[sim1, sim2], processes=processes)
    wdist = wdist_mat[0, 1]
    return wdist
