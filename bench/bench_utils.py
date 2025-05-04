import logging
import threading
import time
import zipfile
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import anndata
import squidpy as sq
logger = logging.getLogger(__name__)


color_simulation = {
    "B-cells": "#ff7f0e", 
    "Cancer/Epithelial": "#d62728", 
    "Endothelial": "#279e68", 
    "T-cells": "#1f77b4",
    "Monocytes/Macrophages": "#aa40fc",
    "Type-1": "#008080",
    "Type-2": "#ff8744",
    "Else": "#bebebe", 
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

color_sim_tme = {
    'B-cells': '#1f77b4',
    'Cancer/Epithelial': '#ff7f0e',
    'Endothelial': '#2ca02c',
    'Monocytes/Macrophages': '#d62728',
    'T-cells': '#9467bd'
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

