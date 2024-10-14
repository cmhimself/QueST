import numpy as np
import random
import logging
import pandas as pd
import squidpy as sq
import networkx as nx
import bench_utils as utils
from collections import Counter
from scipy.spatial import KDTree
logger = logging.getLogger(__name__)


class NicheGenerator:
    def __init__(self, adata_list, sample_names: np.array, cell_type_key, spot_size=1):
        self.adata_list = adata_list
        self.sample_names = sample_names
        self.cell_type_key = cell_type_key
        self.spot_size = spot_size
        self.adata_q = self.adata_list[0]
        self.niche_name = None
        self.G = None

    def check_exist(self, cell_type_list):
        """to make sure these cell types exist in all samples"""
        logger.debug("checking cell type existence")
        for cell_type in cell_type_list:
            for i, sample_id in enumerate(self.sample_names):
                assert cell_type in self.adata_list[i].obs[self.cell_type_key].cat.categories, f"{cell_type} not found in {sample_id}!"

    def save_niche_res(self, niche_ind, visualize=False, invert_y=False, title=None):
        logger.info("saving niche res to adata")
        self.adata_q.obs[f'{self.niche_name}_niche'] = pd.Categorical(['Niche' if i in niche_ind else 'Else' for i in range(self.adata_q.shape[0])])
        self.adata_q.obs[f'{self.niche_name}_cell_type'] = pd.Categorical([self.adata_q.obs[self.cell_type_key][i] if i in niche_ind else 'Else' for i in range(self.adata_q.shape[0])])
        logger.info(self.adata_q)
        if visualize:
            utils.visualize_niche(self.adata_q, niche_ind, self.niche_name, cell_type_key=self.cell_type_key, spot_size=self.spot_size, invert_y=invert_y, title=title)

    def check_candidate(self, candidate_ind, knn_ind, cell_type_list=None, min_sum_ratio=1, min_single_ratio=0):
        """originate from nsclc benchmark"""
        neighbor_ind = knn_ind[candidate_ind, :]
        neighbor_cell_types = np.array(self.adata_q.obs[self.cell_type_key][neighbor_ind])
        cell_type_unique, counts = np.unique(neighbor_cell_types, return_counts=True)
        if not all(cell_type in cell_type_unique for cell_type in cell_type_list):
            return False, None
        proportions = counts / len(neighbor_ind)
        proportion_dict = dict(zip(cell_type_unique, proportions))
        sum_proportion = 0
        flag = True
        for cell_type_required in cell_type_list:
            single_ratio = proportion_dict[cell_type_required]
            if single_ratio < min_single_ratio:
                flag = False
                break
            sum_proportion += proportion_dict[cell_type_required]
        if sum_proportion < min_sum_ratio:
            flag = False
        return flag, neighbor_ind

    def construct_query_spatial_graph(self, coord_type='generic', delaunay=True, radius=None, visualize=False):
        if coord_type == 'generic':
            assert delaunay
            logger.info("building delaunay graph")
            sq.gr.spatial_neighbors(self.adata_q, coord_type=coord_type, delaunay=True)  # grid graph
        elif coord_type == 'grid':
            logger.info("building grid graph")
            sq.gr.spatial_neighbors(self.adata_q, coord_type=coord_type)
        self.G = nx.from_scipy_sparse_array(self.adata_q.obsp['spatial_connectivities'])
        cell_type_dict = {i: cell_type for i, cell_type in enumerate(self.adata_q.obs[self.cell_type_key])}
        pos_dict = {i: pos for i, pos in enumerate(self.adata_q.obsm['spatial'])}
        nx.set_node_attributes(self.G, cell_type_dict, self.cell_type_key)
        nx.set_node_attributes(self.G, pos_dict, 'pos')
        if radius is not None:
            utils.prune_edges_by_distance(self.G, radius=radius)
        if visualize:
            utils.visualize_graph(self.adata_q, self.G)

    def get_niche_with_composition(self, cell_type_list=None, num_cell=50, min_sum_ratio=1, min_single_ratio=0):
        pos = np.array(self.adata_q.obsm['spatial'])
        tree = KDTree(pos)  # Create a KDTree for efficient nearest neighbor search
        knn_dist, knn_ind = tree.query(pos, num_cell)  # Query the k nearest neighbors for each point (including the point itself)
        candidates_list = np.where(self.adata_q.obs[self.cell_type_key].isin(cell_type_list))[0]  # all the ind with given cell type
        logger.info(f"identified {len(candidates_list)} candidates")

        anchor_niche_list = []
        for i in candidates_list:
            flag, neighbor_ind = self.check_candidate(i, knn_ind, cell_type_list, min_sum_ratio, min_single_ratio)
            if flag:
                anchor_niche_list.append((i, neighbor_ind))
        assert len(anchor_niche_list) > 0, "no anchor niche identified!"
        logger.info(f"identified {len(anchor_niche_list)} anchor niches")
        return anchor_niche_list

    def check_niche_composition(self, niche_ind):
        neighbor_cell_types = np.array(self.adata_q.obs[self.cell_type_key][niche_ind])
        cell_type_unique, counts = np.unique(neighbor_cell_types, return_counts=True)
        proportions = counts / len(niche_ind)
        proportion_dict = dict(zip(cell_type_unique, proportions))
        logger.info(f"checking chosen niche composition: {proportion_dict}")


class DlpfcGenerator(NicheGenerator):
    def define_anchor(self, cell_type_list, adata):
        num_type = len(cell_type_list)
        logger.info(f"defining anchor with {num_type} types")

        if num_type == 1:
            anchor_ind = np.where(adata.obs[self.cell_type_key] == cell_type_list[0])[0]  
        elif num_type == 2:
            margin_node1_2 = utils.find_junction(self.G, cell_type_list)
            anchor_ind = [random.choice(margin_node1_2)]
        elif num_type == 3:
            pos = np.array(adata.obsm['spatial'])
            cell_type1, cell_type2, cell_type3 = cell_type_list
            margin_node1_2 = utils.find_junction(self.G, [cell_type1, cell_type2])
            margin_node2_3 = utils.find_junction(self.G, [cell_type2, cell_type3])
            anchor_ind1 = random.choice(margin_node1_2)
            distances = np.linalg.norm(pos[margin_node2_3] - pos[anchor_ind1], axis=1)
            anchor_ind2 = margin_node2_3[np.argmin(distances)]  # Find the index in margin_nodes corresponding to the smallest distance
            anchor_ind = [anchor_ind1, anchor_ind2]
        else:
            assert False, "too many cell types"
        return anchor_ind

    def expand_anchor(self, adata, cell_type_list, anchor_list, num_cell):
        logger.info("expanding anchor")
        if len(cell_type_list) <= 2:
            subgraph = utils.find_subgraph(self.G, anchor_list, num_cell, cell_type_list)
            expanded_ind = list(subgraph.nodes)
        elif len(cell_type_list) == 3:
            expanded_ind = utils.find_spots_in_ellipse(adata, anchor_list, cell_type_list, start=1, step=1, num_limit=num_cell)
        else:
            assert False, "too many cell types"
        return expanded_ind

    def define_niche(self, sample_id='151507', cell_type_list=['Layer4'], num_cell=50):
        self.niche_name = f"{'_'.join(cell_type_list)}_{str(num_cell)}"
        logger.info(f"defining niche: {self.niche_name}")
        self.check_exist(cell_type_list)
        assert isinstance(self.sample_names, np.ndarray)
        adata_ind = np.where(self.sample_names == sample_id)[0][0]  # np.where: (array([0]),)
        self.adata_q = self.adata_list[adata_ind]

        self.construct_query_spatial_graph(coord_type='grid', radius=None, visualize=False)

        anchor_list = self.define_anchor(cell_type_list, self.adata_q)
        niche_ind = self.expand_anchor(self.adata_q, cell_type_list, anchor_list, num_cell)

        logger.info(f"checking niche composition: {Counter(self.adata_q.obs[self.cell_type_key][niche_ind])}")
        self.save_niche_res(niche_ind)

    def metric(self):
        pass
    

class MouseOlfactoryBulbTissueGenerator(NicheGenerator):
    def define_niche(self, sample_id='slidev2_filtered', cell_type_list=None, num_cell=50, min_sum_ratio=1, min_single_ratio=0):
        self.niche_name = f"{'_'.join(cell_type_list)}_{str(num_cell)}\n min-sum={min_sum_ratio}_min-single={min_single_ratio}"
        logger.info(f"defining niche: {self.niche_name}")
        logger.info("skipping existence checking for MOBT dataset!")
        assert isinstance(self.sample_names, np.ndarray)
        adata_ind = np.where(self.sample_names == sample_id)[0][0]  # np.where: (array([0]),)
        self.adata_q = self.adata_list[adata_ind]

        # self.construct_query_spatial_graph(coord_type='generic', radius=100, visualize=False)
        self.construct_query_spatial_graph(coord_type='grid', visualize=False)

        anchor_niche_list = self.get_niche_with_composition(cell_type_list=cell_type_list, num_cell=num_cell, min_sum_ratio=min_sum_ratio, min_single_ratio=min_single_ratio)
        anchor_ind, niche_ind = random.choice(anchor_niche_list)

        self.check_niche_composition(niche_ind)
        self.save_niche_res(niche_ind, visualize=False, invert_y=True)
