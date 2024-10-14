import os
import json
import torch
import scipy
import random
import logging
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import networkx as nx
from collections import Counter
import torch.nn.functional as F
import bench.bench_utils as bench_utils
import models.model_utils as model_utils
from torch_geometric.utils import k_hop_subgraph
from models.model import QueSTV1

logger = logging.getLogger(__name__)
random.seed(2024)
np.random.seed(2024)


class Baseline:
    def __init__(self, adata_q, adata_ref_list, query_sample_id='151507', ref_sample_id_list=None, dataset='DLPFC',
                 cell_type_key='cell_type', library_key='library_id', device=None, save_folder=None, save_query=False):
        self.save_query = save_query
        self.save_folder = save_folder
        self.adata_q = adata_q
        self.adata_ref_list = adata_ref_list
        self.query_sample_id = query_sample_id
        self.ref_sample_id_list = ref_sample_id_list
        self.dataset = dataset
        self.cell_type_key = cell_type_key
        self.library_key = library_key
        self.device = device
        self.set_method_path()
        self.build_graph()
        if device is not None:
            logger.info(f"setting global gpu device id: {device}")
            os.environ["CUDA_VISIBLE_DEVICES"] = device

    def set_method_path(self):
        self.method_path = 'random'

    def build_graph(self):
        logger.info("constructing graph for each anndata object")
        bench_utils.construct_adata_spatial_graph(self.adata_q, dataset=self.dataset)
        bench_utils.construct_adata_list_spatial_graph(self.adata_ref_list, dataset=self.dataset)

    def query(self, k=[3], niche_prefix=None):
        logger.info("***** calling main query function *****")
        self.build_graph()
        query_subgraph_dict, query_sim_dict = {}, {}
        for i, (ref_sample_id, adata_ref) in enumerate(zip(self.ref_sample_id_list, self.adata_ref_list)):
            G = nx.from_scipy_sparse_array(adata_ref.obsp['spatial_connectivities'])
            center_ind = random.choice(range(adata_ref.shape[0]))
            if isinstance(k, int):
                subgraph = nx.ego_graph(G, center_ind, radius=k, undirected=True, center=True)
            elif isinstance(k, list):
                subgraph = nx.ego_graph(G, center_ind, radius=k[i], undirected=True, center=True)
            else:
                assert False, "Unknown input format of k!"
            query_ind = list(subgraph.nodes)
            sim = np.random.uniform(low=0, high=1, size=adata_ref.n_obs)
            query_subgraph_dict[ref_sample_id] = query_ind
            query_sim_dict[ref_sample_id] = sim
            self.save_query_subgraph_to_adata(adata_ref, query_ind, niche_prefix)
            self.save_query_sim_to_adata(adata_ref, sim, niche_prefix, ref_sample_id)
        return query_subgraph_dict, query_sim_dict

    def save_query_subgraph_to_adata(self, adata_ref, query_ind, niche_prefix):
        logger.info(f"checking queried subgraph composition: {Counter(adata_ref.obs[self.cell_type_key][query_ind])}")
        adata_ref.obs[f'{niche_prefix}_subgraph'] = pd.Categorical(['Query' if i in query_ind else 'Else' for i in range(adata_ref.shape[0])])
        adata_ref.obs[f'{niche_prefix}_subgraph_cell_type'] = pd.Categorical([adata_ref.obs[self.cell_type_key][i] if i in query_ind else 'Else' for i in
                                                                              range(adata_ref.shape[0])])

    def save_query_sim_to_adata(self, adata_ref, sim, niche_prefix, ref_sample_id):
        logger.info(f"saving queried similarity value to ref {ref_sample_id}")
        adata_ref.obs[f'{niche_prefix}_sim'] = sim

    def save_ref_data_with_query_res(self, test=False):
        if test:
            save_path = f"{self.save_folder}/{self.method_path}/test/adata/"
        else:
            save_path = f"{self.save_folder}/{self.method_path}/adata/"
        logger.info(f"writing adata with query result to {save_path}")
        for ref_sample_id, adata_ref in zip(self.ref_sample_id_list, self.adata_ref_list):
            # logger.info(f"{ref_sample_id} {adata_ref.obs_keys()}")
            try:
                adata_ref.write_h5ad(filename=f"{save_path}/{ref_sample_id}.h5ad", compression="gzip")
            except:
                logger.info("encountering possible tuple in adata, deleting")
                del adata_ref.uns['spatial_neighbors']['params']['radius']
                adata_ref.write_h5ad(filename=f"{save_path}/{ref_sample_id}.h5ad", compression="gzip")


class QueSTV1Baseline(Baseline):
    def set_method_path(self):
        self.method_path = 'model-v2'
        self.preprocess = False

    def query(self, k=[3], niche_prefix=None):
        device = "cuda:0"

        model_path = f"./results/{self.dataset}/model.pth"
        param_path = f'./results/{self.dataset}/param.json'
        in_dim = 12378 if self.dataset == "DLPFC" else 4132
        with open(param_path, 'r') as f: param = json.load(f)
        model = QueSTV1(in_dim=in_dim, param=param, logger=logger).to(device)
        logger.info(f"loading model from {model_path}"), model.load_state_dict(torch.load(model_path)), model.eval()

        if not self.preprocess:
            model_utils.build_graphs(self.adata_ref_list)
            adata_list = model_utils.preprocess_adata([self.adata_q] + self.adata_ref_list, param=param)
            self.adata_q, self.adata_ref_list = adata_list[0], adata_list[1:]
            self.feature_list, self.edge_ind_list, self.sub_node_sample_list, self.sub_edge_ind_sample_list, _ = model_utils.prepare_graph_data(self.adata_ref_list, param)
            self.preprocess = True

        feature_q = model_utils.get_feature(self.adata_q, query=True, param=param)
        adj_q = self.adata_q.obsp['spatial_connectivities'].tocoo()
        edge_index_q = torch.tensor(np.vstack((adj_q.row, adj_q.col)), dtype=torch.int64).to(device) 
        niche_name = f"{niche_prefix}_niche"
        niche_mask = torch.tensor(self.adata_q.obs[niche_name] == 'Niche').to(device)

        with torch.no_grad():
            query_subgraph_dict, query_sim_dict = {}, {}
            for i, (ref_sample_id, adata_ref) in enumerate(zip(self.ref_sample_id_list, self.adata_ref_list)):
                subgraph_k = k[0] if len(k) < len(self.adata_ref_list) else k[i]
                logger.info(f"performing query: {ref_sample_id}, time: {bench_utils.get_time_str()}")
                feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref = self.feature_list[i], self.edge_ind_list[i], self.sub_node_sample_list[i], self.sub_edge_ind_sample_list[i]
                sim = model_utils.query(feature_q, feature_ref, edge_index_q, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref, model, niche_mask, method=param['query_method'])

                best_node = torch.argmax(sim).item()
                best_subgraph, _, _, _ = k_hop_subgraph(best_node, subgraph_k, edge_index_ref)
                best_subgraph = best_subgraph.cpu().numpy()
                sim = sim.cpu().numpy()

                query_subgraph_dict[ref_sample_id] = best_subgraph
                query_sim_dict[ref_sample_id] = sim

                self.save_query_subgraph_to_adata(adata_ref, best_subgraph, niche_prefix)
                self.save_query_sim_to_adata(adata_ref, sim, niche_prefix, ref_sample_id)

        return query_subgraph_dict, query_sim_dict
