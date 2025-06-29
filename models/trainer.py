import os
import torch
import random
import logging
import argparse
import numpy as np
import scanpy as sc
import torch_geometric
from models.model import QueST
import models.model_utils as model_utils
import bench.bench_utils as bench_utils
logger = logging.getLogger(__name__)


class QueSTTrainer:
    def __init__(self, dataset, data_path, sample_ids, adata_list, query_sample_id=None, query_niches=None, 
                 save_model=True, model_path='', embedding_folder='', epochs=20, hvg=None, min_count=0, 
                 normalize=True, k=3):  
        self.k = k
        self.hvg = hvg
        self.epochs = epochs
        self.save_model = save_model
        self.min_count = min_count
        self.normalize = normalize
        self.get_params()
        self.dataset = dataset
        self.data_path = data_path
        self.model = None
        self.feature_list = None
        self.edge_ind_list = None
        self.sub_node_sample_list = None
        self.sub_edge_ind_sample_list = None
        self.batch_label_list = None

        logger.info(f"QueST Trainer initialized, loading data, time: {bench_utils.get_time_str()}")
        self.sample_id_list = sample_ids
        self.adata_list = adata_list

        if query_sample_id is not None:
            assert query_niches is not None
            self.adata_q = self.adata_list[self.sample_id_list.index(query_sample_id)]
            assert self.adata_q.uns['library_id'] == query_sample_id
            self.adata_ref_list = [self.adata_list[i] for i in range(len(self.adata_list)) if self.sample_id_list[i] != query_sample_id]
            self.ref_id_list = [sample_id for sample_id in self.sample_id_list if sample_id != query_sample_id]
        
        self.query_sample_id = query_sample_id
        self.niche_prefix_list = query_niches
        logger.info(f"Data loaded with {len(self.adata_list)} adata objects, time: {bench_utils.get_time_str()}")
        self.model_param['batch_num'] = len(self.adata_list)
        self.model_path = model_path
        self.embedding_folder = embedding_folder

    def get_params(self):
        logger.info(f"Processing arguments, time: {bench_utils.get_time_str()}")
        parser = argparse.ArgumentParser(description='Process QueST Parameters.')
        model_param_group_name, other_param_group_name = 'model params', 'other params'
        model_param_group = parser.add_argument_group(model_param_group_name)

        model_param_group.add_argument('--seed', type=int, default=2025)
        model_param_group.add_argument('--model-path', type=str, default=None)
        model_param_group.add_argument('--device', type=str, default='cuda:0')
        model_param_group.add_argument('--library_key', type=str, default='library_id')

        model_param_group.add_argument('--model_k', type=int, default=self.k)
        model_param_group.add_argument('--query_k', type=int, default=self.k)
        model_param_group.add_argument('--normalize', type=bool, default=self.normalize)
        model_param_group.add_argument('--hvg', type=int, default=self.hvg)
        model_param_group.add_argument('--min-count', type=int, default=self.min_count)
        model_param_group.add_argument('--epochs', type=int, default=self.epochs)
        model_param_group.add_argument('--shuffle_min_k', default=[self.k])
        model_param_group.add_argument('--shuffle_max_k', default=[self.k])
        model_param_group.add_argument('--fix_portion', default=[0.02])
        model_param_group.add_argument('--min_shuffle_ratio', type=float, default=0.25)
        model_param_group.add_argument('--max_shuffle_ratio', type=float, default=0.75)
        model_param_group.add_argument('--recon_loss', type=str, default='mse')
        model_param_group.add_argument('--residual', type=bool, default=True)
        model_param_group.add_argument('--pooling', type=str, default='mean')
        model_param_group.add_argument('--enc-dims', default=[2048, 256, 32])
        model_param_group.add_argument('--dec-dims', default=[32])
        model_param_group.add_argument('--batch-disc-dims', default=[32, 32, 16])
        model_param_group.add_argument('--dec-batch-dim', type=int, default=2)
        model_param_group.add_argument('--batch-num', type=int, default=2)
        model_param_group.add_argument('--lr_gen', type=float, default=0.001)
        model_param_group.add_argument('--lr_disc', type=float, default=0.005)
        model_param_group.add_argument('--norm', type=str, default='batchnorm')
        model_param_group.add_argument('--dropout', type=float, default=0.1)
        model_param_group.add_argument('--activation', type=str, default='relu')
        model_param_group.add_argument('--weight_decay', type=float, default=5e-4)
        model_param_group.add_argument('--lbd-positive', type=float, default=1)
        model_param_group.add_argument('--lbd-negative', type=float, default=1)
        model_param_group.add_argument('--lbd-gen', type=float, default=1)

        other_param_group = parser.add_argument_group(other_param_group_name)
        other_param_group.add_argument('--save-model', type=bool, default=self.save_model)
        other_param_group.add_argument('--gpu-id', type=str, default='0')

        # args = parser.parse_args()
        args = parser.parse_args([])
        logger.info(f"using GPU {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        args_dict = vars(args)

        group_dicts = {}
        for group in parser._action_groups:
            group_name = group.title
            group_args = {a.dest: args_dict[a.dest] for a in group._group_actions if a.dest in args_dict}
            group_dicts[group_name] = group_args

        self.model_param = group_dicts[model_param_group_name]
        self.other_param = group_dicts[other_param_group_name]
        
    def train_model(self):
        optimizer_gen, optimizer_batch_disc = self.model.build_optimizer()
        self.model.train()
        for epoch in range(self.model_param['epochs']):
            indices = list(range(len(self.adata_list)))
            random.shuffle(indices)
            for batch_idx, i in enumerate(indices):
                adata = self.adata_list[i]
                feature, edge_index, batch_label = self.feature_list[i], self.edge_ind_list[i], self.batch_label_list[i]
                sub_node_list, sub_edge_ind_list = self.sub_node_sample_list[i], self.sub_edge_ind_sample_list[i]
                logger.info(f"Epoch: {epoch}, Batch: {batch_idx + 1}/{len(self.adata_list)}, Sample: {adata.uns['library_id']}, Time: {bench_utils.get_time_str()}")
                for gan_step in ['disc', 'gen']:
                    if gan_step == 'disc':
                        logger.info("********** discriminator step **********")
                        logits_batch = self.model.forward_batch_disc(feature, edge_index, sub_node_list, sub_edge_ind_list)
                        loss_batch_disc = self.model.compute_loss(logits_batch=logits_batch, batch_label=batch_label, epoch=epoch,
                                                                  ref_id=adata.uns['library_id'], gan_step=gan_step)
                        optimizer_batch_disc.zero_grad(), loss_batch_disc.backward(), optimizer_batch_disc.step()
                    elif gan_step == 'gen':
                        logger.info("********** generator step **********")
                        batch_label_fuzzy = torch.full_like(batch_label, 1 / self.model_param['batch_num']).to(self.model_param['device'])
                        min_k, max_k, fix_portion = model_utils.get_shuffle_param(self.model_param, i, len(self.adata_list))
                        
                        adata_shf, fixed_center, fixed_nodes, shuffle_center, feature_shf = model_utils.shuffle(adata, dataset=self.dataset, feature=feature,
                                                                                                                min_k=min_k, max_k=max_k, fix_portion=fix_portion,
                                                                                                                min_shuffle_ratio=self.model_param['min_shuffle_ratio'],
                                                                                                                max_shuffle_ratio=self.model_param['max_shuffle_ratio'],
                                                                                                                sub_node_list=sub_node_list)
                        shuffle_center_sampled = np.random.choice(shuffle_center, size=len(fixed_center), replace=False)
                        z, z_shf, recon, logits_positive, logits_negative, logits_batch = self.model(
                            feature, feature_shf, edge_index, sub_node_list, sub_edge_ind_list, fixed_center, shuffle_center_sampled, batch_label)
                        loss = self.model.compute_loss(x=feature, recon=recon, logits_batch=logits_batch, batch_label=batch_label_fuzzy, epoch=epoch,
                                                       ref_id=adata.uns['library_id'], logits_positive=logits_positive, logits_negative=logits_negative,
                                                       gan_step=gan_step)
                        optimizer_gen.zero_grad(), loss.backward(), optimizer_gen.step()

        if self.other_param['save_model']:
            torch.save(self.model.state_dict(), self.model_path)

    def get_query_mask(self, q_id='Sample1', q_obs_key='Type-1_niche', q_niche_name='Niche'):
        q_niche_mask = self.adata_q.obs[q_obs_key] == q_niche_name
        self.adata_q.obs['mask'] = q_niche_mask.astype(int)
        return q_id, q_niche_mask

    def query(self, q_id, q_niche_mask, q_niche_name):
        logger.info(f"Performing query: q_id={q_id}, q_niche={q_niche_name}, time: {bench_utils.get_time_str()}")

        feat_q = self.feature_list[self.sample_id_list.index(q_id)]
        edge_ind_q = self.edge_ind_list[self.sample_id_list.index(q_id)]

        # ref_id_list = [sample_id for sample_id in self.sample_id_list if sample_id != q_id]
        # adata_ref_list = [self.adata_list[i] for i in range(len(self.adata_list)) if self.sample_id_list[i] != q_id]
        feat_ref_list = [self.feature_list[i] for i in range(len(self.feature_list)) if self.sample_id_list[i] != q_id]
        edge_ind_ref_list = [self.edge_ind_list[i] for i in range(len(self.edge_ind_list)) if self.sample_id_list[i] != q_id]
        sub_node_ref_list = [self.sub_node_sample_list[i] for i in range(len(self.sub_node_sample_list)) if self.sample_id_list[i] != q_id]
        sub_edge_ind_ref_list = [self.sub_edge_ind_sample_list[i] for i in range(len(self.sub_edge_ind_sample_list)) if self.sample_id_list[i] != q_id]

        
        with torch.no_grad():
            for ref_id, adata_ref, feat_ref, edge_ind_ref, sub_node_ref, sub_edge_ind_ref \
            in zip(self.ref_id_list, self.adata_ref_list, feat_ref_list, edge_ind_ref_list, sub_node_ref_list, sub_edge_ind_ref_list):
                logger.info(f"Processing {ref_id}, saving predicted niche matching score to Anndata")
                cosine_sim = self.model.query(feat_q, feat_ref, edge_ind_q, edge_ind_ref, sub_node_ref, sub_edge_ind_ref, q_niche_mask)
                adata_ref.obs[f'{q_niche_name} predicted matching score'] = cosine_sim

    def get_embedding(self):
        logger.info(f"Saving niche embeddings for each sample, time: {bench_utils.get_time_str()}")
        with torch.no_grad():
            for i, (data_id, adata, feat, edge_ind, sub_node, sub_edge_ind) in enumerate(zip(self.sample_id_list, self.adata_list, self.feature_list, self.edge_ind_list,
                                                                                                self.sub_node_sample_list, self.sub_edge_ind_sample_list)):
                logger.info(f"Processing {data_id}")
                _, z = self.model.get_subgraph_rep(feat, edge_ind, sub_node, sub_edge_ind)
                torch.save(z, f"{self.embedding_folder}/{adata.uns['library_id']}.pt")

    def train(self):
        torch_geometric.seed.seed_everything(self.model_param['seed'])
        for adata in self.adata_list:
            assert adata.uns['library_id'] is not None, "Library id must be set for each Anndata object!"
            if 'spatial_connectivities' not in adata.obsp.keys():
                model_utils.build_graphs(adata_list=[adata], dataset=self.dataset)
            else:
                logger.info(f"adata {adata.uns['library_id']} has existing graph")
        # model_utils.build_graphs(adata_list=self.adata_list, dataset=self.dataset)
        model_utils.preprocess_adata(self.adata_list, param=self.model_param)
        self.feature_list, self.edge_ind_list, self.sub_node_sample_list, self.sub_edge_ind_sample_list, self.batch_label_list = model_utils.prepare_graph_data(self.adata_list, self.model_param)
        self.model = QueST(in_dim=self.feature_list[0].shape[1], param=self.model_param, logger=logger).to(self.model_param['device'])
        self.train_model()
    
    def inference(self, save_embedding=False, query=False):
        logger.info(f"Loading model, time: {bench_utils.get_time_str()}")
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        if save_embedding:
            self.get_embedding()
        if query:
            for niche_prefix in self.niche_prefix_list:
                q_id, q_niche_mask = self.get_query_mask(q_id=self.query_sample_id, q_obs_key=niche_prefix, q_niche_name="Niche")
                self.query(q_id, q_niche_mask, niche_prefix)