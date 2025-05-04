import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.data import Data, Batch
from models.layers import BilinearDiscriminator, BatchDiscriminator, BatchEncoder, GINLayers
from torch_geometric.nn.pool import global_mean_pool
logger = logging.getLogger(__name__)


class QueST(nn.Module):
    def __init__(self, in_dim, param=None, logger=None):
        super(QueST, self).__init__()
        self.logger = logger
        self.logger.info("initializing QueST object")
        param = deepcopy(param)
        self.recon_loss = param['recon_loss']
        self.encoder = None
        self.decoder = None
        self.batch_enc = None
        self.batch_disc = None
        self.contrast_disc = None
        assert param['enc_dims'][-1] == param['dec_dims'][0], f"encoder {param['enc_dims']} and decoder {param['dec_dims']} latent dimension does not match!"

        self.in_dim = in_dim
        self.enc_dims = [in_dim] + param['enc_dims']
        self.dec_dims = param['dec_dims'] + [in_dim]

        self.zdim = self.enc_dims[-1]
        self.batch_num = param['batch_num']
        self.batch_disc_dims = param['batch_disc_dims']
        self.dec_batch_dim = param['dec_batch_dim']
        self.batch_transform_dims = [self.batch_num, (self.batch_num + self.dec_batch_dim) // 2, self.dec_batch_dim]
        self.residual = param['residual']
        self.lr_gen = param['lr_gen']
        self.lr_disc = param['lr_disc']
        self.norm = param['norm']
        self.dropout = param['dropout']
        self.activation = param['activation']
        self.weight_decay = param['weight_decay']
        self.lbd_positive = param['lbd_positive']
        self.lbd_negative = param['lbd_negative']
        self.lbd_gen = param['lbd_gen']

        self.build_model()

    def build_model(self):
        self.dec_dims[0] += self.dec_batch_dim
        self.batch_disc_dims = self.batch_disc_dims + [self.batch_num]
        self.batch_disc = BatchDiscriminator(layer_dim=self.batch_disc_dims, norm=self.norm, activation=self.activation)
        self.batch_enc = BatchEncoder(layer_dim=self.batch_transform_dims, norm='no norm', activation=self.activation)
        self.encoder = GINLayers(layer_dim=self.enc_dims, dropout=self.dropout, norm=self.norm, activation=self.activation, res=self.residual, last_norm=True)
        self.decoder = GINLayers(layer_dim=self.dec_dims, dropout=self.dropout, norm=self.norm, activation=self.activation, res=self.residual, last_norm=False)
        self.contrast_disc = BilinearDiscriminator(in_dim=self.zdim)

    def build_optimizer(self):
        param_list_gen = (list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                          list(self.batch_enc.parameters()) + list(self.contrast_disc.parameters()))
        param_list_disc = list(self.batch_disc.parameters())

        optimizer_gen = torch.optim.Adam(param_list_gen, lr=self.lr_gen, weight_decay=self.weight_decay)
        optimizer_disc = torch.optim.Adam(param_list_disc, lr=self.lr_disc)

        return optimizer_gen, optimizer_disc

    def compute_loss(self, x=None, logits_batch=None, batch_label=None, epoch=None, ref_id=None,
                     recon=None, logits_positive=None, logits_negative=None, gan_step='gen'):
        loss_log_str = "Epoch {} Sample {}: ".format(epoch, ref_id)
        if gan_step == 'disc':
            loss_disc = F.cross_entropy(logits_batch, batch_label)
            loss_log_str += "loss disc: {:.3f}".format(loss_disc.item())
            self.logger.info(loss_log_str)
            return loss_disc

        assert gan_step == 'gen'

        loss_recon = F.mse_loss(recon, x)
        loss_log_str += "loss mse: {:.3f}, ".format(loss_recon.item())

        loss = loss_recon
        loss_gen = F.cross_entropy(logits_batch, batch_label)
        loss += self.lbd_gen * loss_gen
        loss_log_str += "loss gen: {:.3f}, ".format(loss_gen.item())

        loss_positive = F.binary_cross_entropy_with_logits(logits_positive, torch.ones_like(logits_positive))
        loss_negative = F.binary_cross_entropy_with_logits(logits_negative, torch.zeros_like(logits_negative))
        loss += self.lbd_positive * loss_positive + self.lbd_negative * loss_negative
        loss_log_str += "loss positive: {:.3f}, loss negative: {:.3f}, ".format(loss_positive.item(), loss_negative.item())

        loss_log_str += "loss total: {:.3f}".format(loss.item())
        self.logger.info(loss_log_str)
        return loss

    def pool(self, x, sub_node_list, sub_edge_list):
        subgraph_list = [Data(x=x[sub_node_list[node_ind]], edge_index=sub_edge_list[node_ind]) for node_ind in range(x.shape[0])]
        subgraph_batch = Batch.from_data_list(subgraph_list)
        z = global_mean_pool(subgraph_batch.x, subgraph_batch.batch)
        return z

    def pool_truncated(self, x, sub_node_list, sub_edge_list, threshold=10000000):
        # print("calling pool_truncated!", x.shape, np.sum([len(sub_node_list[node_ind]) for node_ind in range(x.shape[0])]))
        subgraph_list = []
        total_nodes = 0

        for node_ind in range(x.shape[0]):
            subgraph_list.append(Data(x=x[sub_node_list[node_ind]], edge_index=sub_edge_list[node_ind]))
            total_nodes += subgraph_list[-1].num_nodes

            if total_nodes > threshold:
                # print(total_nodes, "exceeding threshold", threshold, "using truncated pooling")
                subgraph_batch = Batch.from_data_list(subgraph_list)
                z = global_mean_pool(subgraph_batch.x, subgraph_batch.batch)
                if 'z_concat' not in locals():
                    z_concat = z
                else:
                    z_concat = torch.cat((z_concat, z), dim=0)

                subgraph_list = []
                total_nodes = 0

        if subgraph_list:
            subgraph_batch = Batch.from_data_list(subgraph_list)
            z = global_mean_pool(subgraph_batch.x, subgraph_batch.batch)
            if 'z_concat' not in locals():
                z_concat = z
            else:
                z_concat = torch.cat((z_concat, z), dim=0)
        return z_concat

    def get_subgraph_rep(self, x, edge_index, sub_node_list, sub_edge_list):
        z_node = self.encoder(x, edge_index)
        z_subgraph = self.pool(z_node, sub_node_list, sub_edge_list)
        # z_subgraph = self.pool_truncated(z_node, sub_node_list, sub_edge_list)
        return z_node, z_subgraph

    def forward(self, x, x_shf=None, edge_index=None, sub_node_list=None, sub_edge_list=None, positive_ind=None, negative_ind=None, batch_labels=None):
        z_node, z_subgraph = self.get_subgraph_rep(x, edge_index, sub_node_list, sub_edge_list)
        z_node_shf, z_subgraph_shf = self.get_subgraph_rep(x_shf, edge_index, sub_node_list, sub_edge_list)

        batch_emb = self.batch_enc(batch_labels)
        logits_positive = self.contrast_disc(z_subgraph[positive_ind], z_subgraph_shf[positive_ind])
        logits_negative = self.contrast_disc(z_subgraph[positive_ind], z_subgraph_shf[negative_ind])
        logits_batch = self.batch_disc(z_subgraph)
        z_node_batch = torch.cat((z_node, batch_emb), dim=1)
        recon = self.decoder(z_node_batch, edge_index)

        return z_subgraph, z_subgraph_shf, recon, logits_positive, logits_negative, logits_batch

    def forward_batch_disc(self, x, edge_index, sub_node_list, sub_edge_list):
        z_node, z_subgraph = self.get_subgraph_rep(x, edge_index, sub_node_list, sub_edge_list)
        logits_batch = self.batch_disc(z_subgraph)
        return logits_batch

    def encode_subgraph(self, x, edge_index, subgraph_mask):
        z_node = self.encoder(x, edge_index)
        z_node_subgraph = z_node[subgraph_mask]
        z_subgraph = torch.mean(z_node_subgraph, dim=0)
        return z_subgraph

    def query(self, feature_q, feature_ref, edge_index_q, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref, niche_mask):
        niche_z = self.encode_subgraph(feature_q, edge_index_q, niche_mask)
        _, ref_z = self.get_subgraph_rep(feature_ref, edge_index_ref, sub_node_list_ref, sub_edge_ind_list_ref)
        niche_z = torch.reshape(niche_z, (niche_z.shape[0], 1))
        ref_z_normed = F.normalize(ref_z, p=2, dim=1)
        niche_z_normed = F.normalize(niche_z, p=2, dim=0)
        sim = torch.mm(ref_z_normed, niche_z_normed).squeeze(1).cpu().numpy()
        return sim

