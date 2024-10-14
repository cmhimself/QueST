import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from models.layers import GINLayers, BilinearDiscriminator, BatchDiscriminator, BatchTransform, RevGrad
from torch_geometric.nn.pool import global_mean_pool
import bench.bench_utils as bench_utils
logger = logging.getLogger(__name__)


class QueSTV1(nn.Module):
    def __init__(self, in_dim, param=None, logger=None):
        super().__init__()
        self.logger = logger
        self.logger.info(f"building model, in dim={in_dim}, time: {bench_utils.get_time_str()}")

        self.encoder = None
        self.decoder = None
        self.pooling = param['pooling']
        self.batch_transform = None
        self.batch_discriminator = None
        self.contrastive_discriminator = None
        self.grl = RevGrad()

        assert param['enc_dims'][-1] == param['dec_dims'][0], f"encoder {param['enc_dims']} and decoder {param['dec_dims']} latent dimension does not match!"
        self.enc_dims = [in_dim] + param['enc_dims']
        self.dec_dims = param['dec_dims'] + [in_dim]
        self.zdim = self.enc_dims[-1]
        self.batch_num = param['batch_num']
        self.batch_discriminator_dims = param['batch_discriminator_dims']
        self.dec_batch_dim = param['dec_batch_dim']
        self.batch_transform_dims = [self.batch_num, (self.batch_num + self.dec_batch_dim) // 2, self.dec_batch_dim]

        self.lr = param['lr']
        self.norm = param['norm']
        self.dropout = param['dropout']
        self.activation = param['activation']
        self.weight_decay = param['weight_decay']

        self.build_model()

    def get_subgraph_rep(self, x, edge_index, sub_node_list, sub_edge_list):
        x = self.encoder(x, edge_index)  # node embedding
        subgraph_list = [Data(x=x[sub_node_list[node_ind]], edge_index=sub_edge_list[node_ind]) for node_ind in range(x.shape[0])]
        subgraph_batch = Batch.from_data_list(subgraph_list)
        z = global_mean_pool(subgraph_batch.x, subgraph_batch.batch)
        return z

    def encode_subgraph(self, x, edge_index, subgraph_mask):
        """receives whole sample feature and subgraph mask, returns representation of target region"""
        x = self.encoder(x, edge_index)
        sub_x = x[subgraph_mask]
        sub_z = torch.mean(sub_x, dim=0)
        return sub_z

    def build_model(self):
        self.dec_dims[0] += self.dec_batch_dim
        self.batch_discriminator_dims = self.batch_discriminator_dims + [self.batch_num]

        self.batch_discriminator = BatchDiscriminator(layer_dim=self.batch_discriminator_dims, norm=self.norm, activation=self.activation)
        self.batch_transform = BatchTransform(layer_dim=self.batch_transform_dims, norm='no norm', activation=self.activation)
        self.encoder = GINLayers(layer_dim=self.enc_dims, dropout=self.dropout, norm=self.norm, activation=self.activation, last_norm=True)
        self.decoder = GINLayers(layer_dim=self.dec_dims, dropout=self.dropout, norm=self.norm, activation=self.activation, last_norm=False)
        self.contrastive_discriminator = BilinearDiscriminator(in_dim=self.zdim)

    def build_optimizer(self):
        param_list = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.batch_transform.parameters()) +
                      list(self.contrastive_discriminator.parameters()) + list(self.batch_discriminator.parameters()))
        optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def compute_loss(self, x, recon, logits_positive, logits_negative, logits_batch, batch_label, param, epoch, ref_id):
        loss_recon = F.mse_loss(recon, x)
        loss_positive = F.binary_cross_entropy_with_logits(logits_positive, torch.ones_like(logits_positive))
        loss_negative = F.binary_cross_entropy_with_logits(logits_negative, torch.zeros_like(logits_negative))
        loss_batch = F.cross_entropy(logits_batch, batch_label)
        loss = loss_recon + param['lbd_positive'] * loss_positive + param['lbd_negative'] * loss_negative + param['lbd_batch'] * loss_batch

        self.logger.info("epoch {} sample {}: loss recon: {:.3f}, loss+: {:.3f}, loss-: {:.3f}, loss batch: {:.3f}, loss total: {:.3f}".format(
                          epoch, ref_id, loss_recon.item(), loss_positive.item(), loss_negative.item(), loss_batch.item(), loss.item()))

        return loss

    def forward(self, x, x_shf, edge_index, sub_node_list, sub_edge_list, positive_ind, negative_ind, batch_labels=None):
        z = self.get_subgraph_rep(x, edge_index, sub_node_list, sub_edge_list)
        z_shf = self.get_subgraph_rep(x_shf, edge_index, sub_node_list, sub_edge_list)
        batch_z = self.batch_transform(batch_labels)

        logits_positive = self.contrastive_discriminator(z[positive_ind], z_shf[positive_ind])
        logits_negative = self.contrastive_discriminator(z[positive_ind], z_shf[negative_ind])
        logits_batch = self.batch_discriminator(self.grl(z))

        z = torch.cat((z, batch_z), dim=1)
        recon = self.decoder(z, edge_index)
        return recon, logits_positive, logits_negative, logits_batch
