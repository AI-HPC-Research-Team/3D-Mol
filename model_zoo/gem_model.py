#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is an implementation of GeoGNN:
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl

from networks.gnn_block import GIN_2
from networks.compound_encoder import AtomEmbedding, BondEmbedding, BondFloatRBF, BondAngleFloatRBF
from networks.gnn_block import MeanPool, GraphNorm
from networks.basic_block import MLP
from .weighted_nt_xent import WeightedNTXentLoss_func

import paddle.nn.functional as F


class GeoGNNModel_all(nn.Layer):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """

    def __init__(self, model_config={}):
        super(GeoGNNModel_all, self).__init__()
        print(model_config)
        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')
        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config['bond_angle_float_names']
        self.dihes_bond_float_names = ["dihes_angle"]

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        self.init_super_edge_embedding = BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim)

        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()
        self.bond_angle_float_rbf_list = nn.LayerList()
        self.dihes_angle_float_rbf_list = nn.LayerList()
        self.atom_bond_block_list = nn.LayerList()
        self.bond_angle_block_list = nn.LayerList()
        self.dihes_angle_list = nn.LayerList()

        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.bond_angle_float_rbf_list.append(BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
            self.dihes_angle_float_rbf_list.append(BondAngleFloatRBF(self.dihes_bond_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                GeoGNNBlock_2(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.bond_angle_block_list.append(
                GeoGNNBlock_2(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.dihes_angle_list.append(
                GeoGNNBlock_2(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))

        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)
        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)
        print('[GeoGNNModel] atom_names:%s' % str(self.atom_names))
        print('[GeoGNNModel] bond_names:%s' % str(self.bond_names))
        print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, dihes_angle_graph):
        """
        Build the network.
        """
        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)
        super_edge_hidden = self.init_super_edge_embedding(bond_angle_graph.edge_feat)
        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        super_edge_hidden_list = [super_edge_hidden]
        for layer_id in range(self.layer_num):
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
            cur_dihes_angle_hidden = self.dihes_angle_float_rbf_list[layer_id](dihes_angle_graph.edge_feat)
            node_hidden, edge_hidden_1 = self.atom_bond_block_list[layer_id](atom_bond_graph, node_hidden_list[
                layer_id], edge_hidden_list[layer_id])
            edge_hidden, dihes_node_hidden_1 = self.bond_angle_block_list[layer_id](bond_angle_graph,
                                                                                    cur_edge_hidden + edge_hidden_1,
                                                                                    super_edge_hidden_list[layer_id])
            dihes_node_hidden, _ = self.dihes_angle_list[layer_id](dihes_angle_graph,
                                                                   cur_angle_hidden + dihes_node_hidden_1,
                                                                   cur_dihes_angle_hidden)
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
            super_edge_hidden_list.append(dihes_node_hidden)
        edge_repr = edge_hidden_list[-1]
        node_repr = node_hidden_list[-1]
        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        return node_repr, edge_repr, graph_repr


class GeoGNNBlock_2(nn.Layer):
    """
    GeoGNN Block
    """

    def __init__(self, embed_dim, dropout_rate, last_act):
        super(GeoGNNBlock_2, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN_2(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        node_out, edge_out = self.gnn(graph, node_hidden, edge_hidden)
        # node_out, edge_out = gat(graph, node_hidden, edge_hidden)
        node_out = self.norm(node_out)
        edge_out = self.norm(edge_out)
        node_out = self.graph_norm(graph, node_out)
        if self.last_act:
            node_out = self.act(node_out)
            edge_out = self.act(edge_out)
        node_out = self.dropout(node_out)
        edge_out = self.dropout(edge_out)
        node_out = node_out + node_hidden
        edge_out = edge_out + edge_hidden
        return node_out, edge_out


class GeoPredModel_all(nn.Layer):
    """tbd"""

    def __init__(self, model_config, compound_encoder):
        super(GeoPredModel_all, self).__init__()
        self.compound_encoder = compound_encoder
        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']
        x = paddle.zeros(shape=[7], dtype='float32')
        self.coefs = paddle.create_parameter(
            shape=x.shape,
            dtype=str(x.numpy().dtype),
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(x),
                regularizer=None
            ))
        if 'Cm' in self.pretrain_tasks:
            self.Cm_vocab = model_config['Cm_vocab']
            self.Cm_linear = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim,
                                    out_size=self.Cm_vocab + 2, dropout_rate=self.dropout_rate)
            self.Cm_loss = nn.CrossEntropyLoss()
        if 'En' in self.pretrain_tasks:
            self.En_linear = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim,
                                    out_size=1, dropout_rate=self.dropout_rate)
            self.En_loss = nn.SmoothL1Loss()
        self.Fg_linear = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim,
                             out_size=model_config['Fg_size'], dropout_rate=self.dropout_rate)
        self.Fg_loss = nn.BCEWithLogitsLoss()
        if 'Bar' in self.pretrain_tasks:
            self.Bar_vocab = model_config['Bar_vocab']
            self.Bar_mlp_1 = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim * 2,
                                 out_size=compound_encoder.embed_dim, dropout_rate=self.dropout_rate)
            self.Bar_mlp_2 = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim * 2,
                                 out_size=self.Bar_vocab + 2, dropout_rate=self.dropout_rate)
            self.Bar_loss = nn.CrossEntropyLoss()
        if 'Dar' in self.pretrain_tasks:
            self.Dar_vocab = model_config['Dar_vocab']
            self.Dar_mlp1 = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim * 3,
                                out_size=compound_encoder.embed_dim, dropout_rate=self.dropout_rate)
            self.Dar_loss = nn.CrossEntropyLoss()
            self.Dar_mlp2 = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim * 2,
                                out_size=self.Dar_vocab + 2, dropout_rate=self.dropout_rate)
            self.Dar_loss_extra = nn.CrossEntropyLoss()
        if 'Blr' in self.pretrain_tasks:
            self.Blr_vocab = model_config['Blr_vocab']
            self.Blr_mlp = MLP(2, hidden_size=self.hidden_size, act=self.act, in_size=compound_encoder.embed_dim * 2,
                               out_size=1, dropout_rate=self.dropout_rate)
            self.Blr_loss = nn.SmoothL1Loss()
        if 'Adc' in self.pretrain_tasks:
            self.Adc_vocab = model_config['Adc_vocab']
            self.Adc_mlp = MLP(2, hidden_size=self.hidden_size, in_size=self.compound_encoder.embed_dim * 2,
                               act=self.act, out_size=1, dropout_rate=self.dropout_rate)
            self.Adc_loss = nn.SmoothL1Loss()
        if 'Cl' in self.pretrain_tasks:
            self.Cl_vocab = model_config['Cl_vocab']
            self.Cl_linear = MLP(2, hidden_size=self.hidden_size, in_size=compound_encoder.embed_dim,
                                 act=self.act, out_size=1, dropout_rate=self.dropout_rate)
            self.Cl_loss = nn.SmoothL1Loss()

        print('[GeoPredModel] pretrain_tasks:%s' % str(self.pretrain_tasks))

    def _get_Cm_loss(self, feed_dict, node_repr):
        masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
        logits = self.Cm_linear(masked_node_repr)
        loss = self.Cm_loss(logits, feed_dict['Cm_context_id'])
        return loss

    def _get_En_loss(self, feed_dict, graph_repr):
        logits = self.En_linear(graph_repr)
        loss = self.En_loss(logits, feed_dict['eng'])
        return loss

    def _get_Cm_loss_1(self, feed_dict, node_repr, graph):
        masked_node_repr = paddle.gather(node_repr, feed_dict['Cm_node_i'])
        node_feat = None
        for i in ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]:
            temps = graph.node_feat[i]
            if temps.shape[-1] != 1:
                temps = paddle.unsqueeze(temps, axis=-1)
            temps = paddle.cast(temps, dtype='float32')
            if node_feat is None:
                node_feat = temps
            else:
                node_feat = paddle.concat([node_feat, temps], axis=-1)
        node_feat = paddle.gather(node_feat, feed_dict['Cm_node_i'])
        loss = 0
        logits1 = self.Cm1_linear_1(masked_node_repr)
        loss = loss + self.Cm1_loss_1(logits1, paddle.unsqueeze(node_feat[:, 0], axis=-1))
        logits2 = self.Cm1_linear_2(masked_node_repr)
        loss = loss + self.Cm1_loss_2(logits2, paddle.unsqueeze(node_feat[:, 1], axis=-1))
        logits3 = self.Cm1_linear_3(masked_node_repr)
        loss = loss + self.Cm1_loss_3(logits3, paddle.unsqueeze(node_feat[:, 2], axis=-1))
        logits4 = self.Cm1_linear_4(masked_node_repr)
        loss = loss + self.Cm1_loss_4(logits4, paddle.unsqueeze(node_feat[:, 3], axis=-1))
        logits5 = self.Cm1_linear_5(masked_node_repr)
        loss = loss + self.Cm1_loss_5(logits5, paddle.unsqueeze(node_feat[:, 4], axis=-1))
        logits6 = self.Cm1_linear_6(masked_node_repr)
        loss = loss + self.Cm1_loss_6(logits6, paddle.unsqueeze(node_feat[:, 5], axis=-1))
        logits7 = self.Cm1_linear_7(masked_node_repr)
        loss = loss + self.Cm1_loss_7(logits7, paddle.unsqueeze(node_feat[:, 6], axis=-1))

        return loss

    def _get_Fg_loss(self, feed_dict, graph_repr):
        fg_label = paddle.concat([feed_dict['Fg_morgan'], feed_dict['Fg_daylight'], feed_dict['Fg_maccs']], 1)
        logits = self.Fg_linear(graph_repr)
        loss = self.Fg_loss(logits, fg_label)
        return loss

    def _get_Bar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        node_ij_repr = self.Bar_mlp_1(node_ij_repr)
        node_jk_repr = paddle.concat([node_j_repr, node_k_repr], 1)
        node_jk_repr = self.Bar_mlp_1(node_jk_repr)
        pred = self.Bar_mlp_2(paddle.concat([node_ij_repr, node_jk_repr], 1))
        Bar_dist_id = paddle.cast(feed_dict['Ba_bond_angle'] / np.pi * self.Bar_vocab, 'int64')
        loss = self.Bar_loss(pred, Bar_dist_id)
        # loss = self.Bar_loss(pred, feed_dict['Ba_bond_angle'] / np.pi)
        return loss

    def _get_Dar_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Da_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Da_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Da_node_k'])
        node_l_repr = paddle.gather(node_repr, feed_dict['Da_node_l'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        node_jkl_repr = paddle.concat([node_j_repr, node_k_repr, node_l_repr], 1)
        node_ijk_repr = self.Dar_mlp1(node_ijk_repr)
        node_jkl_repr = self.Dar_mlp1(node_jkl_repr)
        pred = self.Dar_mlp2(paddle.concat([node_ijk_repr, node_jkl_repr], 1))
        Dar_dist_id = paddle.cast(((feed_dict['Da_bond_angle'] / np.pi)) * self.Dar_vocab, 'int64')
        loss = self.Dar_loss(pred, Dar_dist_id)

        node_i_repr_extra = paddle.gather(node_repr, feed_dict['Da_node_i_extra'])
        node_j_repr_extra = paddle.gather(node_repr, feed_dict['Da_node_j_extra'])
        node_k_repr_extra = paddle.gather(node_repr, feed_dict['Da_node_k_extra'])
        node_l_repr_extra = paddle.gather(node_repr, feed_dict['Da_node_l_extra'])
        node_ijk_repr_extra = paddle.concat([node_i_repr_extra, node_j_repr_extra, node_k_repr_extra], 1)
        node_kjl_repr_extra = paddle.concat([node_k_repr_extra, node_j_repr_extra, node_l_repr_extra], 1)
        node_ijk_repr_extra = self.Dar_mlp1(node_ijk_repr_extra)
        node_kjl_repr_extra = self.Dar_mlp1(node_kjl_repr_extra)
        pred_extra = self.Dar_mlp2(paddle.concat([node_ijk_repr_extra, node_kjl_repr_extra], 1))
        Dar_dist_id_extra = paddle.cast((feed_dict['Da_bond_angle_extra'] / np.pi) * self.Dar_vocab, 'int64')
        loss_extra = self.Dar_loss_extra(pred_extra, Dar_dist_id_extra)

        return loss + loss_extra

    def _get_Blr_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        pred = self.Blr_mlp(node_ij_repr)
        loss = self.Blr_loss(pred, feed_dict['Bl_bond_length'])
        return loss

    def _get_Adc_loss(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        logits = self.Adc_mlp(node_ij_repr)
        loss = self.Adc_loss(logits, feed_dict['Ad_atom_dist'])
        return loss

    def _get_Cl_loss(self, x1, x2, mols, rms=None):
        return WeightedNTXentLoss_func(x1, x2, mols, rms=rms)


    def forward(self, graph_dict, feed_dict, fp_score, return_subloss=False, dwa=None):
        """
        Build the network.
        """
        node_repr, edge_repr, graph_repr = self.compound_encoder.forward(
            graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'], graph_dict['dihes_angle_graph'], 1)
        node_repr_conf_cl, edge_repr_conf_cl, graph_repr_conf_cl = self.compound_encoder.forward(
            graph_dict['atom_bond_graph_conf_cl'], graph_dict['bond_angle_graph_conf_cl'],
            graph_dict['dihes_angle_graph_conf_cl'], 1)
        masked_node_repr, masked_edge_repr, masked_graph_repr = self.compound_encoder.forward(
            graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'],
            graph_dict['masked_dihes_angle_graph'], 1)
        sub_losses = {}
        if 'Cm' in self.pretrain_tasks:
            sub_losses['Cm_loss'] = self._get_Cm_loss(feed_dict, node_repr)
            sub_losses['Cm_loss'] += self._get_Cm_loss(feed_dict, masked_node_repr)
        if 'En' in self.pretrain_tasks:
            sub_losses['En_loss'] = self._get_En_loss(feed_dict, graph_repr)
            sub_losses['En_loss'] += self._get_En_loss(feed_dict, graph_repr_conf_cl)
        if 'Fg' in self.pretrain_tasks:
            sub_losses['Fg_loss'] = self._get_Fg_loss(feed_dict, graph_repr)
            sub_losses['Fg_loss'] += self._get_Fg_loss(feed_dict, masked_graph_repr)
        if 'Bar' in self.pretrain_tasks:
            sub_losses['Bar_loss'] = self._get_Bar_loss(feed_dict, node_repr)
            sub_losses['Bar_loss'] += self._get_Bar_loss(feed_dict, masked_node_repr)
        if 'Dar' in self.pretrain_tasks:
            sub_losses['Dar_loss'] = self._get_Dar_loss(feed_dict, node_repr)
            sub_losses['Dar_loss'] += self._get_Dar_loss(feed_dict, masked_node_repr)
        if 'Blr' in self.pretrain_tasks:
            sub_losses['Blr_loss'] = self._get_Blr_loss(feed_dict, node_repr)
            sub_losses['Blr_loss'] += self._get_Blr_loss(feed_dict, masked_node_repr)
        if 'Adc' in self.pretrain_tasks:
            sub_losses['Adc_loss'] = self._get_Adc_loss(feed_dict, node_repr)
            sub_losses['Adc_loss'] += self._get_Adc_loss(feed_dict, masked_node_repr)
        if 'Cl' in self.pretrain_tasks:
            sub_losses['Cl_loss'] = self._get_Cl_loss(F.normalize(graph_repr_conf_cl, axis=1),
                                                      F.normalize(masked_graph_repr, axis=1), fp_score,
                                                      rms=feed_dict["rms"])
        loss = 0
        cnt = 0

        for name in sub_losses:
            precision = paddle.exp(-self.coefs[cnt] * 2) / 2.0
            loss += precision * sub_losses[name] + paddle.log(paddle.exp(self.coefs[cnt]) + 1)
            cnt = cnt + 1
        if return_subloss:
            return loss, sub_losses, self.coefs
        else:
            return loss


