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
| Featurizers for pretrain-gnn.

| Adapted from https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/utils.py
"""

import numpy as np
from copy import deepcopy
import pgl
from sklearn.metrics import pairwise_distances
import hashlib
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d_all_cl_conf
import copy
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem

def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph_all_node(
        g,
        superedge_g,
        dihesedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0,
        subgraph_num=None,
        version='gem'):
    """tbd"""

    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)

        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask dihesedge_g
    full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
    masked_dihesedge_indices = []
    for bond_index in masked_superedge_indices:
        left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
        right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
        masked_dihesedge_indices.append(np.append(left_indices, right_indices))
    masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
    for name in dihesedge_g.edge_feat:
        dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
    return [g, superedge_g, dihesedge_g, target_atom_indices, target_labels]


def mask_context_of_geognn_graph_all_subgraph(
        g,
        superedge_g,
        dihesedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0,
        subgraph_num=None,
        version='gem'):

    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_bond_indices = np.arange(E)
    start = random.randint(0, N - 1)
    target_atom_indices = np.array([start])
    if True:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        cnt = 0
        while len(target_atom_indices) < masked_size:
            left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == target_atom_indices[cnt]]
            right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == target_atom_indices[cnt]]
            left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
            right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
            nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
            target_atom_indices = np.concatenate([target_atom_indices, nei_atom_indices], 0)
            target_atom_indices = pd.DataFrame(target_atom_indices)[0].unique()
            cnt = cnt + 1
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)

        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask dihesedge_g
    full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
    masked_dihesedge_indices = []
    for bond_index in masked_superedge_indices:
        left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
        right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
        masked_dihesedge_indices.append(np.append(left_indices, right_indices))
    masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
    for name in dihesedge_g.edge_feat:
        dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
    return [g, superedge_g, dihesedge_g, target_atom_indices, target_labels]

import random
import pandas as pd
def mask_context_of_geognn_graph_all_node_and_subgraph(
        g,
        superedge_g,
        dihesedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0,
        subgraph_num=None,
        version='gem'):
    """tbd"""

    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_bond_indices = np.arange(E)
    start = random.randint(0, N - 1)
    target_atom_indices = np.array([start])
    if True:
        masked_size = max(1, int(N * mask_ratio / 2))  # at least 1 atom will be selected.
        cnt = 0
        while len(target_atom_indices) < masked_size:
            left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == target_atom_indices[cnt]]
            right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == target_atom_indices[cnt]]
            left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
            right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
            nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
            target_atom_indices = np.concatenate([target_atom_indices, nei_atom_indices], 0)
            target_atom_indices = pd.DataFrame(target_atom_indices)[0].unique()
            cnt = cnt + 1

    masked_size_node = max(1, int(N * mask_ratio / 2))
    if masked_size_node > 0:
        atom_remain_indices = [i for i in range(N) if i not in target_atom_indices]
        mask_nodes_indices_2 = random.sample(atom_remain_indices, masked_size_node)
        target_atom_indices = target_atom_indices + mask_nodes_indices_2
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)

        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask dihesedge_g
    full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
    masked_dihesedge_indices = []
    for bond_index in masked_superedge_indices:
        left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
        right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
        masked_dihesedge_indices.append(np.append(left_indices, right_indices))
    masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
    for name in dihesedge_g.edge_feat:
        dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
    return [g, superedge_g, dihesedge_g, target_atom_indices, target_labels]


def mask_context_cl_of_geognn_graph_all_node(
        g,
        superedge_g,
        dihesedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0):
    """tbd"""

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask dihesedge_g
    full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
    masked_dihesedge_indices = []
    for bond_index in masked_superedge_indices:
        left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
        right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
        masked_dihesedge_indices.append(np.append(left_indices, right_indices))
    masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
    for name in dihesedge_g.edge_feat:
        dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
    return [g, superedge_g, dihesedge_g]


def mask_context_cl_of_geognn_graph_all_subgraph(
        g,
        superedge_g,
        dihesedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0):
    """tbd"""

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_bond_indices = np.arange(E)

    start = random.randint(0, N - 1)
    target_atom_indices = np.array([start])
    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))
        cnt = 0
        while len(target_atom_indices) < masked_size:
            left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == target_atom_indices[cnt]]
            right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == target_atom_indices[cnt]]
            left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
            right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
            nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
            target_atom_indices = np.concatenate([target_atom_indices, nei_atom_indices], 0)
            target_atom_indices = pd.DataFrame(target_atom_indices)[0].unique()
            cnt = cnt + 1
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in superedge_g.edge_feat:
        superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask dihesedge_g
    full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
    masked_dihesedge_indices = []
    for bond_index in masked_superedge_indices:
        left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
        right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
        masked_dihesedge_indices.append(np.append(left_indices, right_indices))
    masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
    for name in dihesedge_g.edge_feat:
        dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
    return [g, superedge_g, dihesedge_g]


def get_pretrain_bond_angle(edges, atom_poses):
    """tbd"""

    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle

    def _add_item(node_i_indices, node_j_indices, node_k_indices, bond_angles, node_i_index, node_j_index,
                  node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(
                    node_i_indices, node_j_indices, node_k_indices, bond_angles,
                    a1, a0, b1)
            if a0 == b1:
                _add_item(
                    node_i_indices, node_j_indices, node_k_indices, bond_angles,
                    a1, a0, b0)
            if a1 == b0:
                _add_item(
                    node_i_indices, node_j_indices, node_k_indices, bond_angles,
                    a0, a1, b1)
            if a1 == b1:
                _add_item(
                    node_i_indices, node_j_indices, node_k_indices, bond_angles,
                    a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]


class GeoPredTransformFn(object):
    """Gen features for downstream model"""

    def __init__(self, pretrain_tasks, mask_ratio):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio

    def prepare_pretrain_task(self, data):
        node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(data['edges'], data['atom_pos'])
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles
        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])
        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])
        return data

    def __call__(self, raw_data):
        smiles = raw_data
        print('smiles', smiles)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data, data_1, rms = mol_to_geognn_graph_data_MMFF3d_all_cl_conf(mol)
        data = self.prepare_pretrain_task(data)
        data['smiles'] = smiles
        data['rms'] = np.array([rms])
        d = list(data_1.keys())
        d_copy = copy.deepcopy(d)
        for i in list(d_copy):
            data_1.update({i + '_conf_cl': data.pop(i)})
        data = dict(data, **data_1)
        return data


class GeoPredCollateFn_all_cl(object):
    """tbd"""

    def __init__(self,
                 atom_names,
                 bond_names,
                 bond_float_names,
                 bond_angle_float_names,
                 pretrain_tasks,
                 mask_ratio,
                 Cm_vocab):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.Cm_vocab = Cm_vocab
        self.bond_angle_float_names = bond_angle_float_names
        self.face_angle_float_names = ["dihes_angle"]

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihes_angle_graph_list = []
        atom_bond_graph_conf_cl_list = []
        bond_angle_graph_conf_cl_list = []
        dihes_angle_graph_conf_cl_list = []
        atom_bond_graph_mask_cl_list = []
        bond_angle_graph_mask_cl_list = []
        dihes_angle_graph_mask_cl_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_dihes_angle_graph_list = []
        rms_list = []
        mol_list = []
        Cm_node_i = []
        Cm_context_id = []
        Fg_morgan = []
        Fg_daylight = []
        Fg_maccs = []
        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        Ba_bond_angle = []
        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []
        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []
        Da_node_i = []
        Da_node_j = []
        Da_node_k = []
        Da_node_l = []
        Da_bond_angle = []
        Da_node_i_extra = []
        Da_node_j_extra = []
        Da_node_k_extra = []
        Da_node_l_extra = []
        Da_bond_angle_extra = []

        node_count = 0
        for data in batch_data_list:
            N = len(data[self.atom_names[0]])
            E = len(data['edges'])
            rms_list.append(data['rms'][0])
            mol_list.append(Chem.MolFromSmiles(data['smiles']))
            data['dihes_angle'] = np.nan_to_num(data['dihes_angle'])
            data['dihes_angle' + '_conf_cl'] = np.nan_to_num(data['dihes_angle' + '_conf_cl'])
            data['dihes_angle_extra'] = np.nan_to_num(data['dihes_angle_extra'])
            data['dihes_angle_extra' + '_conf_cl'] = np.nan_to_num(data['dihes_angle_extra' + '_conf_cl'])
            data['DihesAngleGraph_edges'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra'], data['DihesAngleGraph_edges']],
                axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle_extra'], data['dihes_angle']], axis=0)
            data['DihesAngleGraph_edges' + '_conf_cl'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra' + '_conf_cl'], data['DihesAngleGraph_edges' + '_conf_cl']], axis=0)
            data['dihes_angle' + '_conf_cl'] = np.concatenate(
                [data['dihes_angle_extra' + '_conf_cl'], data['dihes_angle' + '_conf_cl']], axis=0)
            len1 = len(data['edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges'] = np.concatenate([data['BondAngleGraph_edges'], a], axis=0)
            data['bond_angle'] = np.concatenate([data['bond_angle'], b], axis=0)
            len1 = len(data['edges' + '_conf_cl'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges' + '_conf_cl'] = np.concatenate([data['BondAngleGraph_edges' + '_conf_cl'], a],
                                                                       axis=0)
            data['bond_angle' + '_conf_cl'] = np.concatenate([data['bond_angle' + '_conf_cl'], b], axis=0)
            len1 = len(data['BondAngleGraph_edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges'] = np.concatenate([data['DihesAngleGraph_edges'], a], axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle'], b], axis=0)
            len1 = len(data['BondAngleGraph_edges' + '_conf_cl'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges' + '_conf_cl'] = np.concatenate([data['DihesAngleGraph_edges' + '_conf_cl'], a],
                                                                        axis=0)
            data['dihes_angle' + '_conf_cl'] = np.concatenate([data['dihes_angle' + '_conf_cl'], b], axis=0)
            data['dihes_angle' + '_conf_cl'] = np.abs(data['dihes_angle' + '_conf_cl'])
            data['dihes_angle'] = np.abs(data['dihes_angle'])

            ab_g = pgl.graph.Graph(num_nodes=N,
                                   edges=data['edges'],
                                   node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                                   edge_feat={name: data[name].reshape([-1, 1]) for name in
                                              self.bond_names + self.bond_float_names})
            ba_g = pgl.graph.Graph(num_nodes=E,
                                   edges=data['BondAngleGraph_edges'],
                                   node_feat={},
                                   edge_feat={name: data[name].reshape([-1, 1]) for name in
                                              self.bond_angle_float_names})
            da_g = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges']),
                edges=data['DihesAngleGraph_edges'],
                node_feat={},
                edge_feat={name: data[name].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

            ab_g_conf_cl = pgl.graph.Graph(num_nodes=N,
                                           edges=data['edges' + '_conf_cl'],
                                           node_feat={name: data[name + '_conf_cl'].reshape([-1, 1]) for name in
                                                      self.atom_names},
                                           edge_feat={name: data[name + '_conf_cl'].reshape([-1, 1]) for name in
                                                      self.bond_names + self.bond_float_names})
            ba_g_conf_cl = pgl.graph.Graph(num_nodes=E,
                                           edges=data['BondAngleGraph_edges' + '_conf_cl'],
                                           node_feat={},
                                           edge_feat={name: data[name + '_conf_cl'].reshape([-1, 1]) for name in
                                                      self.bond_angle_float_names})
            da_g_conf_cl = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges' + '_conf_cl']),
                edges=data['DihesAngleGraph_edges' + '_conf_cl'],
                node_feat={},
                edge_feat={name: data[name + '_conf_cl'].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

            masked_ab_g, masked_ba_g, masked_da_g, mask_node_i, context_id = mask_context_of_geognn_graph_all_node(
                ab_g, ba_g, da_g, mask_ratio=0.1, subgraph_num=self.Cm_vocab)
            masked_cl_ab_g, masked_cl_ba_g, masked_cl_da_g = mask_context_cl_of_geognn_graph_all_node(
                ab_g, ba_g, da_g, mask_ratio=self.mask_ratio)

            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihes_angle_graph_list.append(da_g)

            atom_bond_graph_conf_cl_list.append(ab_g_conf_cl)
            bond_angle_graph_conf_cl_list.append(ba_g_conf_cl)
            dihes_angle_graph_conf_cl_list.append(da_g_conf_cl)

            atom_bond_graph_mask_cl_list.append(masked_cl_ab_g)
            bond_angle_graph_mask_cl_list.append(masked_cl_ba_g)
            dihes_angle_graph_mask_cl_list.append(masked_cl_da_g)

            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_dihes_angle_graph_list.append(masked_da_g)

            if 'Cm' in self.pretrain_tasks or 'Cm1' in self.pretrain_tasks:
                Cm_node_i.append(mask_node_i + node_count)
                Cm_context_id.append(context_id)
            if 'Fg' in self.pretrain_tasks:
                Fg_morgan.append(data['morgan_fp'])
                Fg_daylight.append(data['daylight_fg_counts'])
                Fg_maccs.append(data['maccs_fp'])
            if 'Bar' in self.pretrain_tasks:
                Ba_node_i.append(data['Ba_node_i'] + node_count)
                Ba_node_j.append(data['Ba_node_j'] + node_count)
                Ba_node_k.append(data['Ba_node_k'] + node_count)
                Ba_bond_angle.append(data['Ba_bond_angle'])
            if 'Blr' in self.pretrain_tasks:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
                Bl_bond_length.append(data['Bl_bond_length'])
            if 'Adc' in self.pretrain_tasks:
                Ad_node_i.append(data['Ad_node_i'] + node_count)
                Ad_node_j.append(data['Ad_node_j'] + node_count)
                Ad_atom_dist.append(data['Ad_atom_dist'])
            if 'Dar' in self.pretrain_tasks:
                if len(data['dihes_angle']) != 0:
                    Da_node_i.append(data['dihes_angle_node_i'] + node_count)
                    Da_node_j.append(data['dihes_angle_node_j'] + node_count)
                    Da_node_k.append(data['dihes_angle_node_k'] + node_count)
                    Da_node_l.append(data['dihes_angle_node_l'] + node_count)
                    Da_bond_angle_temp = data['dihes_angle'][
                                         len(data['dihes_angle_node_i_extra']):len(
                                             data['dihes_angle_node_i_extra']) + len(
                                             data['dihes_angle_node_i'])]
                    # Da_bond_angle_temp = Da_bond_angle_temp[0::2]
                    Da_bond_angle_temp = np.abs(Da_bond_angle_temp) * np.pi / 180
                    Da_bond_angle.append(Da_bond_angle_temp)

                if len(data['dihes_angle_extra']) != 0:
                    Da_node_i_extra.append(data['dihes_angle_node_i_extra'] + node_count)
                    Da_node_j_extra.append(data['dihes_angle_node_j_extra'] + node_count)
                    Da_node_k_extra.append(data['dihes_angle_node_k_extra'] + node_count)
                    Da_node_l_extra.append(data['dihes_angle_node_l_extra'] + node_count)
                    Da_bond_angle_temp_extra = data['dihes_angle_extra']
                    Da_bond_angle_temp_extra = np.abs(Da_bond_angle_temp_extra) * np.pi / 180
                    Da_bond_angle_extra.append(Da_bond_angle_temp_extra)

            node_count += N

        graph_dict = {}
        feed_dict = {}

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        graph_dict['atom_bond_graph'] = atom_bond_graph

        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        graph_dict['bond_angle_graph'] = bond_angle_graph

        dihes_angle_graph = pgl.Graph.batch(dihes_angle_graph_list)
        self._flat_shapes(dihes_angle_graph.node_feat)
        self._flat_shapes(dihes_angle_graph.edge_feat)
        graph_dict['dihes_angle_graph'] = dihes_angle_graph

        atom_bond_graph_conf_cl = pgl.Graph.batch(atom_bond_graph_conf_cl_list)
        self._flat_shapes(atom_bond_graph_conf_cl.node_feat)
        self._flat_shapes(atom_bond_graph_conf_cl.edge_feat)
        graph_dict['atom_bond_graph_conf_cl'] = atom_bond_graph_conf_cl

        bond_angle_graph_conf_cl = pgl.Graph.batch(bond_angle_graph_conf_cl_list)
        self._flat_shapes(bond_angle_graph_conf_cl.node_feat)
        self._flat_shapes(bond_angle_graph_conf_cl.edge_feat)
        graph_dict['bond_angle_graph_conf_cl'] = bond_angle_graph_conf_cl

        dihes_angle_graph_conf_cl = pgl.Graph.batch(dihes_angle_graph_conf_cl_list)
        self._flat_shapes(dihes_angle_graph_conf_cl.node_feat)
        self._flat_shapes(dihes_angle_graph_conf_cl.edge_feat)
        graph_dict['dihes_angle_graph_conf_cl'] = dihes_angle_graph_conf_cl

        atom_bond_graph_mask_cl = pgl.Graph.batch(atom_bond_graph_mask_cl_list)
        self._flat_shapes(atom_bond_graph_mask_cl.node_feat)
        self._flat_shapes(atom_bond_graph_mask_cl.edge_feat)
        graph_dict['atom_bond_graph_mask_cl'] = atom_bond_graph_mask_cl

        bond_angle_graph_mask_cl = pgl.Graph.batch(bond_angle_graph_mask_cl_list)
        self._flat_shapes(bond_angle_graph_mask_cl.node_feat)
        self._flat_shapes(bond_angle_graph_mask_cl.edge_feat)
        graph_dict['bond_angle_graph_mask_cl'] = bond_angle_graph_mask_cl

        dihes_angle_graph_mask_cl = pgl.Graph.batch(dihes_angle_graph_mask_cl_list)
        self._flat_shapes(dihes_angle_graph_mask_cl.node_feat)
        self._flat_shapes(dihes_angle_graph_mask_cl.edge_feat)
        graph_dict['dihes_angle_graph_mask_cl'] = dihes_angle_graph_mask_cl

        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph

        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat)
        graph_dict['masked_bond_angle_graph'] = masked_bond_angle_graph

        masked_dihes_angle_graph = pgl.Graph.batch(masked_dihes_angle_graph_list)
        self._flat_shapes(masked_dihes_angle_graph.node_feat)
        self._flat_shapes(masked_dihes_angle_graph.edge_feat)
        graph_dict['masked_dihes_angle_graph'] = masked_dihes_angle_graph

        feed_dict['rms'] = np.array(rms_list, 'float32')
        if 'Cm' in self.pretrain_tasks or 'Cm1' in self.pretrain_tasks:
            feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')
        if 'Fg' in self.pretrain_tasks:
            feed_dict['Fg_morgan'] = np.array(Fg_morgan, 'float32')
            feed_dict['Fg_daylight'] = (np.array(Fg_daylight) > 0).astype('float32')  # >1: 1x
            feed_dict['Fg_maccs'] = np.array(Fg_maccs, 'float32')
        if 'Bar' in self.pretrain_tasks:
            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
            feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')
        if 'Dar' in self.pretrain_tasks:
            feed_dict['Da_node_i'] = np.concatenate(Da_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_j'] = np.concatenate(Da_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_k'] = np.concatenate(Da_node_k, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_l'] = np.concatenate(Da_node_l, 0).reshape(-1).astype('int64')
            feed_dict['Da_bond_angle'] = np.concatenate(Da_bond_angle, 0).reshape(-1, 1).astype('float32')
            feed_dict['Da_node_i_extra'] = np.concatenate(Da_node_i_extra, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_j_extra'] = np.concatenate(Da_node_j_extra, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_k_extra'] = np.concatenate(Da_node_k_extra, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_l_extra'] = np.concatenate(Da_node_l_extra, 0).reshape(-1).astype('int64')
            feed_dict['Da_bond_angle_extra'] = np.concatenate(Da_bond_angle_extra, 0).reshape(-1, 1).astype('float32')
        if 'Blr' in self.pretrain_tasks:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')
        if 'Adc' in self.pretrain_tasks:
            feed_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')

        fp_score = np.zeros((len(mol_list), len(mol_list) - 1))
        fps = [AllChem.GetMorganFingerprint(Chem.AddHs(x), 2, useFeatures=True) for x in mol_list]
        for i in range(len(mol_list)):
            for j in range(i + 1, len(mol_list)):
                fp_sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                fp_score[i, j - 1] = fp_sim
                fp_score[j, i] = fp_sim
        return graph_dict, feed_dict, fp_score
