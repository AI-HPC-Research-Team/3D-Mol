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
import networkx as nx
from copy import deepcopy
import pgl
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from sklearn.metrics import pairwise_distances
import hashlib
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d_all, mol_to_geognn_graph_data_MMFF3d_all_cl_conf, \
    mol_to_geognn_graph_data_MMFF3d_all_cl_conf_eng
import copy
from rdkit import Chem
# from utils.compound_tools import Compound3DKit
from rdkit import DataStructs, Chem
import paddle.nn.functional as F
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt


def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(
        g,
        superedge_g,
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
    return [g, superedge_g, target_atom_indices, target_labels]


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


def mask_context_of_geognn_graph_all_node_limit(
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


def mask_context_of_geognn_graph_all_node_ram(
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

    if superedge_g.num_nodes > 0:
        masked_size = max(1, int(superedge_g.num_edges * mask_ratio))  # at least 1 bond will be selected.
        masked_bond_indices = np.random.choice(full_bond_indices, size=masked_size, replace=False)
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

        if dihesedge_g.num_nodes > 0:
            masked_size = max(1, int(N * mask_ratio))  # at least 1 face will be selected.
            masked_superedge_indices = np.random.choice(full_superedge_indices, size=masked_size, replace=False)
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


def mask_context_of_geognn_graph_all_node_rms(
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

    rms = random.uniform(0, 0.1)
    # masked_size = max(int(N * rms), 1)
    masked_size = max(int(N * (rms + 0.05)), 1)
    rms = rms * 80
    # masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
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
    return [g, superedge_g, dihesedge_g, target_atom_indices, target_labels, rms]


import pandas as pd


def mask_context_of_geognn_graph_all_subgraph(
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
    # full_atom_indices = np.arange(N)
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


import random


# def mask_context_cl_of_geognn_graph_all(
#         g,
#         superedge_g,
#         dihesedge_g,
#         target_atom_indices=None,
#         mask_ratio=None,
#         mask_value=0):
#     """tbd"""
#
#     g = deepcopy(g)
#     N = g.num_nodes
#     start = random.randint(0, N - 1)
#     E = g.num_edges
#     rms = random.uniform(0, mask_ratio)
#     # num_subg_mask = max(int(N * rms), 1)
#     num_subg_mask = max(int(N * mask_ratio), 1)
#
#     full_atom_indices = np.arange(N)
#     full_bond_indices = np.arange(E)
#     Cm_node_i = np.array([start])
#     masked_bond_indices = []
#
#     left_nei_bond_indices_st = full_bond_indices[g.edges[:, 0] == Cm_node_i[0]]
#     right_nei_bond_indices_st = full_bond_indices[g.edges[:, 1] == Cm_node_i[0]]
#     nei_bond_indices_st = np.append(left_nei_bond_indices_st, right_nei_bond_indices_st)
#     masked_bond_indices.append(nei_bond_indices_st)
#
#     cnt = 0
#     while len(Cm_node_i) < num_subg_mask:
#         left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == Cm_node_i[cnt]]
#         right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == Cm_node_i[cnt]]
#         nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
#         left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
#         right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
#         nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
#         Cm_node_i = np.concatenate([Cm_node_i, nei_atom_indices], 0)
#         masked_bond_indices.append(nei_bond_indices)
#         cnt = cnt + 1
#
#
#     masked_bond_indices = np.concatenate(masked_bond_indices, 0)
#     Cm_node_i = np.unique(Cm_node_i, axis=0)
#     masked_bond_indices = np.unique(masked_bond_indices, axis=0)
#
#     num_node_mask = max(int(mask_ratio * N) - len(Cm_node_i), 0)
#     if num_node_mask > 0:
#         res_atom_index = np.delete(full_atom_indices, Cm_node_i)
#         target_atom_indices = np.random.choice(res_atom_index, size=num_node_mask, replace=False)
#         Cm_node_j = []
#         masked_bond_indices_j = []
#         for atom_index in target_atom_indices:
#             left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
#             right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
#             nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
#             Cm_node_j.append([atom_index])
#             masked_bond_indices_j.append(nei_bond_indices)
#         Cm_node_j = np.concatenate(Cm_node_j, 0)
#         masked_bond_indices_j = np.concatenate(masked_bond_indices_j, 0)
#         Cm_node_i = np.concatenate([Cm_node_i, Cm_node_j], axis=0)
#         masked_bond_indices = np.concatenate([masked_bond_indices, masked_bond_indices_j], axis=0)
#         masked_bond_indices = np.unique(masked_bond_indices, axis=0)
#     if len(Cm_node_i)/N > 0.33:
#         Cm_node_i = []
#         masked_bond_indices = []
#     for name in g.node_feat:
#         g.node_feat[name][Cm_node_i] = mask_value
#     for name in g.edge_feat:
#         g.edge_feat[name][masked_bond_indices] = mask_value
#
#     # mask superedge_g
#     full_superedge_indices = np.arange(superedge_g.num_edges)
#     masked_superedge_indices = []
#     if len(masked_bond_indices) != 0:
#         for bond_index in masked_bond_indices:
#             left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
#             right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
#             masked_superedge_indices.append(np.append(left_indices, right_indices))
#         masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
#         for name in superedge_g.edge_feat:
#             superedge_g.edge_feat[name][masked_superedge_indices] = mask_value
#
#     # mask dihesedge_g
#     full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
#     masked_dihesedge_indices = []
#     if len(masked_superedge_indices) != 0:
#         for bond_index in masked_superedge_indices:
#             left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
#             right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
#             masked_dihesedge_indices.append(np.append(left_indices, right_indices))
#         masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
#         for name in dihesedge_g.edge_feat:
#             dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
#     return [g, superedge_g, dihesedge_g]
#
#
# def mask_context_cl_of_geognn_graph_all_rms(
#         g,
#         superedge_g,
#         dihesedge_g,
#         target_atom_indices=None,
#         mask_ratio=None,
#         mask_value=0):
#     """tbd"""
#
#     g = deepcopy(g)
#     N = g.num_nodes
#     start = random.randint(0, N - 1)
#     E = g.num_edges
#
#     rms = random.uniform(0, 0.1)
#     num_subg_mask = max(int(N * (rms + 0.1)), 1)
#     rms = rms * 80
#     full_atom_indices = np.arange(N)
#     full_bond_indices = np.arange(E)
#     Cm_node_i = np.array([start])
#     masked_bond_indices = []
#
#     left_nei_bond_indices_st = full_bond_indices[g.edges[:, 0] == Cm_node_i[0]]
#     right_nei_bond_indices_st = full_bond_indices[g.edges[:, 1] == Cm_node_i[0]]
#     nei_bond_indices_st = np.append(left_nei_bond_indices_st, right_nei_bond_indices_st)
#     masked_bond_indices.append(nei_bond_indices_st)
#
#     cnt = 0
#     while len(Cm_node_i) < num_subg_mask:
#         left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == Cm_node_i[cnt]]
#         right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == Cm_node_i[cnt]]
#         nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
#         left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
#         right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
#         nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)
#         Cm_node_i = np.concatenate([Cm_node_i, nei_atom_indices], 0)
#         masked_bond_indices.append(nei_bond_indices)
#         cnt = cnt + 1
#
#
#     masked_bond_indices = np.concatenate(masked_bond_indices, 0)
#     Cm_node_i = np.unique(Cm_node_i, axis=0)
#     masked_bond_indices = np.unique(masked_bond_indices, axis=0)
#
#     num_node_mask = max(int(mask_ratio * N) - len(Cm_node_i), 0)
#     if num_node_mask > 0:
#         res_atom_index = np.delete(full_atom_indices, Cm_node_i)
#         target_atom_indices = np.random.choice(res_atom_index, size=num_node_mask, replace=False)
#         Cm_node_j = []
#         masked_bond_indices_j = []
#         for atom_index in target_atom_indices:
#             left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
#             right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
#             nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
#             Cm_node_j.append([atom_index])
#             masked_bond_indices_j.append(nei_bond_indices)
#         Cm_node_j = np.concatenate(Cm_node_j, 0)
#         masked_bond_indices_j = np.concatenate(masked_bond_indices_j, 0)
#         Cm_node_i = np.concatenate([Cm_node_i, Cm_node_j], axis=0)
#         masked_bond_indices = np.concatenate([masked_bond_indices, masked_bond_indices_j], axis=0)
#         masked_bond_indices = np.unique(masked_bond_indices, axis=0)
#     if len(Cm_node_i)/N > 0.33:
#         Cm_node_i = []
#         masked_bond_indices = []
#     for name in g.node_feat:
#         g.node_feat[name][Cm_node_i] = mask_value
#     for name in g.edge_feat:
#         g.edge_feat[name][masked_bond_indices] = mask_value
#
#     # mask superedge_g
#     full_superedge_indices = np.arange(superedge_g.num_edges)
#     masked_superedge_indices = []
#     if len(masked_bond_indices) != 0:
#         for bond_index in masked_bond_indices:
#             left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
#             right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
#             masked_superedge_indices.append(np.append(left_indices, right_indices))
#         masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
#         for name in superedge_g.edge_feat:
#             superedge_g.edge_feat[name][masked_superedge_indices] = mask_value
#
#     # mask dihesedge_g
#     full_dihesedge_indices = np.arange(dihesedge_g.num_edges)
#     masked_dihesedge_indices = []
#     if len(masked_superedge_indices) != 0:
#         for bond_index in masked_superedge_indices:
#             left_indices = full_dihesedge_indices[dihesedge_g.edges[:, 0] == bond_index]
#             right_indices = full_dihesedge_indices[dihesedge_g.edges[:, 1] == bond_index]
#             masked_dihesedge_indices.append(np.append(left_indices, right_indices))
#         masked_dihesedge_indices = np.concatenate(masked_dihesedge_indices, 0)
#         for name in dihesedge_g.edge_feat:
#             dihesedge_g.edge_feat[name][masked_dihesedge_indices] = mask_value
#     return [g, superedge_g, dihesedge_g, rms]


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
    # full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    start = random.randint(0, N - 1)
    target_atom_indices = np.array([start])
    if target_atom_indices is None:
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


def mask_context_cl_of_geognn_graph_all_node_rms(
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

    rms = random.uniform(0, 0.1)
    # masked_size = max(int(N * rms), 1)
    masked_size = max(int(N * (rms + 0.05)), 1)
    rms = rms * 80

    if target_atom_indices is None:
        # masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
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
    return [g, superedge_g, dihesedge_g, rms]


def mask_context_cl_of_geognn_graph_all_subgraph_rms(
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
    # full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    rms = random.uniform(0, 0.1)
    masked_size = max(int(N * (rms + 0.1)), 1)
    rms = rms * 80

    start = random.randint(0, N - 1)
    target_atom_indices = np.array([start])
    if True:
        # masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
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
    return [g, superedge_g, dihesedge_g, rms]


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


from rdkit import Chem
from rdkit.Chem import BRICS


def fragment_recursive(mol):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) == 0:
            return None
        idxs, labs = list(zip(*bonds))
        dict_frag = {}
        # dict_frag = []
        for i in range(len(idxs)):
            dict_frag[idxs[i]] = labs[i]
            # dict_frag.append([idxs[i], labs[i]])
        return dict_frag
    except Exception as e:
        print(e)
        return None


def get_keys_by_value(dict_obj, value):
    keys = []
    for k, v in dict_obj.items():
        if v == value:
            keys.append(k)
    return keys


def mix_up(mol_1, mol_2):
    aspirin_1 = Chem.MolFromSmiles(mol_1)
    aspirin_2 = Chem.MolFromSmiles(mol_2)

    fragments_1 = fragment_recursive(aspirin_1)
    fragments_2 = fragment_recursive(aspirin_2)
    if fragments_2 is None or fragments_1 is None:
        return None

    tar_bond = set()

    for j in fragments_1.keys():
        for k in fragments_2.keys():
            tar_bond.add((j, k, fragments_1[j], fragments_2[k]))

    temp_mol = []
    flag_df = False
    for i in tar_bond:
        bond_1, bond_2, labs_1, labs_2 = i[0], i[1], i[2], i[3]

        break_bond_1 = aspirin_1.GetBondBetweenAtoms(bond_1[0], bond_1[1])
        break_bond_idx_1 = break_bond_1.GetIdx()
        break_bond_frag_1 = Chem.FragmentOnBonds(aspirin_1, bondIndices=[break_bond_idx_1], dummyLabels=[(0, 0)])
        head_1, tail_1 = Chem.GetMolFrags(break_bond_frag_1, asMols=True)

        smiles_head_1 = Chem.MolToSmiles(head_1)
        smiles_tail_1 = Chem.MolToSmiles(tail_1)
        smiles_head_1 = smiles_head_1.replace('*', '[' + labs_1[0] + '*]')
        smiles_tail_1 = smiles_tail_1.replace('*', '[' + labs_1[1] + '*]')

        break_bond_2 = aspirin_2.GetBondBetweenAtoms(bond_2[0], bond_2[1])
        break_bond_idx_2 = break_bond_2.GetIdx()
        break_bond_frag_2 = Chem.FragmentOnBonds(aspirin_2, bondIndices=[break_bond_idx_2], dummyLabels=[(0, 0)])
        head_2, tail_2 = Chem.GetMolFrags(break_bond_frag_2, asMols=True)

        smiles_head_2 = Chem.MolToSmiles(head_2)
        smiles_tail_2 = Chem.MolToSmiles(tail_2)
        smiles_head_2 = smiles_head_2.replace('*', '[' + labs_2[0] + '*]')
        smiles_tail_2 = smiles_tail_2.replace('*', '[' + labs_2[1] + '*]')

        try:
            gen_1 = BRICS.BRICSBuild([Chem.MolFromSmiles(smiles_head_1), Chem.MolFromSmiles(smiles_tail_2)],
                                     scrambleReagents=False)
            prods_1 = [next(gen_1) for x in range(1)]
            prods_1[0].UpdatePropertyCache(strict=False)
            new_mol_1 = Chem.MolFromSmiles(Chem.MolToSmiles(prods_1[0]))
            temp_mol.append((Chem.MolToSmiles(new_mol_1), smiles_head_1, smiles_tail_2))
        except:
            flag_df = False

        try:
            gen_2 = BRICS.BRICSBuild([Chem.MolFromSmiles(smiles_head_2), Chem.MolFromSmiles(smiles_tail_1)],
                                     scrambleReagents=False)
            prods_2 = [next(gen_2) for x in range(1)]
            prods_2[0].UpdatePropertyCache(strict=False)
            new_mol_2 = Chem.MolFromSmiles(Chem.MolToSmiles(prods_2[0]))
            temp_mol.append((Chem.MolToSmiles(new_mol_2), smiles_head_2, smiles_tail_1))
        except:
            flag_df = False

    if len(temp_mol) != 0:
        return temp_mol
    else:
        return None


def smiles_pros(mol_temp_smiles):
    index_temp = []
    for i in range(len(mol_temp_smiles) - 1):
        if mol_temp_smiles[i] == "*" and mol_temp_smiles[i + 1] == "]":
            index_temp.append(i)
    for i in index_temp[::-1]:
        j = i
        while True:
            j = j - 1
            if mol_temp_smiles[j] == "[":
                break
            mol_temp_smiles = mol_temp_smiles[:j] + mol_temp_smiles[j + 1:]
    mol_temp_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mol_temp_smiles))
    mol_temp_smiles = mol_temp_smiles.replace('*', "[H]")
    return mol_temp_smiles



class GeoPredTransformFn(object):
    """Gen features for downstream model"""

    def __init__(self, pretrain_tasks, mask_ratio):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """

        node_i, node_j, node_k, bond_angles = \
            get_pretrain_bond_angle(data['edges'], data['atom_pos'])
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
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data
        print('smiles', smiles)

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            print("error of turning smiles to mol", smiles)
            return None

        data, data_1, data_2, data_3, rms_12, rms_13, rms_14, rms_23, rms_24, rms_34, energy, energy_1, energy_2, \
        energy_3 = mol_to_geognn_graph_data_MMFF3d_all_cl_conf(mol)

        if data is None or data_1 is None or data_2 is None or data_3 is None:
            print("error of turning mol to data", smiles)
            return None

        data = self.prepare_pretrain_task(data)
        data_1 = self.prepare_pretrain_task(data_1)
        data_2 = self.prepare_pretrain_task(data_2)
        data_3 = self.prepare_pretrain_task(data_3)

        data['smiles'] = smiles

        data['rms_12'] = rms_12
        data['rms_13'] = rms_13
        data['rms_14'] = rms_14
        data['rms_23'] = rms_23
        data['rms_24'] = rms_24
        data['rms_34'] = rms_34
        data['energy'] = energy
        data['energy_1'] = energy_1
        data['energy_2'] = energy_2
        data['energy_3'] = energy_3


        d = list(data_1.keys())
        d_copy = copy.deepcopy(d)
        # for i in list(d_copy):
        #     data_1.update({i + '_conf_cl_1': data_1.pop(i)})
        #     data_2.update({i + '_conf_cl_2': data_2.pop(i)})
        #     data_3.update({i + '_conf_cl_3': data_3.pop(i)})
        for i in list(d_copy):
            data.update({i + '_conf_cl_1': data_1.pop(i)})
            data.update({i + '_conf_cl_2': data_2.pop(i)})
            data.update({i + '_conf_cl_3': data_3.pop(i)})
        # print(data)

        return data


# class GeoPredTransformFn_3D(object):
#     """Gen features for downstream model"""
#
#     def __init__(self, pretrain_tasks, mask_ratio):
#         self.pretrain_tasks = pretrain_tasks
#         self.mask_ratio = mask_ratio
#
#     def prepare_pretrain_task(self, data):
#         """
#         prepare data for pretrain task
#         """
#
#         node_i, node_j, node_k, bond_angles = \
#             get_pretrain_bond_angle(data['edges'], data['atom_pos'])
#         data['Ba_node_i'] = node_i
#         data['Ba_node_j'] = node_j
#         data['Ba_node_k'] = node_k
#         data['Ba_bond_angle'] = bond_angles
#
#         data['Bl_node_i'] = np.array(data['edges'][:, 0])
#         data['Bl_node_j'] = np.array(data['edges'][:, 1])
#         data['Bl_bond_length'] = np.array(data['bond_length'])
#
#         n = len(data['atom_pos'])
#         dist_matrix = pairwise_distances(data['atom_pos'])
#         indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
#         data['Ad_node_i'] = indice.reshape([-1, 1])
#         data['Ad_node_j'] = indice.T.reshape([-1, 1])
#         data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])
#         return data
#
#     def __call__(self, raw_data):
#         """
#         Gen features according to raw data and return a single graph data.
#         Args:
#             raw_data: It contains smiles and label,we convert smiles
#             to mol by rdkit,then convert mol to graph data.
#         Returns:
#             data: It contains reshape label and smiles.
#         """
#         smiles = raw_data
#         print('smiles', smiles)
#
#         mol = AllChem.MolFromSmiles(smiles)
#         if mol is None:
#             print("error of turning smiles to mol", smiles)
#             return None
#
#         data, data_1, data_2, data_3, rms_12, rms_13, rms_14, rms_23, rms_24, rms_34, energy, energy_1, energy_2, \
#         energy_3, fp3D = mol_to_geognn_graph_data_MMFF3d_all_cl_conf(mol)
#
#         if data is None or data_1 is None or data_2 is None or data_3 is None:
#             print("error of turning mol to data", smiles)
#             return None
#
#         data = self.prepare_pretrain_task(data)
#         data_1 = self.prepare_pretrain_task(data_1)
#         data_2 = self.prepare_pretrain_task(data_2)
#         data_3 = self.prepare_pretrain_task(data_3)
#
#         data['smiles'] = smiles
#
#         data['rms_12'] = rms_12
#         data['rms_13'] = rms_13
#         data['rms_14'] = rms_14
#         data['rms_23'] = rms_23
#         data['rms_24'] = rms_24
#         data['rms_34'] = rms_34
#         data['energy'] = energy
#         data['energy_1'] = energy_1
#         data['energy_2'] = energy_2
#         data['energy_3'] = energy_3
#
#         data['usrcat_1'], data['usrcat_2'], data['usrcat_3'], data['usrcat_4'] = fp3D["usrcat"]
#         data['rdf_1'], data['rdf_2'], data['rdf_3'], data['rdf_4'] = fp3D["rdf"]
#         data['autocorr3d_1'], data['autocorr3d_2'], data['autocorr3d_3'], data['autocorr3d_4'] = fp3D["autocorr3d"]
#         data['morse_1'], data['morse_2'], data['morse_3'], data['morse_4'] = fp3D["morse"]
#         data['whim_1'], data['whim_2'], data['whim_3'], data['whim_4'] = fp3D["whim"]
#         data['getaway_1'], data['getaway_2'], data['getaway_3'], data['getaway_4'] = fp3D["getaway"]
#         data['pharmacophore_1'], data['pharmacophore_2'], data['pharmacophore_3'], data['pharmacophore_4'] = fp3D["pharmacophore"]
#
#         d = list(data_1.keys())
#         d_copy = copy.deepcopy(d)
#         # for i in list(d_copy):
#         #     data_1.update({i + '_conf_cl_1': data_1.pop(i)})
#         #     data_2.update({i + '_conf_cl_2': data_2.pop(i)})
#         #     data_3.update({i + '_conf_cl_3': data_3.pop(i)})
#         for i in list(d_copy):
#             data.update({i + '_conf_cl_1': data_1.pop(i)})
#             data.update({i + '_conf_cl_2': data_2.pop(i)})
#             data.update({i + '_conf_cl_3': data_3.pop(i)})
#         # print(data)
#
#         return data

class GeoPredTransformFn_3D(object):
    """Gen features for downstream model"""

    def __init__(self, pretrain_tasks, mask_ratio, is_inference=False):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.is_inference = is_inference

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """

        node_i, node_j, node_k, bond_angles = \
            get_pretrain_bond_angle(data['edges'], data['atom_pos'])
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
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data['smiles']
        print('smiles_raw: ', smiles)

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            print("error of turning smiles to mol", smiles)
            return None

        data, data_1, data_2, data_3, rms_12, rms_13, rms_14, rms_23, rms_24, rms_34, energy, energy_1, energy_2, \
        energy_3, fp3D = mol_to_geognn_graph_data_MMFF3d_all_cl_conf(mol)

        if data is None or data_1 is None or data_2 is None or data_3 is None:
            print("error of turning mol to data", smiles)
            return None

        data = self.prepare_pretrain_task(data)
        data_1 = self.prepare_pretrain_task(data_1)
        data_2 = self.prepare_pretrain_task(data_2)
        data_3 = self.prepare_pretrain_task(data_3)

        data['smiles'] = smiles

        data['rms_12'] = rms_12
        data['rms_13'] = rms_13
        data['rms_14'] = rms_14
        data['rms_23'] = rms_23
        data['rms_24'] = rms_24
        data['rms_34'] = rms_34
        data['energy'] = energy
        data['energy_1'] = energy_1
        data['energy_2'] = energy_2
        data['energy_3'] = energy_3

        # data['usrcat_1'], data['usrcat_2'], data['usrcat_3'], data['usrcat_4'] = fp3D["usrcat"]
        data['rdf_1'], data['rdf_2'], data['rdf_3'], data['rdf_4'] = fp3D["rdf"]
        data['autocorr3d_1'], data['autocorr3d_2'], data['autocorr3d_3'], data['autocorr3d_4'] = fp3D["autocorr3d"]
        data['morse_1'], data['morse_2'], data['morse_3'], data['morse_4'] = fp3D["morse"]
        data['whim_1'], data['whim_2'], data['whim_3'], data['whim_4'] = fp3D["whim"]
        data['getaway_1'], data['getaway_2'], data['getaway_3'], data['getaway_4'] = fp3D["getaway"]
        # data['pharmacophore_1'], data['pharmacophore_2'], data['pharmacophore_3'], data['pharmacophore_4'] = fp3D["pharmacophore"]

        d = list(data_1.keys())
        d_copy = copy.deepcopy(d)
        # for i in list(d_copy):
        #     data_1.update({i + '_conf_cl_1': data_1.pop(i)})
        #     data_2.update({i + '_conf_cl_2': data_2.pop(i)})
        #     data_3.update({i + '_conf_cl_3': data_3.pop(i)})
        for i in list(d_copy):
            data.update({i + '_conf_cl_1': data_1.pop(i)})
            data.update({i + '_conf_cl_2': data_2.pop(i)})
            data.update({i + '_conf_cl_3': data_3.pop(i)})
        # print(data)
        if not self.is_inference:
            data['label'] = raw_data['label'].reshape([-1])

        return data


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / ((norm_vec1 * norm_vec2) + 0.0001)
    return similarity

import pickle
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
        """tbd"""

        with open('/userhome/FP3D_log/rdf_new_list.pkl', 'rb') as f:
            rdf_new_list = pickle.load(f)

        with open('/userhome/FP3D_log/autocorr3d_new_list.pkl', 'rb') as f:
            autocorr3d_new_list = pickle.load(f)

        with open('/userhome/FP3D_log/morse_new_list.pkl', 'rb') as f:
            morse_new_list = pickle.load(f)

        with open('/userhome/FP3D_log/whim_new_list.pkl', 'rb') as f:
            whim_new_list = pickle.load(f)

        rdf_max = np.array(rdf_new_list[0], 'float32')
        autocorr3d_max = np.array(autocorr3d_new_list[0], 'float32')
        morse_max = np.array(morse_new_list[0], 'float32')
        whim_max = np.array(whim_new_list[0], 'float32')

        rdf_min = np.array(rdf_new_list[1], 'float32')
        autocorr3d_min = np.array(autocorr3d_new_list[1], 'float32')
        morse_min = np.array(morse_new_list[1], 'float32')
        whim_min = np.array(whim_new_list[1], 'float32')

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihes_angle_graph_list = []
        atom_bond_graph_conf_cl_1_list = []
        bond_angle_graph_conf_cl_1_list = []
        dihes_angle_graph_conf_cl_1_list = []
        atom_bond_graph_conf_cl_2_list = []
        bond_angle_graph_conf_cl_2_list = []
        dihes_angle_graph_conf_cl_2_list = []
        atom_bond_graph_conf_cl_3_list = []
        bond_angle_graph_conf_cl_3_list = []
        dihes_angle_graph_conf_cl_3_list = []
        atom_bond_graph_mask_cl_list = []
        bond_angle_graph_mask_cl_list = []
        dihes_angle_graph_mask_cl_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_dihes_angle_graph_list = []
        rms_12_list = []
        rms_13_list = []
        rms_14_list = []
        rms_23_list = []
        rms_24_list = []
        rms_34_list = []
        en_list = []
        mol_list = []

        # rdf_fp_list_1 = []
        # morse_fp_list_1 = []
        # #        getaway_fp_list_1 = []
        # autocorr3d_fp_list_1 = []
        # whim_fp_list_1 = []


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

        Cm_node_i_cl = []
        Cm_context_id_cl = []
        Fg_morgan_cl = []
        Fg_daylight_cl = []
        Fg_maccs_cl = []
        Ba_node_i_cl = []
        Ba_node_j_cl = []
        Ba_node_k_cl = []
        Ba_bond_angle_cl = []
        Bl_node_i_cl = []
        Bl_node_j_cl = []
        Bl_bond_length_cl = []
        Ad_node_i_cl = []
        Ad_node_j_cl = []
        Ad_atom_dist_cl = []
        Da_node_i_cl = []
        Da_node_j_cl = []
        Da_node_k_cl = []
        Da_node_l_cl = []
        Da_bond_angle_cl = []
        Da_node_i_extra_cl = []
        Da_node_j_extra_cl = []
        Da_node_k_extra_cl = []
        Da_node_l_extra_cl = []
        Da_bond_angle_extra_cl = []

        node_count = 0
        for data in batch_data_list:

            N = len(data[self.atom_names[0]])
            E = len(data['edges'])

            # rms_atom_dis = np.linalg.norm(
            #     data['Ad_atom_dist'].reshape(-1, int(np.sqrt(len(data['Ad_atom_dist'])))) -
            #     data['Ad_atom_dist' + '_conf_cl_1'].reshape(-1, int(np.sqrt(len(data['Ad_atom_dist' + '_conf_cl_1'])))),

            # rms_atom_dis = cosine_similarity(data['Ad_atom_dist'], data['Ad_atom_dist' + '_conf_cl_1'])
            # rms_atom_dis = cosine_similarity(data['rdf_1'], data['rdf_2'])
            # rms_atom_dis = cosine_similarity(data['autocorr3d_1'], data['autocorr3d_2'])
            # rms_atom_dis = cosine_similarity(data['morse_1'], data['morse_2'])
            rms_atom_dis = cosine_similarity(data['whim_1'], data['whim_2'])
            # rms_atom_dis = cosine_similarity(data['getaway_1'], data['getaway_2'])
#            rms_atom_dis = data['rms_12']
            rms_12_list.append(rms_atom_dis)
            rms_13_list.append(data['rms_13'])
            rms_14_list.append(data['rms_14'])
            rms_23_list.append(data['rms_23'])
            rms_24_list.append(data['rms_24'])
            rms_34_list.append(data['rms_34'])
            en_list.append(data['energy'])

            # rdf_1 = (data['rdf_1'] - rdf_min) / (data['rdf_1'] - rdf_max)
            # rdf_2 = (data['rdf_2'] - rdf_min) / (data['rdf_2'] - rdf_max)
            # cos_sim = rdf_1.dot(rdf_2) / (np.linalg.norm(rdf_1) * np.linalg.norm(rdf_2))
            # rms_12_list.append(cos_sim)
            # morse_fp_list_1.append(10 * (data['morse_1'] - morse_min) / (data['morse_1'] - morse_max))
            # autocorr3d_fp_list_1.append(
            #     10 * (data['autocorr3d_1'] - autocorr3d_min) / (data['autocorr3d_1'] - autocorr3d_max))
            # whim_fp_list_1.append(10 * (data['whim_1'] - whim_min) / (data['whim_1'] - whim_max))
            #            getaway_fp_list_1.append(data['getaway_1'])

            mol_list.append(Chem.MolFromSmiles(data['smiles']))

            data['dihes_angle'] = np.nan_to_num(data['dihes_angle'])
            data['dihes_angle' + '_conf_cl_1'] = np.nan_to_num(data['dihes_angle' + '_conf_cl_1'])
            data['dihes_angle' + '_conf_cl_2'] = np.nan_to_num(data['dihes_angle' + '_conf_cl_2'])
            data['dihes_angle' + '_conf_cl_3'] = np.nan_to_num(data['dihes_angle' + '_conf_cl_3'])

            data['dihes_angle_extra'] = np.nan_to_num(data['dihes_angle_extra'])
            data['dihes_angle_extra' + '_conf_cl_1'] = np.nan_to_num(data['dihes_angle_extra' + '_conf_cl_1'])
            data['dihes_angle_extra' + '_conf_cl_2'] = np.nan_to_num(data['dihes_angle_extra' + '_conf_cl_2'])
            data['dihes_angle_extra' + '_conf_cl_3'] = np.nan_to_num(data['dihes_angle_extra' + '_conf_cl_3'])

            data['DihesAngleGraph_edges'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra'], data['DihesAngleGraph_edges']], axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle_extra'], data['dihes_angle']], axis=0)

            data['DihesAngleGraph_edges' + '_conf_cl_1'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra' + '_conf_cl_1'], data['DihesAngleGraph_edges' + '_conf_cl_1']], axis=0)
            data['dihes_angle' + '_conf_cl_1'] = np.concatenate(
                [data['dihes_angle_extra' + '_conf_cl_1'], data['dihes_angle' + '_conf_cl_1']], axis=0)

            data['DihesAngleGraph_edges' + '_conf_cl_2'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra' + '_conf_cl_2'], data['DihesAngleGraph_edges' + '_conf_cl_2']],
                axis=0)
            data['dihes_angle' + '_conf_cl_2'] = np.concatenate(
                [data['dihes_angle_extra' + '_conf_cl_2'], data['dihes_angle' + '_conf_cl_2']], axis=0)

            data['DihesAngleGraph_edges' + '_conf_cl_3'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra' + '_conf_cl_3'], data['DihesAngleGraph_edges' + '_conf_cl_3']],
                axis=0)
            data['dihes_angle' + '_conf_cl_3'] = np.concatenate(
                [data['dihes_angle_extra' + '_conf_cl_3'], data['dihes_angle' + '_conf_cl_3']], axis=0)

            len1 = len(data['edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges'] = np.concatenate([data['BondAngleGraph_edges'], a], axis=0)
            data['bond_angle'] = np.concatenate([data['bond_angle'], b], axis=0)

            len1 = len(data['edges' + '_conf_cl_1'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges' + '_conf_cl_1'] = np.concatenate(
                [data['BondAngleGraph_edges' + '_conf_cl_1'], a], axis=0)
            data['bond_angle' + '_conf_cl_1'] = np.concatenate([data['bond_angle' + '_conf_cl_1'], b], axis=0)

            len1 = len(data['edges' + '_conf_cl_2'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges' + '_conf_cl_2'] = np.concatenate(
                [data['BondAngleGraph_edges' + '_conf_cl_2'], a], axis=0)
            data['bond_angle' + '_conf_cl_2'] = np.concatenate([data['bond_angle' + '_conf_cl_2'], b], axis=0)

            len1 = len(data['edges' + '_conf_cl_3'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges' + '_conf_cl_3'] = np.concatenate(
                [data['BondAngleGraph_edges' + '_conf_cl_3'], a], axis=0)
            data['bond_angle' + '_conf_cl_3'] = np.concatenate([data['bond_angle' + '_conf_cl_3'], b], axis=0)

            len1 = len(data['BondAngleGraph_edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges'] = np.concatenate([data['DihesAngleGraph_edges'], a], axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle'], b], axis=0)

            len1 = len(data['BondAngleGraph_edges' + '_conf_cl_1'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges' + '_conf_cl_1'] = np.concatenate(
                [data['DihesAngleGraph_edges' + '_conf_cl_1'], a], axis=0)
            data['dihes_angle' + '_conf_cl_1'] = np.concatenate([data['dihes_angle' + '_conf_cl_1'], b], axis=0)

            len1 = len(data['BondAngleGraph_edges' + '_conf_cl_2'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges' + '_conf_cl_2'] = np.concatenate(
                [data['DihesAngleGraph_edges' + '_conf_cl_2'], a], axis=0)
            data['dihes_angle' + '_conf_cl_2'] = np.concatenate([data['dihes_angle' + '_conf_cl_2'], b], axis=0)

            len1 = len(data['BondAngleGraph_edges' + '_conf_cl_3'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges' + '_conf_cl_3'] = np.concatenate(
                [data['DihesAngleGraph_edges' + '_conf_cl_3'], a], axis=0)
            data['dihes_angle' + '_conf_cl_3'] = np.concatenate([data['dihes_angle' + '_conf_cl_3'], b], axis=0)

            data['dihes_angle'] = np.abs(data['dihes_angle'])
            data['dihes_angle' + '_conf_cl_1'] = np.abs(data['dihes_angle' + '_conf_cl_1'])
            data['dihes_angle' + '_conf_cl_2'] = np.abs(data['dihes_angle' + '_conf_cl_2'])
            data['dihes_angle' + '_conf_cl_3'] = np.abs(data['dihes_angle' + '_conf_cl_3'])

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

            ab_g_conf_cl_1 = pgl.graph.Graph(num_nodes=N,
                                           edges=data['edges' + '_conf_cl_1'],
                                           node_feat={name: data[name + '_conf_cl_1'].reshape([-1, 1]) for name in
                                                      self.atom_names},
                                           edge_feat={name: data[name + '_conf_cl_1'].reshape([-1, 1]) for name in
                                                      self.bond_names + self.bond_float_names})
            ba_g_conf_cl_1 = pgl.graph.Graph(num_nodes=E,
                                           edges=data['BondAngleGraph_edges' + '_conf_cl_1'],
                                           node_feat={},
                                           edge_feat={name: data[name + '_conf_cl_1'].reshape([-1, 1]) for name in
                                                      self.bond_angle_float_names})
            da_g_conf_cl_1 = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges' + '_conf_cl_1']),
                edges=data['DihesAngleGraph_edges' + '_conf_cl_1'],
                node_feat={},
                edge_feat={name: data[name + '_conf_cl_1'].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

            ab_g_conf_cl_2 = pgl.graph.Graph(num_nodes=N,
                                             edges=data['edges' + '_conf_cl_2'],
                                             node_feat={name: data[name + '_conf_cl_2'].reshape([-1, 1]) for name in
                                                        self.atom_names},
                                             edge_feat={name: data[name + '_conf_cl_2'].reshape([-1, 1]) for name in
                                                        self.bond_names + self.bond_float_names})
            ba_g_conf_cl_2 = pgl.graph.Graph(num_nodes=E,
                                             edges=data['BondAngleGraph_edges' + '_conf_cl_2'],
                                             node_feat={},
                                             edge_feat={name: data[name + '_conf_cl_2'].reshape([-1, 1]) for name in
                                                        self.bond_angle_float_names})
            da_g_conf_cl_2 = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges' + '_conf_cl_2']),
                edges=data['DihesAngleGraph_edges' + '_conf_cl_2'],
                node_feat={},
                edge_feat={name: data[name + '_conf_cl_2'].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

            ab_g_conf_cl_3 = pgl.graph.Graph(num_nodes=N,
                                             edges=data['edges' + '_conf_cl_3'],
                                             node_feat={name: data[name + '_conf_cl_3'].reshape([-1, 1]) for name in
                                                        self.atom_names},
                                             edge_feat={name: data[name + '_conf_cl_3'].reshape([-1, 1]) for name in
                                                        self.bond_names + self.bond_float_names})
            ba_g_conf_cl_3 = pgl.graph.Graph(num_nodes=E,
                                             edges=data['BondAngleGraph_edges' + '_conf_cl_3'],
                                             node_feat={},
                                             edge_feat={name: data[name + '_conf_cl_3'].reshape([-1, 1]) for name in
                                                        self.bond_angle_float_names})
            da_g_conf_cl_3 = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges' + '_conf_cl_3']),
                edges=data['DihesAngleGraph_edges' + '_conf_cl_3'],
                node_feat={},
                edge_feat={name: data[name + '_conf_cl_3'].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

            masked_ab_g, masked_ba_g, masked_da_g, masked_node_i, context_id = mask_context_of_geognn_graph_all_node(
                ab_g, ba_g, da_g, mask_ratio=0.1, subgraph_num=self.Cm_vocab)
            # masked_ab_g, masked_ba_g, masked_da_g, masked_node_i, context_id = mask_context_of_geognn_graph_all_node_limit(
            #     ab_g, ba_g, da_g, mask_ratio=0.1, subgraph_num=self.Cm_vocab)
            # masked_ab_g, masked_ba_g, masked_da_g, masked_node_i, context_id = mask_context_of_geognn_graph_all_node_ram(
            #     ab_g, ba_g, da_g, mask_ratio=0.1, subgraph_num=self.Cm_vocab)
            masked_ab_g_cl, masked_ba_g_cl, masked_da_g_cl, masked_node_i_cl, context_id_cl = mask_context_of_geognn_graph_all_node(
                ab_g_conf_cl_1, ba_g_conf_cl_1, da_g_conf_cl_1, mask_ratio=0.1, subgraph_num=self.Cm_vocab)

            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihes_angle_graph_list.append(da_g)

            atom_bond_graph_conf_cl_1_list.append(ab_g_conf_cl_1)
            bond_angle_graph_conf_cl_1_list.append(ba_g_conf_cl_1)
            dihes_angle_graph_conf_cl_1_list.append(da_g_conf_cl_1)

            atom_bond_graph_conf_cl_2_list.append(ab_g_conf_cl_2)
            bond_angle_graph_conf_cl_2_list.append(ba_g_conf_cl_2)
            dihes_angle_graph_conf_cl_2_list.append(da_g_conf_cl_2)

            atom_bond_graph_conf_cl_3_list.append(ab_g_conf_cl_3)
            bond_angle_graph_conf_cl_3_list.append(ba_g_conf_cl_3)
            dihes_angle_graph_conf_cl_3_list.append(da_g_conf_cl_3)

            atom_bond_graph_mask_cl_list.append(masked_ab_g_cl)
            bond_angle_graph_mask_cl_list.append(masked_ba_g_cl)
            dihes_angle_graph_mask_cl_list.append(masked_da_g_cl)

            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_dihes_angle_graph_list.append(masked_da_g)

            if 'Cm' in self.pretrain_tasks or 'Cm1' in self.pretrain_tasks:

                Cm_node_i.append(masked_node_i + node_count)
                Cm_context_id.append(context_id)

                Cm_node_i_cl.append(masked_node_i_cl + node_count)
                Cm_context_id_cl.append(context_id_cl)

            if 'Fg' in self.pretrain_tasks:

                Fg_morgan.append(data['morgan_fp'])
                Fg_daylight.append(data['daylight_fg_counts'])
                Fg_maccs.append(data['maccs_fp'])

            if 'Bar' in self.pretrain_tasks:

                Ba_node_i.append(data['Ba_node_i'] + node_count)
                Ba_node_j.append(data['Ba_node_j'] + node_count)
                Ba_node_k.append(data['Ba_node_k'] + node_count)
                Ba_bond_angle.append(data['Ba_bond_angle'])

                Ba_node_i_cl.append(data['Ba_node_i' + '_conf_cl_1'] + node_count)
                Ba_node_j_cl.append(data['Ba_node_j' + '_conf_cl_1'] + node_count)
                Ba_node_k_cl.append(data['Ba_node_k' + '_conf_cl_1'] + node_count)
                Ba_bond_angle_cl.append(data['Ba_bond_angle' + '_conf_cl_1'])

            if 'Blr' in self.pretrain_tasks:

                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
                Bl_bond_length.append(data['Bl_bond_length'])

                Bl_node_i_cl.append(data['Bl_node_i' + '_conf_cl_1'] + node_count)
                Bl_node_j_cl.append(data['Bl_node_j' + '_conf_cl_1'] + node_count)
                Bl_bond_length_cl.append(data['Bl_bond_length' + '_conf_cl_1'])

            if 'Adc' in self.pretrain_tasks:

                Ad_node_i.append(data['Ad_node_i'] + node_count)
                Ad_node_j.append(data['Ad_node_j'] + node_count)
                Ad_atom_dist.append(data['Ad_atom_dist'])

                Ad_node_i_cl.append(data['Ad_node_i' + '_conf_cl_1'] + node_count)
                Ad_node_j_cl.append(data['Ad_node_j' + '_conf_cl_1'] + node_count)
                Ad_atom_dist_cl.append(data['Ad_atom_dist' + '_conf_cl_1'])

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

                if len(data['dihes_angle' + '_conf_cl_1']) != 0:
                    Da_node_i_cl.append(data['dihes_angle_node_i' + '_conf_cl_1'] + node_count)
                    Da_node_j_cl.append(data['dihes_angle_node_j' + '_conf_cl_1'] + node_count)
                    Da_node_k_cl.append(data['dihes_angle_node_k' + '_conf_cl_1'] + node_count)
                    Da_node_l_cl.append(data['dihes_angle_node_l' + '_conf_cl_1'] + node_count)
                    Da_bond_angle_temp_cl = data['dihes_angle' + '_conf_cl_1'][
                                         len(data['dihes_angle_node_i_extra' + '_conf_cl_1']):len(
                                             data['dihes_angle_node_i_extra' + '_conf_cl_1']) + len(
                                             data['dihes_angle_node_i' + '_conf_cl_1'])]
                    # Da_bond_angle_temp = Da_bond_angle_temp[0::2]
                    Da_bond_angle_temp_cl = np.abs(Da_bond_angle_temp_cl) * np.pi / 180
                    Da_bond_angle_cl.append(Da_bond_angle_temp_cl)
                
                if len(data['dihes_angle_extra' + '_conf_cl_1']) != 0:
                    Da_node_i_extra_cl.append(data['dihes_angle_node_i_extra' + '_conf_cl_1'] + node_count)
                    Da_node_j_extra_cl.append(data['dihes_angle_node_j_extra' + '_conf_cl_1'] + node_count)
                    Da_node_k_extra_cl.append(data['dihes_angle_node_k_extra' + '_conf_cl_1'] + node_count)
                    Da_node_l_extra_cl.append(data['dihes_angle_node_l_extra' + '_conf_cl_1'] + node_count)
                    Da_bond_angle_temp_extra_cl = data['dihes_angle_extra' + '_conf_cl_1']
                    Da_bond_angle_temp_extra_cl = np.abs(Da_bond_angle_temp_extra_cl) * np.pi / 180
                    Da_bond_angle_extra_cl.append(Da_bond_angle_temp_extra_cl)

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

        atom_bond_graph_conf_cl_1 = pgl.Graph.batch(atom_bond_graph_conf_cl_1_list)
        self._flat_shapes(atom_bond_graph_conf_cl_1.node_feat)
        self._flat_shapes(atom_bond_graph_conf_cl_1.edge_feat)
        graph_dict['atom_bond_graph_conf_cl_1'] = atom_bond_graph_conf_cl_1

        bond_angle_graph_conf_cl_1 = pgl.Graph.batch(bond_angle_graph_conf_cl_1_list)
        self._flat_shapes(bond_angle_graph_conf_cl_1.node_feat)
        self._flat_shapes(bond_angle_graph_conf_cl_1.edge_feat)
        graph_dict['bond_angle_graph_conf_cl_1'] = bond_angle_graph_conf_cl_1

        dihes_angle_graph_conf_cl_1 = pgl.Graph.batch(dihes_angle_graph_conf_cl_1_list)
        self._flat_shapes(dihes_angle_graph_conf_cl_1.node_feat)
        self._flat_shapes(dihes_angle_graph_conf_cl_1.edge_feat)
        graph_dict['dihes_angle_graph_conf_cl_1'] = dihes_angle_graph_conf_cl_1

        atom_bond_graph_conf_cl_2 = pgl.Graph.batch(atom_bond_graph_conf_cl_2_list)
        self._flat_shapes(atom_bond_graph_conf_cl_2.node_feat)
        self._flat_shapes(atom_bond_graph_conf_cl_2.edge_feat)
        graph_dict['atom_bond_graph_conf_cl_2'] = atom_bond_graph_conf_cl_2

        bond_angle_graph_conf_cl_2 = pgl.Graph.batch(bond_angle_graph_conf_cl_2_list)
        self._flat_shapes(bond_angle_graph_conf_cl_2.node_feat)
        self._flat_shapes(bond_angle_graph_conf_cl_2.edge_feat)
        graph_dict['bond_angle_graph_conf_cl_2'] = bond_angle_graph_conf_cl_2

        dihes_angle_graph_conf_cl_2 = pgl.Graph.batch(dihes_angle_graph_conf_cl_2_list)
        self._flat_shapes(dihes_angle_graph_conf_cl_2.node_feat)
        self._flat_shapes(dihes_angle_graph_conf_cl_2.edge_feat)
        graph_dict['dihes_angle_graph_conf_cl_2'] = dihes_angle_graph_conf_cl_2

        atom_bond_graph_conf_cl_3 = pgl.Graph.batch(atom_bond_graph_conf_cl_3_list)
        self._flat_shapes(atom_bond_graph_conf_cl_3.node_feat)
        self._flat_shapes(atom_bond_graph_conf_cl_3.edge_feat)
        graph_dict['atom_bond_graph_conf_cl_3'] = atom_bond_graph_conf_cl_3

        bond_angle_graph_conf_cl_3 = pgl.Graph.batch(bond_angle_graph_conf_cl_3_list)
        self._flat_shapes(bond_angle_graph_conf_cl_3.node_feat)
        self._flat_shapes(bond_angle_graph_conf_cl_3.edge_feat)
        graph_dict['bond_angle_graph_conf_cl_3'] = bond_angle_graph_conf_cl_3

        dihes_angle_graph_conf_cl_3 = pgl.Graph.batch(dihes_angle_graph_conf_cl_3_list)
        self._flat_shapes(dihes_angle_graph_conf_cl_3.node_feat)
        self._flat_shapes(dihes_angle_graph_conf_cl_3.edge_feat)
        graph_dict['dihes_angle_graph_conf_cl_3'] = dihes_angle_graph_conf_cl_3

        atom_bond_graph_mask_cl = pgl.Graph.batch(atom_bond_graph_mask_cl_list)
        self._flat_shapes(atom_bond_graph_mask_cl.node_feat)
        self._flat_shapes(atom_bond_graph_mask_cl.edge_feat)
        graph_dict['masked_atom_bond_graph_conf_cl_1'] = atom_bond_graph_mask_cl
        
        bond_angle_graph_mask_cl = pgl.Graph.batch(bond_angle_graph_mask_cl_list)
        self._flat_shapes(bond_angle_graph_mask_cl.node_feat)
        self._flat_shapes(bond_angle_graph_mask_cl.edge_feat)
        graph_dict['masked_bond_angle_graph_conf_cl_1'] = bond_angle_graph_mask_cl
        
        dihes_angle_graph_mask_cl = pgl.Graph.batch(dihes_angle_graph_mask_cl_list)
        self._flat_shapes(dihes_angle_graph_mask_cl.node_feat)
        self._flat_shapes(dihes_angle_graph_mask_cl.edge_feat)
        graph_dict['masked_dihes_angle_graph_conf_cl_1'] = dihes_angle_graph_mask_cl

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

        feed_dict['rms_12'] = np.array(rms_12_list, 'float32')
        feed_dict['rms_13'] = np.array(rms_13_list, 'float32')
        feed_dict['rms_14'] = np.array(rms_14_list, 'float32')
        feed_dict['rms_23'] = np.array(rms_23_list, 'float32')
        feed_dict['rms_24'] = np.array(rms_24_list, 'float32')
        feed_dict['rms_34'] = np.array(rms_34_list, 'float32')
        feed_dict['en'] = np.array(en_list, 'float32')

        if "FP3D" in self.pretrain_tasks:

            feed_dict['rdf_fp_1'] = np.array(rdf_fp_list_1, 'float32')
            feed_dict['morse_fp_1'] = np.array(morse_fp_list_1, 'float32')
            feed_dict['getaway_fp_1'] = np.array(getaway_fp_list_1, 'float32')


        if 'Cm' in self.pretrain_tasks or 'Cm1' in self.pretrain_tasks:

            feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')

            feed_dict['Cm_node_i' + '_conf_cl_1'] = np.concatenate(Cm_node_i_cl, 0).reshape(-1).astype('int64')
            feed_dict['Cm_context_id' + '_conf_cl_1'] = np.concatenate(Cm_context_id_cl, 0).reshape(-1, 1).astype('int64')

        if 'Fg' in self.pretrain_tasks:
            feed_dict['Fg_morgan'] = np.array(Fg_morgan, 'float32')
            feed_dict['Fg_daylight'] = (np.array(Fg_daylight) > 0).astype('float32')  # >1: 1x
            feed_dict['Fg_maccs'] = np.array(Fg_maccs, 'float32')

        if 'Bar' in self.pretrain_tasks:

            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
            feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')

            feed_dict['Ba_node_i' + '_conf_cl_1'] = np.concatenate(Ba_node_i_cl, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j' + '_conf_cl_1'] = np.concatenate(Ba_node_j_cl, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k' + '_conf_cl_1'] = np.concatenate(Ba_node_k_cl, 0).reshape(-1).astype('int64')
            feed_dict['Ba_bond_angle' + '_conf_cl_1'] = np.concatenate(Ba_bond_angle_cl, 0).reshape(-1, 1).astype('float32')

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

            feed_dict['Da_node_i' + '_conf_cl_1'] = np.concatenate(Da_node_i_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_j' + '_conf_cl_1'] = np.concatenate(Da_node_j_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_k' + '_conf_cl_1'] = np.concatenate(Da_node_k_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_l' + '_conf_cl_1'] = np.concatenate(Da_node_l_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_bond_angle' + '_conf_cl_1'] = np.concatenate(Da_bond_angle_cl, 0).reshape(-1, 1).astype('float32')
           
            feed_dict['Da_node_i_extra' + '_conf_cl_1'] = np.concatenate(Da_node_i_extra_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_j_extra' + '_conf_cl_1'] = np.concatenate(Da_node_j_extra_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_k_extra' + '_conf_cl_1'] = np.concatenate(Da_node_k_extra_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_node_l_extra' + '_conf_cl_1'] = np.concatenate(Da_node_l_extra_cl, 0).reshape(-1).astype('int64')
            feed_dict['Da_bond_angle_extra' + '_conf_cl_1'] = np.concatenate(Da_bond_angle_extra_cl, 0).reshape(-1, 1).astype('float32')

        if 'Blr' in self.pretrain_tasks:

            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')

            feed_dict['Bl_node_i' + '_conf_cl_1'] = np.concatenate(Bl_node_i_cl, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j' + '_conf_cl_1'] = np.concatenate(Bl_node_j_cl, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length' + '_conf_cl_1'] = np.concatenate(Bl_bond_length_cl, 0).reshape(-1, 1).astype('float32')

        if 'Adc' in self.pretrain_tasks:

            feed_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')

            feed_dict['Ad_node_i' + '_conf_cl_1'] = np.concatenate(Ad_node_i_cl, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j' + '_conf_cl_1'] = np.concatenate(Ad_node_j_cl, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist' + '_conf_cl_1'] = np.concatenate(Ad_atom_dist_cl, 0).reshape(-1, 1).astype('float32')

        fp_score = np.zeros((len(mol_list), len(mol_list) - 1))
        fps = [AllChem.GetMorganFingerprint(Chem.AddHs(x), 2, useFeatures=True) for x in mol_list]
        for i in range(len(mol_list)):
            for j in range(i + 1, len(mol_list)):
                fp_sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                fp_score[i, j - 1] = fp_sim
                fp_score[j, i] = fp_sim

        return graph_dict, feed_dict, fp_score


