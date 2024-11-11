#!/usr/bin/python                                                                                                                                                                                             
# -*-coding:utf-8-*-
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
downstream featurizer
"""

import numpy as np
import networkx as nx
from copy import deepcopy
import pgl
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from sklearn.metrics import pairwise_distances
import hashlib
from utils.compound_tools import mol_to_geognn_graph_data_MMFF3d_finetune

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import random
import warnings
import copy
warnings.filterwarnings("ignore", message='not removing hydrogen atom with dummy atom neighbors')
warnings.filterwarnings("ignore", message='non-ring atom 0 marked aromatic')
warnings.filterwarnings("ignore", message="Can't kekulize mol.  Unkekulized atoms: 0 3 4")


class DownstreamTransformFn(object):
    """Gen features for downstream model"""

    def __init__(self, is_inference=False):
        self.is_inference = is_inference

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
        print(smiles)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_geognn_graph_data_MMFF3d_finetune(mol)
        if data is None:
            return None
        if not self.is_inference:
            data['label'] = raw_data['label'].reshape([-1])
        data['smiles'] = smiles
        return data


class DownstreamCollateFn(object):
    """CollateFn for downstream model"""

    def __init__(self,
                 atom_names,
                 bond_names,
                 bond_float_names,
                 bond_angle_float_names,
                 task_type,
                 is_inference=False):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.face_angle_float_names = ["dihes_angle"]
        self.task_type = task_type
        self.is_inference = is_inference

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, data_list):
        """
        Collate features about a sublist of graph data and return join_graph,
        masked_node_indice and masked_node_labels.
        Args:
            data_list : the graph data in gen_features.for data in data_list,
            create node features and edge features according to pgl graph,and then
            use graph wrapper to feed join graph, then the label can be arrayed to batch label.
        Returns:
            The batch data contains finetune label and valid,which are
            collected from batch_label and batch_valid.
        """
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihes_angle_graph_list = []
        label_list = []
        for data in data_list:
            data['dihes_angle'] = np.nan_to_num(data['dihes_angle'])
            data['dihes_angle_extra'] = np.nan_to_num(data['dihes_angle_extra'])
            
            data['DihesAngleGraph_edges'] = np.concatenate(
                [data['DihesAngleGraph_edges_extra'], data['DihesAngleGraph_edges']],
                axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle_extra'], data['dihes_angle']], axis=0)

            len1 = len(data['BondAngleGraph_edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['DihesAngleGraph_edges'] = np.concatenate([data['DihesAngleGraph_edges'], a], axis=0)
            data['dihes_angle'] = np.concatenate([data['dihes_angle'], b], axis=0)

            len1 = len(data['edges'])
            a = np.arange(len1)
            a = np.broadcast_to(a, (2, len1))
            a = a.transpose([1, 0])
            b = np.zeros([len1]).astype('float32')
            data['BondAngleGraph_edges'] = np.concatenate([data['BondAngleGraph_edges'], a], axis=0)
            data['bond_angle'] = np.concatenate([data['bond_angle'], b], axis=0)

            data['dihes_angle'] = np.abs(data['dihes_angle'])

            ab_g = pgl.Graph(
                num_nodes=len(data[self.atom_names[0]]),
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat=dict({name: data[name].reshape([-1, 1]) for name in self.bond_names},
                               **{name: data[name].reshape([-1, 1]) for name in
                                  self.bond_float_names}))
            ba_g = pgl.Graph(
                num_nodes=len(data['edges']),
                edges=data['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: data[name].reshape([-1, 1]).astype('float32') for name in self.bond_angle_float_names})

            da_g = pgl.Graph(
                num_nodes=len(data['BondAngleGraph_edges']),
                edges=data['DihesAngleGraph_edges'],
                node_feat={},
                edge_feat={name: data[name].reshape([-1, 1]).astype('float32') *
                                 np.pi / 180 for name in self.face_angle_float_names})

            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihes_angle_graph_list.append(da_g)
            if not self.is_inference:
                label_list.append(data['label'])

        # TODO: append default mol
        smiles = "CCC(C)C"
        mol_default = Chem.MolFromSmiles(smiles)
        mol_default_3d = mol_to_geognn_graph_data_MMFF3d_finetune(mol_default)
        
        ab_g_default = pgl.Graph(
            num_nodes=len(mol_default_3d[self.atom_names[0]]),
            edges=mol_default_3d['edges'],
            node_feat={name: mol_default_3d[name].reshape([-1, 1]) for name in self.atom_names},
            edge_feat=dict({name: mol_default_3d[name].reshape([-1, 1]) for name in self.bond_names},
                           **{name: mol_default_3d[name].reshape([-1, 1]) for name in
                              self.bond_float_names}))
        ba_g_default = pgl.Graph(
            num_nodes=len(mol_default_3d['edges']),
            edges=mol_default_3d['BondAngleGraph_edges'],
            node_feat={},
            edge_feat={name: mol_default_3d[name].reshape([-1, 1]).astype('float32') for name in
                       self.bond_angle_float_names})

        da_g_default = pgl.Graph(
            num_nodes=len(mol_default_3d['BondAngleGraph_edges']),
            edges=mol_default_3d['DihesAngleGraph_edges'],
            node_feat={},
            edge_feat={
                name: mol_default_3d[name].reshape([-1, 1]).astype(
                    'float32') * np.pi / 180 for name in self.face_angle_float_names})

        atom_bond_graph_list.append(ab_g_default)
        bond_angle_graph_list.append(ba_g_default)
        dihes_angle_graph_list.append(da_g_default)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        dihes_angle_graph = pgl.Graph.batch(dihes_angle_graph_list)
        # TODO: reshape due to pgl limitations on the shape
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        self._flat_shapes(dihes_angle_graph.node_feat)
        self._flat_shapes(dihes_angle_graph.edge_feat)

        if not self.is_inference:
            if self.task_type == 'class':
                labels = np.array(label_list)
                # label: -1 -> 0, 1 -> 1
                labels = ((labels + 1.0) / 2)
                valids = (labels != 0.5)
                return [atom_bond_graph, bond_angle_graph, dihes_angle_graph, valids, labels]
            else:
                labels = np.array(label_list, 'float32')
                return atom_bond_graph, bond_angle_graph, dihes_angle_graph, labels
        else:
            return atom_bond_graph, bond_angle_graph, dihes_angle_graph

