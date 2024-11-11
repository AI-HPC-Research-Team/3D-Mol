#!/usr/bin/python                                                                                  
# -*-coding:utf-8-*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
GEM pretrain
"""

import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging
import paddle

from datasets.inmemory_dataset import InMemoryDataset
from utils.basic_utils import load_json_config
from featurizers.gem_featurizer import GeoPredTransformFn, GeoPredCollateFn
from model_zoo.gem_model import GeoPredModel, GeoGNNModel
# from src.utils import exempt_parameters

import copy


def train(args, model, optimizer, data_gen, dwa=None):
    """tbd"""
    model.train()
    step = 0
    list_loss = []
    dict_loss = {}
    for graph_dict, feed_dict in data_gen:
        print('rank:%s step:%s' % (0, step))

        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        train_loss, sub_losses, coef = model(graph_dict, feed_dict, return_subloss=True, dwa=dwa)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(train_loss.numpy().mean())
        step += 1
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return np.mean(list_loss), dict_loss


@paddle.no_grad()
def evaluate(args, model, data_gen, dict_loss=None):
    """tbd"""
    model.eval()

    dict_loss = {'loss': []}
    coefs = None
    step = 0
    for graph_dict, feed_dict in data_gen:
        print('rank:%s step:%s' % (0, step))
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in feed_dict:
            feed_dict[k] = paddle.to_tensor(feed_dict[k])
        loss, sub_losses, coef = model(graph_dict, feed_dict, return_subloss=True)
        coefs = coef
        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []
            v_np = sub_losses[name].numpy()
            dict_loss[name].append(v_np)
        dict_loss['loss'] = loss.numpy()
        step += 1
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return dict_loss, coefs


def load_smiles_to_dataset(data_path):
    """tbd"""
    files = [data_path]
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            tmp_data_list = [line.strip() for line in f.readlines()]
        data_list.extend(tmp_data_list)
    dataset = InMemoryDataset(data_list=data_list)
    return dataset


def main(args):
    # time.sleep(3300)
    s = time.time()
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate
    print("load config Time used:%ss" % (time.time() - s))
    ### load data
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = load_smiles_to_dataset(args.data_path)
        if args.DEBUG:
            dataset = dataset[100:180]
        # dataset = dataset[dist.get_rank()::dist.get_world_size()]
        smiles_lens = [len(smiles) for smiles in dataset]
        print('Total size:%s' % (len(dataset)))
        print('Dataset smiles min/max/avg length: %s/%s/%s' % (
            np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))
        transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
        # this step will be time consuming due to rdkit 3d calculation
        dataset.transform(transform_fn, num_workers=args.num_workers)
        dataset._none_remove()
        dataset.save_data(args.cached_data_path)
        return
    else:
        if args.cached_data_path is None or args.cached_data_path == "":
            print('Processing data...')
            dataset = load_smiles_to_dataset(args.data_path)
            if args.DEBUG:
                dataset = dataset[100:180]
            # dataset = dataset[dist.get_rank()::dist.get_world_size()]
            smiles_lens = [len(smiles) for smiles in dataset]
            print('Total size:%s' % (len(dataset)))
            print('Dataset smiles min/max/avg length: %s/%s/%s' % (
                np.min(smiles_lens), np.max(smiles_lens), np.mean(smiles_lens)))
            transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
            # this step will be time consuming due to rdkit 3d calculation
            dataset.transform(transform_fn, num_workers=args.num_workers)
        else:
            print('Read preprocessing data...')
            dataset = InMemoryDataset(npz_data_path=args.cached_data_path)
            dataset._smiles_remove()
#            dataset._energy_remove()
            if args.DEBUG:
                dataset = dataset[0:1000]
    print("load data Time used:%ss" % (time.time() - s))

    """tbd"""
    compound_encoder = GeoGNNModel(compound_encoder_config)
    model = GeoPredModel(model_config, compound_encoder)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    print('Total param num: %s' % (len(model.parameters())))
    for i, param in enumerate(model.named_parameters()):
        print(i, param[0], param[1].name)

    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)
    print("init model Time used:%ss" % (time.time() - s))

    test_index = int(len(dataset) * (1 - args.test_ratio))
    train_dataset = dataset[:test_index]
    test_dataset = dataset[test_index:]
    del dataset
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    collate_fn = GeoPredCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        pretrain_tasks=model_config['pretrain_tasks'],
        mask_ratio=model_config['mask_ratio'],
        Cm_vocab=model_config['Cm_vocab'])
    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_fn)
    del train_dataset
    test_data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_fn)
    del test_dataset
    print("process data Time used:%ss" % (time.time() - s))
    list_test_loss = []
    list_train_loss = []
    min_test = 1000000
    for epoch_id in range(args.max_epoch):
        s = time.time()
        train_loss, _ = train(args, model, opt, train_data_gen)
        test_loss, coef = evaluate(args, model, test_data_gen)
        paddle.save(compound_encoder.state_dict(), '%s/epoch%d.pdparams' % (args.model_dir, epoch_id))
        if min_test > test_loss['loss']:
            min_test = test_loss['loss']
            paddle.save(compound_encoder.state_dict(),
                        '%s/regr.pdparams' % (args.model_dir + "/pretrain_models"))
            paddle.save(compound_encoder.state_dict(),
                        '%s/class.pdparams' % (args.model_dir + "/pretrain_models"))
        list_test_loss.append(test_loss['loss'])
        list_train_loss.append(train_loss)
        print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
        print("Time used:%ss" % (time.time() - s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    gen = paddle.seed(4321)
    np.random.seed(4321)
    parser.add_argument("--task", choices=['train', 'data'], default='train')
    parser.add_argument("--DEBUG", action='store_true', default=False)
    parser.add_argument("--distributed", action='store_true', default=False)
    parser.add_argument("--cached_data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='zinc')
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
    
