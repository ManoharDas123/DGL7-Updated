import new_partitioing_graph
# from ..tools.launch import main
import train_dist
import dgl
from dgl.data import DGLDataset
import pandas as pd
import torch
import argparse
import time
from collections import defaultdict
import pymetis
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import json


import dgl.data
#from ogb.nodeproppred import DglNodePropPredDataset
#dataset = DglNodePropPredDataset('ogbn-arxiv')
#dataset = dgl.data.CoraGraphDataset()
# dataset = dgl.data.KarateClubDataset()

#G, node_labels = dataset[0]
#G = dgl.add_reverse_edges(G)
#G.ndata['label'] = node_labels[:, 0]
#node_features = G.ndata['feat']
#nlabels = G.ndata['label']
#num_classes = dataset.num_classes
# G.is_block = False

dataset = dgl.data.CoraGraphDataset()
G = dataset[0]
nlabels = G.ndata['label']
num_classes = dataset.num_classes
# G.is_block = False

from load_graph_copy import load_reddit, load_ogb

def weight(node, sampling_length, degrees):
    probability_weight = 1 - pow((1 - 1 / degrees[node]), sampling_length)
    return int(probability_weight * 100)

def xadj_adjncy(g, sample_length):
    eweights = []
    adjncy = []
    xadj = [0]
    degrees = (g.in_degrees()).tolist()
    size = 1

    for node in g.adj():
        adjacency1 = (node._indices()).tolist()
        adjacency = [item for sublist in adjacency1 for item in sublist]
        adjncy += adjacency
        for neighbor in adjacency:
            eweights.append(weight(neighbor, sample_length, degrees))
        xadj.append(xadj[size - 1] + len(adjacency))
        size += 1
    return [xadj, adjncy, eweights]

def bias_metis_partition(num_parts, g, sample_length, xadj, adjncy, eweights):
    partition_sets = []
    n_cuts, membership = pymetis.part_graph(num_parts, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None, eweights=eweights, recursive=False)
    
    for i in range(num_parts):
        partition_data = np.argwhere(np.array(membership) == i).ravel()
        partition_sets.append(partition_data.tolist())
        node_part_var = []
        for j in range(g.number_of_nodes()):
          g_l = [*range(0,g.number_of_nodes(), 1)]
          if g_l[j] in partition_sets[0]:
              node_part_var.append(0)
          else:
              node_part_var.append(1)
    return [partition_data, partition_sets, node_part_var]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    argparser.add_argument('--sample_length', type=int, default='5',
                           help='length of sample node.')
    argparser.add_argument('--reshuffle', type=bool,
                           help='reshuffle is allowed or not')
    argparser.add_argument('--graph_name', type=str, help='graph name')
    argparser.add_argument('--id', type=int, help='the partition id')
    argparser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    argparser.add_argument('--part_config', type=str, help='The path to the partition config file')
    argparser.add_argument('--num_clients', type=int, help='The number of clients')
    argparser.add_argument('--n_classes', type=int, help='the number of classes')
    argparser.add_argument('--backend', type=str, default='gloo', help='pytorch distributed backend')
    argparser.add_argument('--num_gpus', type=int, default=-1,
                        help="the number of GPU device. Use -1 for CPU training")
    argparser.add_argument('--num_epochs', type=int, default=20)
    argparser.add_argument('--num_hidden', type=int, default=16)
    argparser.add_argument('--num_layers', type=int, default=2)
    argparser.add_argument('--fan_out', type=str, default='10,25')
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--batch_size_eval', type=int, default=100000)
    argparser.add_argument('--log_every', type=int, default=20)
    argparser.add_argument('--eval_every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--local_rank', type=int, help='get rank of the process')
    argparser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    argparser.add_argument('--pad-data', default=False, action='store_true',
                        help='Pad train nid to the same length across machine, to ensure num of batches to be the same.')
    argparser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    argparser.add_argument("--ssh_username", default="", help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
                "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
                "instead of 'ssh 1.2.3.4 CMD'")
    argparser.add_argument('--workspace', type=str,
                           help='Path of user directory of distributed tasks. \
                           This is used to specify a destination location where \
                           the contents of current directory will be rsyncd')
    argparser.add_argument('--num_trainers', type=int,
                           help='The number of trainer processes per machine')
    argparser.add_argument('--num_omp_threads', type=int,
                           help='The number of OMP threads per trainer')
    argparser.add_argument('--num_samplers', type=int, default=0,
                           help='The number of sampler processes per trainer process')
    argparser.add_argument('--num_servers', type=int,
                           help='The number of server processes per machine')
    argparser.add_argument('--num_server_threads', type=int, default=1, help='The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on the same machine. By default, it is 1.')
    argparser.add_argument('--graph_format', type=str, default='csc',
                        help='The format of the graph structure of each partition. \
                        The allowed formats are csr, csc and coo. A user can specify multiple \
                        formats, separated by ",". For example, the graph format is "csr,csc".')
    argparser.add_argument('--extra_envs', nargs='+', type=str, default=[],
                        help='Extra environment parameters need to be set. For example, \
                        you can set the LD_LIBRARY_PATH and NCCL_DEBUG by adding: \
                        --extra_envs LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH NCCL_DEBUG=INFO ')
    argparser.add_argument('--graph_metadata_path', type=str, default='graph_metadata')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogb-arxiv':
        g, _ = load_ogb('ogbn-arxiv')
    elif args.dataset == 'ogb-paper100M':
        g, _ = load_ogb('ogbn-papers100M')
    # print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    # print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    # print('train: {}, valid: {}, test: {}'.format(th.sum(g.ndata['train_mask']),
    #                                               th.sum(g.ndata['val_mask']),
    #                                               th.sum(g.ndata['test_mask'])))
    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
    # xadj, adjncy, eweights = xadj_adjncy(g, args.sample_length)
    # partition_data, partition_sets, node_part_var = bias_metis_partition(args.num_parts, g, args.sample_length, xadj, adjncy, eweights)
    # new_partitioing_graph.improved_partition_graph(g, args.num_parts, args.sample_length, args.output, args.reshuffle, partition_data, xadj, adjncy, eweights, node_part_var, graph_name='test', balance_ntypes=None, balance_edges=False, num_hops=1, return_mapping=False)

xadj, adjncy, eweights = xadj_adjncy(G, args.sample_length)
partition_data, partition_sets, node_part_var = bias_metis_partition(args.num_parts, G, args.sample_length, xadj, adjncy, eweights)
new_partitioing_graph.improved_partition_graph(G, args.num_parts, args.sample_length, args.output, args.reshuffle, partition_data, xadj, adjncy, eweights, node_part_var, graph_name='test', balance_ntypes=None, balance_edges=False, num_hops=1, return_mapping=False)

graph_metadeta = { 
    'xadj': xadj,  
    'adjncy': adjncy, 
    'partition_data': partition_data.tolist(), 
    'partition_sets': partition_sets,
    'node_part_var': node_part_var
}
with open('{}/{}.json'.format(args.output, args.graph_metadata_path), 'w') as outfile:
    json.dump(graph_metadeta, outfile, sort_keys=True, indent=4)