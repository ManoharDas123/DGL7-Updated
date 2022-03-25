import dgl
import numpy as np
import torch as th
import argparse
import time
import new_partitioing_graph
import networkx as nx
import pandas as pd
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# karate_club_graph = nx.karate_club_graph()


import torch
import pandas as pd

from dgl.data import DGLDataset
import torch
import os

class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')

    def process(self):
        nodes_data = pd.read_csv('./members.csv')
        edges_data = pd.read_csv('./interactions.csv')
        node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = KarateClubDataset()
G = dataset[0]



def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    g.add_edges(dst, src)

    return g

# G = build_karate_club_graph()
# print('We have %d nodes in Karate Club Dataset.' % G.number_of_nodes())
# print('We have %d edges in Karate Club Dataset.' % G.number_of_edges())

#---------------Assign features to nodes-------------------------------- 
# embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
# G.ndata['feat'] = embed.weight
# print(G.ndata['feat'])

# ####---------------Facebook Dataset------------------------------------#####
# edges_path = 'facebook_large/musae_facebook_edges.csv'
# edges = pd.read_csv(edges_path)
# edges.columns = ['source', 'target']

# features_path = 'facebook_large/musae_facebook_features.json'
# with open(features_path) as json_data:
#     features = json.load(json_data)

# max_feature = np.max([v for v_list in features.values() for v in v_list])
# features_matrix = np.zeros(shape=(len(list(features.keys())), max_feature + 1))

# i = 0
# for k, vs in tqdm(features.items()):
#     for v in vs:
#         features_matrix[i, v] = 1
#     i += 1

# node_features = pd.DataFrame(features_matrix, index=features.keys())
# # print("node_features", node_features[5])
# G = nx.from_pandas_edgelist(edges)
# #
# # print(edges.sample(frac=1).head(5))
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())
# # G = dgl.from_networkx(nx_g)
# ####---------------End of Facebook Dataset------------------------------------#####


#-----------------------Cora dataset --------------------------------
# import dgl.data

# dataset = dgl.data.CoraGraphDataset()
# # print('Number of categories:', dataset.num_classes)
# G = dataset[0]
# #-------------------------End Cora dataset------------------------------

#-------------------------Start Citeseer dataset------------------------------
# import dgl.data
# dataset = dgl.data.CiteseerGraphDataset()
# G = dataset[0]
# print('We have %d nodes in Citeseer Dataset.' % G.number_of_nodes())
# print('We have %d edges in Citeseer Dataset.' % G.number_of_edges())
# #-------------------------End Citeseer dataset------------------------------

from load_graph_copy import load_reddit, load_ogb

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
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products')
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
    

    # dgl_g = dgl.from_networkx(G)
    # nx_g = dgl.to_networkx(build_karate_club_graph())

    # print(nx_g)

    # new_partitioing_graph.improved_partition_graph(nx_g, args.num_parts)
    # print(g.in_degrees())

    # new_partitioing_graph.improved_partition_graph(build_karate_club_graph(), args.num_parts, args.sample_length, args.output,
    #                                                 args.reshuffle, balance_ntypes=balance_ntypes,
    #                                                                 balance_edges=args.balance_edges)
    
    new_partitioing_graph.improved_partition_graph(G, args.num_parts, args.sample_length, args.output,
                                                    args.reshuffle, balance_ntypes=balance_ntypes,
                                                                    balance_edges=args.balance_edges)
    # new_partitioing_graph.improved_partition_graph(g, args.num_parts, args.sample_length, args.output,
    #                                                 args.reshuffle, balance_ntypes=balance_ntypes,
    #                                                                 balance_edges=args.balance_edges)
    
    # # new_partitioing_graph.improved_partition_graph(karate_club_graph, args.num_parts)
    # new_partitioing_graph.improved_partition_graph(G, args.num_parts, args.sample_length, args.reshuffle)
    
    # # dgl.distributed.partition_graph(build_karate_club_graph(), args.num_parts, args.output)
    # dgl.distributed.partition_graph(build_karate_club_graph(), args.dataset, args.num_parts, args.output,
    #                                 part_method=args.part_method,
    #                                 balance_ntypes=balance_ntypes,
    #                                 balance_edges=args.balance_edges
    #                                 )

    # dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
    #                                 part_method=args.part_method,
    #                                 balance_ntypes=balance_ntypes,
    #                                 balance_edges=args.balance_edges
    #                                 )
