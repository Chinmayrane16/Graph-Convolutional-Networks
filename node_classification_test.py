import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import matplotlib.pyplot as plt

https://docs.dgl.ai/en/0.4.x/tutorials/basics/4_batch.html
https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/1_gcn.html

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

dataset = load_cora_data()
g, node_labels = dataset[0], dataset[2]


plt.figure(1,figsize=(14,12)) 
nx.draw(g.to_networkx(), cmap=plt.get_cmap('Set1'),node_color = node_labels,node_size=75,linewidths=6)
plt.show()










# import networkx as nx
# import torch
# import numpy as np
# import pandas as pd
# from torch_geometric.datasets import Planetoid
# from torch_geometric.utils.convert import to_networkx

# dataset1 = Planetoid(root = '/content/cora',name='Cora')

# cora = dataset1 [0]

# coragraph = to_networkx(cora)

# node_labels = cora.y[list(coragraph.nodes)].numpy()

# import matplotlib.pyplot as plt
# plt.figure(1,figsize=(14,12)) 
# nx.draw(coragraph, cmap=plt.get_cmap('Set1'),node_color = node_labels,node_size=75,linewidths=6)
# plt.show()