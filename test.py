import matplotlib.pyplot as plt
import networkx as nx
import torch
from model import GraphCN
from loss import unsupervised_loss

G = nx.karate_club_graph()
# nx.draw(G)
# plt.show()
A = nx.adjacency_matrix(G)
# print(([n for n in G.neighbors(0)]))
A = torch.tensor(A.todense())
feats = torch.eye(A.shape[0])
# print(A.todense())
# print(A.shape[0])
print(unsupervised_loss(A, feats))


# model = GraphCN(5, [256, 128, 64, 32, 2], feats.shape[0])
# out = model(A, feats).detach().numpy()
# plt.scatter(out[:, 0], out[:, 1])
# plt.savefig('emb.png')
