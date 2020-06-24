from torched.trainer_utils import Train
from model import GraphCN
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import torch
from loss import UnsupervisedLoss
import matplotlib.pyplot as plt


class GDS(Dataset):
    def __init__(self):
        G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(G).todense()
        self.emb = torch.eye(self.A.shape[0])

    def __getitem__(self, i):
        return (torch.Tensor(self.A).float(), self.emb), torch.Tensor(self.A).float()

    def __len__(self):
        return 1


dl = DataLoader(GDS(), batch_size=1)
model = GraphCN(1, [2], 34)
trainer = Train(model, [dl, dl], cuda=False)
trainer.train(1e-2, 1000, 1, crit=UnsupervisedLoss())

trained_model = trainer.model
with torch.no_grad():
    out = trained_model(GDS()[0][0])
# print(out.shape)
# plt.scatter(out[:, 0].numpy(), out[:, 1].numpy())
nx.draw()
plt.show()
plt.savefig("emb.png")
