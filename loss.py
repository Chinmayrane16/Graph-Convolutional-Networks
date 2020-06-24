import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def neg_k_sampling(neg_samples_space, k):
    idxs = torch.randperm(len(neg_samples_space))[:k]
    return neg_samples_space[idxs]


class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()

    def forward(self, X, A):
        A = torch.squeeze(A, dim=0)
        X = torch.squeeze(X, dim=0)
        # print(X.shape, 'X')
        # print(A.shape, 'A')
        loss = 0
        for v in range(A.shape[0]):

            v_emb = X[v]
            emb_neigh_space = X[A[v] == 1.0]
            non_neigh_space = X[A[v] == 0.0]
            non_neigh_emb = neg_k_sampling(
                non_neigh_space, k=min(20, len(non_neigh_space))
            )

            loss += (
                -F.logsigmoid(torch.matmul(v_emb, emb_neigh_space.T)).mean()
                - F.logsigmoid(-torch.matmul(v_emb, non_neigh_emb.T)).mean()
            )

        return loss
