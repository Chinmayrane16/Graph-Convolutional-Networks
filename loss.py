import torch
import torch.nn.functional as F
import numpy as np

def neg_k_sampling(neg_samples_space, k):
    sample_points = np.random.choice(neg_samples_space, k, replace=False)
    return sample_points


def unsupervised_loss(A, X, G):
    loss = 0
    for v in range(A.shape[0]-1):

        v_neighbors = [n for n in G.neighbors(v)]
        
        emb_neighbors = torch.stack([X[v] for v in v_neighbors])

        v_emb = X[v]
        v_neighbors += [v]
        neg_samples_space = [i for i in range(A.shape[0]) if i not in v_neighbors]
        neg_samples = neg_k_sampling(neg_samples_space,k=5)

        neg_samples_emb = torch.stack([X[v] for v in neg_samples])

        loss += -F.logsigmoid(torch.matmul(v_emb, emb_neighbors.T)).mean() - \
                    F.logsigmoid(-torch.matmul(v_emb, neg_samples_emb.T)).mean()

    return loss


