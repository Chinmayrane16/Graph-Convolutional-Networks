import torch
import torch.nn as nn


def compute_degree(A):
    return A.sum(1)


def normalize(A, add_self_loops=False):
    if add_self_loops:
        A = A + torch.eye(A.shape[0])
    D = compute_degree(A)
    return (D ** -0.5).T * A * (D ** -0.5)


class GCL(nn.Module):
    def __init__(self, p_h_dims, h_dims, ifrelu=True):
        super(GCL, self).__init__()
        self.weight = nn.Parameter(
            data=torch.randn((p_h_dims, h_dims)), requires_grad=True
        )
        self.relu = nn.ReLU()
        self.ifrelu = ifrelu

    def forward(self, x):
        A, X = x
        out = torch.matmul(torch.matmul(A, X), self.weight)
        if self.ifrelu:
            return (A, self.relu(out))
        return (A, out)


class GraphCN(nn.Module):
    def __init__(self, num_layers, hid_dims, feat_dims):
        super(GraphCN, self).__init__()
        if isinstance(hid_dims, int):
            hid_dims = [hid_dims] * num_layers
        assert len(hid_dims) == num_layers, "Invalid hid_dim vector"
        hid_dims = [feat_dims] + hid_dims
        self.model = nn.ModuleList(
            [GCL(hid_dims[i], hid_dims[i + 1]) for i in range(len(hid_dims) - 2)]
        )
        self.model.append(GCL(hid_dims[-2], hid_dims[-1], ifrelu=False))

    def forward(self, x):
        A, X = x
        A = torch.squeeze(A, dim=0)
        X = torch.squeeze(X, dim=0)
        A = normalize(A, add_self_loops=True)
        for i in range(len(self.model)):
            A, X = self.model[i]((A, X))
        return X
