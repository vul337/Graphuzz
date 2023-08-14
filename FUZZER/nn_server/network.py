import dgl
import torch
import torch.nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from layer import ConvPoolBlock, SAGPool

class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=5,
        pool_ratio: float = 0.8,
        dropout: float = 0.0,
        num_lins=6
    ):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_convs

        convpools = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )

        self.convpools = torch.nn.ModuleList(convpools)

        self.num_lins = num_lins
        self.lins = []
        self.MLP_IN_DIM = hid_dim * 2 * num_convs
        self.lins.append(torch.nn.Linear(hid_dim * 2 * num_convs, hid_dim * 2 * num_convs + 1))

        INPUT_FEATURE = hid_dim * 2 * num_convs + 1
        for i in range(0, self.num_lins - 2):
            self.lins.append(torch.nn.Linear(INPUT_FEATURE, int(INPUT_FEATURE * 2 // 3)))
            INPUT_FEATURE = int(INPUT_FEATURE * 2 // 3)

        self.lins.append(torch.nn.Linear(INPUT_FEATURE, 1))

        self.lins = torch.nn.ModuleList(self.lins)

    def forward(self, graph: dgl.DGLGraph):
        #print("=======================")
        feat = graph.ndata["attr"]
        #feat = torch.tensor(feat, dtype=torch.float32)
        final_readout = None
        
        perms = torch.tensor([])
        for i in range(self.num_convpools):
            graph, feat, readout, perm = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = torch.cat([final_readout, readout], -1)
            perms = torch.cat([perms, perm.cpu()])
        hardtanh = torch.nn.Hardtanh(-10, 10)
        final_readout = hardtanh(final_readout)
        #print(final_readout)
        for i in range(0, self.num_lins):
            #print("======")
            #print(final_readout)
            final_readout = self.lins[i](final_readout)
            #print(final_readout)
            final_readout = hardtanh(final_readout)
            #print(final_readout)
            if i < self.num_lins - 1:
                final_readout = F.dropout(final_readout, p=self.dropout, training=self.training)
            #final_readout = torch.nan_to_num(final_readout, nan=0.0, posinf=6.0)
        
        #print(final_readout)
        #softplus = torch.nn.Softplus()
        #print(final_readout)
        final_readout = hardtanh(final_readout) * 100.0

        return final_readout.squeeze(-1), perms


class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_convs=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)

    def forward(self, graph: dgl.DGLGraph):
        feat = graph.ndata["attr"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)

        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat(
            [self.avg_readout(graph, feat), self.max_readout(graph, feat)],
            dim=-1,
        )

        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        feat = F.log_softmax(self.lin3(feat), dim=-1)

        return feat


def get_sag_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError(
            "SAGNetwork type {} is not supported.".format(net_type)
        )
