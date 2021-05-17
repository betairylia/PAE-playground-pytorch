import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class NetFiltersLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, proxy_channels=2, position_dims=3, act=nn.ReLU()):
        
        super(NetFiltersLayer, self).__init__(aggr='mean')

        self.spatial_mlp = nn.Sequential(
            nn.Linear(position_dims, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, proxy_channels * out_channels)
        )

        self.proxy_mlp = nn.Linear(in_channels, proxy_channels * out_channels)

        self.cin = in_channels
        self.cout = out_channels
        self.cproxy = proxy_channels

        self.act = act
        
    def forward(self, batch):

        proxy_features = self.proxy_mlp(batch.x)

        batch.x = self.propagate(batch.edge_index, pos = batch.pos, proxy_features = proxy_features)

        batch.x = self.act(batch.x)

        return batch
    
    def message(self, pos_i, pos_j, proxy_features_j):
        
        relative_pos = pos_j - pos_i
        conv_filters = self.spatial_mlp(relative_pos)
        preagg_proxy = conv_filters * proxy_features_j

        preagg_proxy = preagg_proxy.view(-1, self.cproxy, self.cout) # => [Nedge, Cproxy, Cout]
        output = preagg_proxy.sum(1) # => [Nedge, Cout]

        return output
