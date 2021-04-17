import torch
from torch_geometric.nn import MessagePassing

class NetFiltersLayer(MessagePassing):

    def __init__(self, in_channels, out_channels, proxy_channels=2):
        
        super(NetFiltersLayer, self).__init__(aggr='mean')
        
        # TODO: implement NetFilters

    def forward(self, batch):

        # TODO: implement NetFilters

    # TODO: message, update ...
