import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from einops import rearrange
# graph libraries
import dgl
import networkx as nx

def get_lp(choice='default', input_dim=128, bottleneck_dim=128, use_bias=True):
    if choice == 'default':
        return torch.nn.Linear(input_dim, bottleneck_dim, bias=use_bias)
    elif choice == '2mlp_up':
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim*2, bias=use_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(bottleneck_dim*2, bottleneck_dim))
    elif choice == '2mlp_down':
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim//2, bias=use_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(bottleneck_dim//2, bottleneck_dim))
    elif choice == 'mlp_act':
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim, bias=use_bias),
            torch.nn.ReLU())
    elif choice == 'mlp_bn_act':
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim, bias=use_bias),
            torch.nn.LayerNorm(bottleneck_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(bottleneck_dim, bottleneck_dim, bias=use_bias))
    else:
        return torch.nn.Identity()

@MODELS.register_module()
class ClusterGraph(BaseModule):
    """
    Graph Module to support -
    1. G = (V, E) where V is the set of nodes and E is the set of edges
    2. Graph Convolutional Networks (GCN)
    3. Clustering or Clique Detection
    dgl version = 2.0.0+cu118
    networkx version = 3.1
    """
    def __init__(self,
                input_dim=384,
                output_dim=128,
                hidden_dim=2048,
                feat_drop=0.0,
                attn_drop=0.0,
                temperature=0.1,
                similarity_criteria='cosine',
                similarity_threshold=0.5,
                weighted_adjacency=False,
                bottleneck_dim=128,
                gcn_type='gat',
                num_heads=4,
                pool_heads=False,
                apply_linear_projection=False,
                lp_choice='default',
                num_graph_layers=1,
                use_readout=True,
                use_bias=True,
                activation='elu',
                affine_fn_on_feats=False,
                remove_self_loops=True
                ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.similarity_criteria = similarity_criteria
        self.similarity_threshold = similarity_threshold
        self.weighted_adjacency = weighted_adjacency
        self.gcn_type = gcn_type
        self.num_heads = num_heads
        self.pool_heads = pool_heads
        self.apply_linear_projection = apply_linear_projection
        self.num_graph_layers = num_graph_layers
        self.use_readout = use_readout
        self.use_bias = use_bias
        self.temperature = temperature
        self.affine_fn_on_feats = affine_fn_on_feats
        self.remove_self_loops = remove_self_loops
        # self.lp_choice = lp_choice

        # edge norm
        #self.edge_norm = dgl.nn.EdgeWeightNorm(norm='right', eps=1e-6)

        # global pooling layer
        # self.global_pool = dgl.nn.GlobalAttentionPooling(gate_nn=torch.nn.Linear(self.output_dim, 1))

        if self.affine_fn_on_feats:
            self.affn1 = torch.nn.Linear(input_dim, input_dim, bias=False)
            self.affn2 = torch.nn.Linear(input_dim, input_dim, bias=False)
        else:
            self.affn1 = torch.nn.Identity()
            self.affn2 = torch.nn.Identity()
        
        if self.apply_linear_projection:
            self.linear_projection = get_lp(choice=lp_choice, input_dim=self.input_dim, 
                                            bottleneck_dim=self.bottleneck_dim, use_bias=self.use_bias)
            input_dim = self.bottleneck_dim
        else:
            self.linear_projection = torch.nn.Identity()
            input_dim = self.input_dim

        # define the graph convolutional layers            
        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim if i < num_graph_layers - 1 else output_dim
            # act = self.activation if i < num_graph_layers - 1 else None
            # act = self.activation
            self.gcn_layers.append(dgl.nn.GATv2Conv(in_feats=in_dim, 
                                                    out_feats=out_dim, 
                                                    num_heads=num_heads, 
                                                    feat_drop=feat_drop, 
                                                    attn_drop=attn_drop, 
                                                    residual=True, 
                                                    activation=getattr(F, activation), 
                                                    allow_zero_in_degree=True))

    def create_graph(self, feats: torch.Tensor) -> dgl.DGLGraph:
        feats = F.normalize(feats, dim=-1)
        affinity = -torch.cdist(feats, feats, p=2)
        adj_matrix = torch.exp(affinity/self.temperature) # lower t (0.1) values are more spread out
        adj_matrix = adj_matrix / adj_matrix.sum(dim=2, keepdim=True)

        # apply similarity threshold
        thresholed_matrix = adj_matrix > self.similarity_threshold
        if self.weighted_adjacency:
            adj_matrix *= thresholed_matrix
        else:
            adj_matrix = thresholed_matrix.float()

        # create batched graph
        b, n, d = feats.size()
        batched_graph = []
        adj_matrix = adj_matrix.detach().cpu().numpy()
        for i in range(b):
            B = nx.DiGraph(adj_matrix[i]) # edges with 0 values are not included
            if self.remove_self_loops:
                B.remove_edges_from(nx.selfloop_edges(B))
            g = dgl.from_networkx(B, device=feats.device)
            # Add edge weights from adjacency matrix
            edge_weights = torch.tensor([c for (_, _, c) in B.edges.data('weight')], device=feats.device)
            g.edata['w'] = edge_weights
            batched_graph.append(g)
        return dgl.batch(batched_graph)

    def message_passing(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        feats = self.linear_projection(feats)
        
        b, n, d = feats.size()
        feats = rearrange(feats, 'b n d -> (b n) d')

        out = feats
        for i, gcn in enumerate(self.gcn_layers):
            out = gcn(g, out).flatten(1, 2)

        if self.use_readout:
            # add features to 'h' node data
            g.ndata['h'] = out
            out = dgl.readout_nodes(g, 'h', op='mean') # b x num_heads x out_dim
        else:
            if self.pool_heads:
                out = rearrange(out, '(b n) (h d) -> b n h d', b=b, n=n, h=self.num_heads)
                out = out.mean(dim=2)
            else:
                out = rearrange(out, '(b n) (h d) -> b (n h) d', b=b, n=n, h=self.num_heads)

        return out
    

def get_activation(activation):
    if activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'softmax':
        return torch.nn.Softmax(dim=-1)
    elif activation == 'none':
        return torch.nn.Identity()
    elif activation == 'gelu':
        return torch.nn.GELU(approximate='none')
    elif activation == 'silu':
        return torch.nn.SiLU()
    elif activation == 'swish':
        return torch.nn.SiLU()
    elif activation == 'mish':
        return torch.nn.Mish()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'celu':
        return torch.nn.CELU()
    else:
        return getattr(F, activation)

# new version of cluster graph
# NOTE: 1) used after the clusters are created. Is it useful?
@MODELS.register_module()
class ClusterMessagePassingGraph(BaseModule):
    def __init__(self,
                input_dim=384,
                output_dim=128,
                hidden_dim=2048,
                feat_drop=0.0,
                attn_drop=0.0,
                temperature=0.1,
                similarity_threshold=0.5,
                weighted_adjacency=False,
                bottleneck_dim=128,
                num_heads=4,
                pool_heads=False,
                apply_linear_projection=False,
                norm_input_feats='default',
                lp_choice='default',
                num_graph_layers=1,
                use_bias=True,
                activation='elu',
                ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.similarity_threshold = similarity_threshold
        self.weighted_adjacency = weighted_adjacency
        self.num_heads = num_heads
        self.pool_heads = pool_heads
        self.apply_linear_projection = apply_linear_projection
        self.norm_input_feats = norm_input_feats
        self.num_graph_layers = num_graph_layers
        self.use_bias = use_bias
        self.temperature = temperature
        self.activation = activation
        
        if self.apply_linear_projection:
            self.linear_projection = get_lp(choice=lp_choice, input_dim=self.input_dim, 
                                            bottleneck_dim=self.bottleneck_dim, use_bias=self.use_bias)
            input_dim = self.bottleneck_dim
        else:
            self.linear_projection = torch.nn.Identity()
            input_dim = self.input_dim

        # define the graph convolutional layers            
        self.gcn_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            in_dim = input_dim if i == 0 else hidden_dim * num_heads
            out_dim = hidden_dim if i < num_graph_layers - 1 else output_dim
            # act = self.activation if i < num_graph_layers - 1 else None
            self.gcn_layers.append(dgl.nn.GATv2Conv(in_feats=in_dim, 
                                                    out_feats=out_dim, 
                                                    num_heads=num_heads, 
                                                    feat_drop=feat_drop, 
                                                    attn_drop=attn_drop, 
                                                    residual=True, 
                                                    activation=None, 
                                                    allow_zero_in_degree=True))

    def create_graph(self, feats: torch.Tensor) -> dgl.DGLGraph:
        feats = F.normalize(feats, dim=-1)
        affinity = -torch.cdist(feats, feats, p=2)
        adj_matrix = torch.exp(affinity/self.temperature) # lower t (0.1) values are more spread out
        adj_matrix = adj_matrix / adj_matrix.sum(dim=2, keepdim=True)
    
        if not self.weighted_adjacency:
            thresh_matrix = adj_matrix > self.similarity_threshold
            adj_matrix = thresh_matrix.float()

        batched_graph = []
        adj_matrix = adj_matrix.detach().cpu().numpy()
        for i in range(feats.size(0)):
            B = nx.DiGraph(adj_matrix[i]) # edges with 0 values are not included
            B.remove_edges_from(nx.selfloop_edges(B))
            g = dgl.from_networkx(B, device=feats.device)
            batched_graph.append(g)
        return dgl.batch(batched_graph)
    
    def normalize_input_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if self.norm_input_feats == 'ln':
            return F.layer_norm(feats, feats.size()[1:])
        elif self.norm_input_feats == 'default':
            return F.normalize(feats, dim=-1)
        else:
            return feats

    def message_passing(self, g: dgl.DGLGraph, feats: torch.Tensor) -> torch.Tensor:
        feats = self.normalize_input_feats(feats)

        out = self.linear_projection(feats)
        b, n, d = out.size()
        out = rearrange(out, 'b n d -> (b n) d')

        for gcn in self.gcn_layers:
            out = gcn(g, out).flatten(1, 2)

        out = get_activation(self.activation)(out)

        if self.pool_heads:
            out = rearrange(out, '(b n) (h d) -> b n h d', b=b, n=n, h=self.num_heads)
            out = out.mean(dim=2)
        else:
            out = rearrange(out, '(b n) (h d) -> b (n h) d', b=b, n=n, h=self.num_heads)

        return out