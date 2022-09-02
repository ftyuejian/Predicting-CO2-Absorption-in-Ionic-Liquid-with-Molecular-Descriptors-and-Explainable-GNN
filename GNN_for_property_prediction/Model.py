import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,MessagePassing
from torch_geometric.nn import global_max_pool,global_mean_pool,global_add_pool
from torch_geometric.utils import add_self_loops, degree, softmax

# global argument
num_atom_type = 119 # including the extra mask tokens
num_Hbrid = 8
num_Aro = 2
num_degree = 7
num_charge = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_isAromatic = 2
num_bond_isInRing = 2

# GCN
class IL_Net_GCN(torch.nn.Module):
    def __init__(self, args):
        super(IL_Net_GCN,self).__init__()
        self.args = args
        n_features = self.args['n_features']
        self.l1 = GCNConv(n_features, 512, normalize = True)
        self.l2 = GCNConv(512, 1024, normalize=True)
        self.l3 = GCNConv(1024, 1024, normalize=True)
        self.l4 = GCNConv(1024, 512, normalize=True)

        self.l5 = nn.Sequential(
            nn.Linear(514, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0)

        return g

    def forward(self, data_i, cond):
        x, edge_index = data_i.x.to(torch.float), data_i.edge_index
        edge_weight = torch.sum(data_i.edge_attr,dim=1).to(torch.float)

        x = self.l1(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l2(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l3(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l4(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.extract(x,data_i.batch)

        x = torch.cat([x, cond], dim=1)
        x = self.l5(x)

        return x

# GAT
class IL_GAT(torch.nn.Module):
    def __init__(self, args):
        super(IL_GAT,self).__init__()
        self.args = args
        n_features = self.args['n_features']

        self.l1 = GATv2Conv(n_features, 512)
        self.l2 = GATv2Conv(512, 1024)
        self.l3 = GATv2Conv(1024, 1024)
        self.l4 = GATv2Conv(1024, 512)

        self.l5 = nn.Sequential(
            nn.Linear(514, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0).to('cuda')

        return g

    def forward(self, data_i, cond):
        x, edge_index = data_i.x.to(torch.float), data_i.edge_index

        x,(edge1,attention1) = self.l1(x, edge_index, return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge2,attention2) = self.l2(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge3,attention3) = self.l3(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge4,attention4) = self.l4(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x = self.extract(x,data_i.batch)

        x = torch.cat([x, cond], dim=1)
        x = self.l5(x)

        return x


# GIN
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_isInRing, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_isAromatic, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:, 0] = num_bond_type - 1 # bond type for self-loop edge
        self_loop_attr[:, 1] = num_bond_isInRing - 1  # bond type for self-loop edge
        self_loop_attr[:, 2] = num_bond_isAromatic - 1  # bond type for self-loop edge

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
                          self.edge_embedding2(edge_attr[:,1]) + \
                          self.edge_embedding3(edge_attr[:,2])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_layer = args['num_gin_layer']
        self.emb_dim = args['emb_dim']
        self.feat_dim = args['feat_dim']
        self.drop_ratio = args['drop_ratio']
        pool = args['pool']

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINEConv(self.emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(self.feat_dim + 2, self.feat_dim),
            nn.Softplus(),
            nn.Linear(self.feat_dim, int(self.feat_dim/2)),
            nn.Softplus(),
            nn.Linear(int(self.feat_dim/2), 1)
        )
    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0)

        return g
    def forward(self, pair_graph, cond):
        # GIN layer
        h = self.x_embedding1(pair_graph.x[:, 0]) + \
            self.x_embedding2(pair_graph.x[:, 1]) + \
            self.x_embedding3(pair_graph.x[:, 2]) + \
            self.x_embedding4(pair_graph.x[:, 3]) + \
            self.x_embedding5(pair_graph.x[:, 4])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, pair_graph.edge_index, pair_graph.edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.feat_lin(h)
        h_pair = self.extract(h, pair_graph.batch)
        h = torch.cat([h_pair, cond], dim=1)

        return self.pred_head(h)