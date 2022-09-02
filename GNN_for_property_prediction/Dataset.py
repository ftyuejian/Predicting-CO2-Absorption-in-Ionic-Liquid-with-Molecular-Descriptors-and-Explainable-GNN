import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader

args = {
    'add_global':True,
    'bi_direction':True
}


def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return combined_Graph

def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
    node = torch.tensor([0,0,0,0,0]).reshape(1, -1)
    # node.shape
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []
    for i in range(num_node):
        # print(i)
        start.append(i)
        end.append(new_node)
        attr.append([0, 0, 0])
    if args['bi_direction'] == True:
        for i in range(num_node):
            # print(i)
            start.append(new_node)
            end.append(i)
            attr.append([0, 0, 0])

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    g = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return g


class IL_set(torch.utils.data.Dataset):
    """
    torch dataset
    """
    def __init__(self,path):
        super(IL_set, self).__init__()
        data_path = path + 'data.npy'
        label_path = path + 'label.npy'

        self.data = np.load(data_path,allow_pickle=True)
        self.label = np.load(label_path,allow_pickle=True)

        self.length = self.label.shape[0]

        # show basic information
        print("----info----")
        print("data_length",self.length)
        print("------------")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cation = self.data[idx][0]
        anion = self.data[idx][1]
        T = self.data[idx][2]
        P = self.data[idx][3]
        # # debug
        # print("cation",cation)
        # print("anion",anion)

        cation = self.mol2graph(cation)
        anion = self.mol2graph(anion)
        Combine_Graph = combine_Graph([cation, anion])
        # print('before',Combine_Graph.x.shape)
        if args['add_global'] == True:
            Combine_Graph = add_global(Combine_Graph)
        # print('after',Combine_Graph.x.shape )
        condition = torch.tensor([T,P],dtype=torch.float)


        label = torch.tensor(self.label[idx],dtype=torch.float)


        return Combine_Graph,condition,label

    def mol2graph(self,mol):
        x = torch.tensor(mol[0],dtype=torch.long)
        edge_index = torch.tensor(mol[1],dtype=torch.long)
        edge_attr = torch.tensor(mol[2],dtype=torch.long)

        # debug
        # print("mol",x.shape,edge_index.shape,edge_attr.shape)

        Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return Graph

    def collate_fn(batch):
        batch_x = torch.as_tensor(batch)

        return batch_x


if __name__ == '__main__':
    args = {
        'data_path':"clean/"
    }
    D = IL_set(path = args['data_path'])
    print(len(D))
    for item in D:
        G,c,l = item
        print(c,l)