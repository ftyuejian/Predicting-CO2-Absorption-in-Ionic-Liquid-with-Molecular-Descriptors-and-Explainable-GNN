'''
1,import molecule
2,count the fragment and store the fragment atom idx
3,make it into a graph and pass into dataloader
4,load pretrain model and explain each molecule
5,count the score for each fragment and store them in a globel variable
'''
import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
import matplotlib as mpl
from matplotlib import pyplot as plt
from Dataset_fragment import IL_set
from Model import IL_Net_GCN,GIN,IL_GAT
from Explainer import IL_Explainer
from GIN_Runner import Args
from tqdm import tqdm

args = {
    'data_path':"clean/",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # define importance dict()
    Frag_importance = {
        # range(0,7)
        '[OH]': [],
        '[NH3]': [],
        '[NH2]': [],
        '[NH]': [],
        '[CH]': [],
        '[CH2]': [],
        '[CH3]': [],
        # No add H: range(7,13)
        '[N]': [],
        'OC': [],
        'C(F)F': [],
        'C(F)(F)F': [],
        '[P]': [],
        '[S]': [],
        # smiles: range(13,15)
        'C[N+]1=CN(C=C1)C': [],
        'C[N+]1=CC(=CC=C1)': []
    }

    # define node imp
    node_feat_imp = np.zeros(5)

    # loading pretrain model
    model = GIN(Args).to(device)
    state_dict_mod = torch.load('best_model_para.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict_mod)

    # loading dataset
    Whole_set = IL_set(path=args['data_path'])
    explain_loader = DataLoader(Whole_set, batch_size=1, shuffle=False, collate_fn=IL_set.collate_fn)

    # define explainer
    explainer = IL_Explainer(model, epochs=100, lr=0.001, return_type='log_prob')

    # define batch_bar
    batch_bar = tqdm(total=len(explain_loader), dynamic_ncols=True, position=0, leave=False, desc='explain')

    # optimizing explain result
    for i, (G, c, l, dic, b_num) in enumerate(explain_loader):
        G = G.to(device)
        c = c.to(device)

        node_feat_mask, edge_mask = explainer.explain_graph(G, c)
        node_feat_imp += node_feat_mask.cpu().numpy()

        bi_node_mask = edge_mask[b_num:].reshape(2, -1)
        node_mask = torch.mean(bi_node_mask, dim=0)
        atom_mask = node_mask.tolist()
        mean_score = torch.mean(node_mask).item()

        for frag in dic:

            if len(dic[frag]) == 0:
                continue

            for piece in dic[frag]:
                piece_abs = []
                for atom_id in piece:
                    atom_id = atom_id[0].item()
                    piece_abs.append(atom_mask[atom_id])
                piece_abs = np.mean(np.array(piece_abs))
                piece_relative = piece_abs - mean_score
                Frag_importance[frag].append(piece_relative)
        batch_bar.set_postfix()
        batch_bar.update()

    batch_bar.close()

    # pre_save
    raw = np.array(Frag_importance,dtype= object)
    np.save(file='frag_importance_raw', arr=raw, allow_pickle=True)
    np.save(file='node_feat_imp', arr=node_feat_imp, allow_pickle=False)

    # summing the score list
    result = dict()
    for frag_ in Frag_importance:
        score = Frag_importance[frag_]
        score = np.array(score).mean()
        result[frag_] = score

    # saving the result
    result = np.array(result, dtype=object)
    np.save(file='frag_importance_mean', arr=result, allow_pickle=True)