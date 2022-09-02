import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

from Dataset import IL_set
from Model import IL_Net_GCN

Args = {
    # general hyperparameters
    'smiles_dict_path':'data/smiles.csv',
    'data_path':'clean/',
    'load_history_model':True,
    'batch_size':64,
    'lr':0.001,
    'epoch':0,
    'weight_decay':1e-6,
    'warmup': 40,

    # GCN model hyperparameter
    'dropout_rate':0.5,
    'n_features':5

}

class Runner(object):
    """
    include all the function needed for training
    """
    def __init__(self,args):
        self.args = args
        self._device = self._get_device()

        if args['load_history_model'] == True:
            print("loading history model..")
            self._model = IL_Net_GCN(args)
            state_dict_mod = torch.load('pretrained_model/GCN_300/best_model_para.pth')
            self._model.load_state_dict(state_dict_mod)

            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            # state_dict_opt = torch.load('pretrained_model/GCN_300/best_optimizer_para.pth')
            # self._optimizer.load_state_dict(state_dict_opt)
            # self._optimizer.lr = args['lr']
            #
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=args['epoch']-9)
            # state_dict_sch = torch.load('pretrained_model/GCN_300/best_scheduler_para.pth')
            # self._scheduler.load_state_dict(state_dict_sch)
            print("finish loading")
        else:
            self._model = IL_Net_GCN(args)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=args['epoch'] - 9)

        self._criterion = nn.L1Loss()


    def _get_device(self):
        """

        Returns: the device for the training process

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on:", device)

        return device

    def _save_para(self,title):
        """
        save the state_dict for the model and other workers
        Returns: N/A

        """
        print("current epoch is more accurate than before and did not overfit the data, so save it")
        torch.save(self._model.state_dict(), title + '_model_para.pth')
        torch.save(self._optimizer.state_dict(), title + '_optimizer_para.pth')
        torch.save(self._scheduler.state_dict(), title + '_scheduler_para.pth')


    def train(self,train_loader,dev_loader,args):
        """

        Args:
            train_loader: Dataloader
            dev_loader: Dataloader

        Returns: float type validate loss

        """

        # initialize the parameter
        model = self._model.to(self._device)
        optimizer = self._optimizer
        scheduler = self._scheduler

        # initialize the recorder
        batch_loss = []
        v_loss = -1
        for epoch in range(1,args['epoch'] + 1):

            model.train()
            batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Train')
            # training process
            train_loss = 0
            for batch_idx,(graph,cond,label) in enumerate(train_loader):
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)

                optimizer.zero_grad()
                y = model(graph,cond)
                loss = self._criterion(y.flatten(),label.flatten())
                loss.backward()
                optimizer.step()
                train_loss += loss

                batch_loss.append(loss)
                # record the progress
                batch_bar.set_postfix(
                    loss="{:.04f}".format(float(train_loss/(batch_idx + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                batch_bar.update()
            batch_bar.close()
            # validate process
            model.eval()
            batch_bar = tqdm(total=len(dev_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
            with torch.no_grad():
                Loss_v = 0
                batch_num = 0
                for batch_idx,(graph,cond,label) in enumerate(dev_loader):
                    graph = graph.to(self._device)
                    cond = cond.to(self._device)
                    label = label.to(self._device)

                    y = model(graph,cond)
                    loss_v = self._criterion(y.flatten(),label.flatten())
                    Loss_v += loss_v
                    batch_num += 1

                    batch_bar.set_postfix(
                        v_loss="{:.04f}".format(float(Loss_v / (batch_idx + 1)))
                    )
                    batch_bar.update()

                batch_bar.close()
                total_loss = Loss_v

                if v_loss == -1:
                    v_loss = total_loss
                    self._save_para('init')
                elif total_loss <= v_loss:
                    v_loss = total_loss
                    self._save_para('best')

                if epoch == 80:
                    self._save_para('80')

            # scheduler step
            if epoch >= args['warmup']:
                scheduler.step()

            # basic information for the current epoch
            print(
                "Epoch {}/{}: Train loss {:.04f}, Validation Loss {:.04f}, Learning Rate {:.04f}".format(
                    epoch,
                    args['epoch'],
                    float(train_loss / len(train_loader)),
                    float(total_loss / len(dev_loader)),
                    float(optimizer.param_groups[0]['lr'])))

        return total_loss

    def test(self,test_loader):
        """

        Args:
            test_loader: Dataloader

        Returns: list(), list()

        """
        # initialize the parameter
        model = self._model.to(self._device)

        pred_y = []
        true_y = []
        model.eval()
        batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
        with torch.no_grad():
            for batch_idx,(graph,cond,label) in enumerate(test_loader):
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)


                pred = model(graph,cond)

                try:
                    pred_y.extend(pred.flatten().tolist())
                    true_y.extend(label.flatten().tolist())
                except:
                    print("test result appended failed, seems like need to clarify the device")

                batch_bar.set_postfix()
                batch_bar.update()
        batch_bar.close()
        mae = mean_absolute_error(true_y, pred_y)
        r2 = r2_score(true_y,pred_y)
        print("MAE: {} ,R2: {} ".format(mae,r2))
        return pred_y,true_y

    def get_model(self):
        """

        Returns:torch.nn.model

        """
        return self._model


def plot(labels, predictions, tl, tp, name):
    xymin = min(np.min(labels), np.max(predictions))
    xymax = max(np.max(labels), np.max(predictions))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(labels, predictions, s=25, edgecolors='None', linewidths=0.4, c='orange', label='train data')
    ax.scatter(tl, tp, s=25, edgecolors='None', linewidths=0.4, c='yellow', label='test data')
    x = np.linspace(xymin, xymax)
    y = np.linspace(xymin, xymax)
    ax.plot(x, y, linestyle='dashed', c='black')

    ax.set_xlabel('Label', fontsize=18)
    ax.set_ylabel('Prediction', fontsize=18)
    ax.tick_params(direction='in', width=2, labelsize=15)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.legend(fontsize=15)
    plt.savefig(name)


if __name__ == '__main__':
    # init dataset
    Whole_set = IL_set(path = Args['data_path'])

    # print basic information
    Whole_size = len(Whole_set)
    print("num of data",Whole_size)

    # data split
    train_size = int(len(Whole_set) * 0.8)
    dev_size = Whole_size - train_size
    train_set,dev_set = random_split(Whole_set,[train_size,dev_size])

    # data loading
    train_loader = DataLoader(train_set,batch_size = Args['batch_size'],shuffle=True,collate_fn=IL_set.collate_fn)
    dev_loader = DataLoader(dev_set,batch_size = Args['batch_size'],shuffle=True,collate_fn=IL_set.collate_fn)
    explain_loader = DataLoader(dev_set,batch_size = Args['batch_size'],shuffle=False,collate_fn=IL_set.collate_fn)

    # init Runner
    run_G = Runner(Args)

    # model train & test
    if Args['epoch'] != 0:# for test mode
        run_G.train(train_loader,dev_loader,Args)

    # train plot
    train_pred, train_true = run_G.test(train_loader)

    # test plot
    test_pred,test_true = run_G.test(explain_loader)
    plot(train_true, train_pred,test_true,test_pred,'GCN')