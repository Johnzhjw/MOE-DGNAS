# -*- coding: utf-8 -*-
import numpy as np
import math
import geatpy as ea

import time
import torch
from torch.nn import functional as F
import dgl
from dgl.data import load_data
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from GNN_model import GNNmodel
from utils import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from gnn_torch import GraphNet


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def f1_multi_label(model, subgraph, loss_fcn):
    with torch.no_grad():
        model.eval()
        output = model(subgraph.ndata['feat'], subgraph)
        labels = subgraph.ndata['label']
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0., 1, 0)
        score = f1_score(labels.data.cpu().numpy(),
                         predict, average='micro')
        return score, loss_data.item()


def evaluate(model, loss_fcn, g, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        loss = loss_fcn(logits, labels)
        return accuracy(logits, labels), loss


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, args, search_space, maxN_layers):
        self.name = 'evoGNN'  # 初始化name（函数名称，可以随意设置）
        self.Dim = search_space.get_dim_num(maxN_layers)  # 初始化Dim（决策变量维数）
        self.M = 2
        self.maxormins = [-1, 1]  # [-1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        self.varTypes = [ty not in ['float'] for ty in
                         search_space.get_varTypes(maxN_layers)]  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        # self.varTypes = [0] * self.Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        self.lb = search_space.get_lb(maxN_layers)  # 决策变量下界
        self.ub = search_space.get_ub(maxN_layers)  # 决策变量上界
        self.lbin = [1] * self.Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        self.ubin = [1] * self.Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.xlimits = np.concatenate((np.array(self.lb).reshape(-1, 1), np.array(self.ub).reshape(-1, 1)), axis=1)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, self.name, self.M, self.maxormins, self.Dim, self.varTypes, self.lb, self.ub,
                            self.lbin, self.ubin)
        #
        self.__DEBUG = True
        #
        if args.gpu < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:" + str(args.gpu))
        self.args = args
        self.search_space = search_space
        self.maxN_layers = maxN_layers
        # load and preprocess dataset
        if args.dataset in ["cora", "citeseer", "pubmed"]:
            data = load_data(args)

            self.g = data[0]
            if args.gpu < 0:
                self.cuda = False
            else:
                self.cuda = True
                self.g = self.g.int().to(self.device)

            self.labels = self.g.ndata['label']
            self.train_mask = self.g.ndata['train_mask']
            self.val_mask = self.g.ndata['val_mask']
            self.test_mask = self.g.ndata['test_mask']
            self.num_feats = self.g.ndata['feat'].shape[1]
            self.n_classes = data.num_labels
            self.n_edges = data.graph.number_of_edges()
            print("""----Data statistics------'
              #Edges %d
              #Classes %d
              #Train samples %d
              #Val samples %d
              #Test samples %d""" %
                  (self.n_edges, self.n_classes,
                   self.train_mask.int().sum().item(),
                   self.val_mask.int().sum().item(),
                   self.test_mask.int().sum().item()))

            # add self loop
            if (self.g.in_degrees() == 0).any():
                print('There are 0-in-degree nodes in the graph, ')
                print('self loop will be added.')
                self.g = dgl.remove_self_loop(self.g)
                self.g = dgl.add_self_loop(self.g)
                self.n_edges = self.g.number_of_edges()
            degs = self.g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            self.g.ndata['norm'] = norm.unsqueeze(1)

        elif args.dataset in ["ppi"]:
            self.ppi_train = PPIDataset(mode='train')
            self.ppi_valid = PPIDataset(mode='valid')
            self.ppi_test = PPIDataset(mode='test')
            self.train_dataloader = GraphDataLoader(self.ppi_train, batch_size=args.batch_size)
            self.valid_dataloader = GraphDataLoader(self.ppi_valid, batch_size=args.batch_size)
            self.test_dataloader = GraphDataLoader(self.ppi_test, batch_size=args.batch_size)
            '''
            if args.gpu < 0:
                self.cuda = False
            else:
                self.cuda = True
                self.ppi_train = [g.int().to(self.device) for g in self.ppi_train]
                self.ppi_valid = [g.int().to(self.device) for g in self.ppi_valid]
                self.ppi_test = [g.int().to(self.device) for g in self.ppi_test]
            '''
            self.num_feats = self.ppi_train[0].ndata['feat'].shape[1]
            self.n_classes = self.ppi_train[0].ndata['label'].shape[1]
            print("""----Data statistics------""")
            print("#Train Edges ", [g.number_of_edges() for g in self.ppi_train])
            print("#Train Edges Sum ", np.sum([g.number_of_edges() for g in self.ppi_train]))
            print("#Val Edges ", [g.number_of_edges() for g in self.ppi_valid])
            print("#Val Edges Sum ", np.sum([g.number_of_edges() for g in self.ppi_valid]))
            print("#Test Edges ", [g.number_of_edges() for g in self.ppi_test])
            print("#Test Edges Sum ", np.sum([g.number_of_edges() for g in self.ppi_test]))
            print("#Classes ", self.n_classes)
            print("#Train samples ", [g.number_of_nodes() for g in self.ppi_train])
            print("#Train samples Sum ", np.sum([g.number_of_nodes() for g in self.ppi_train]))
            print("#Val samples ", [g.number_of_nodes() for g in self.ppi_valid])
            print("#Val samples Sum ", np.sum([g.number_of_nodes() for g in self.ppi_valid]))
            print("#Test samples ", [g.number_of_nodes() for g in self.ppi_test])
            print("#Test samples Sum ", np.sum([g.number_of_nodes() for g in self.ppi_test]))
            self.n_edges = np.sum([g.number_of_edges() for g in self.ppi_train])
            '''
            # add self loop
            self.ppi_train_list = []
            for g in self.ppi_train:
                if (g.in_degrees() == 0).any():
                    print('There are 0-in-degree nodes in the graph, ')
                    print('self loop will be added.')
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                self.ppi_train_list.append(g)
            self.ppi_valid_list = []
            for g in self.ppi_valid:
                if (g.in_degrees() == 0).any():
                    print('There are 0-in-degree nodes in the graph, ')
                    print('self loop will be added.')
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                self.ppi_valid_list.append(g)
            self.ppi_test_list = []
            for g in self.ppi_test:
                if (g.in_degrees() == 0).any():
                    print('There are 0-in-degree nodes in the graph, ')
                    print('self loop will be added.')
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                self.ppi_test_list.append(g)
            '''
            self.n_edges_train = np.sum([g.number_of_edges() for g in self.ppi_train])
            self.n_edges_valid = np.sum([g.number_of_edges() for g in self.ppi_valid])
            self.n_edges_test = np.sum([g.number_of_edges() for g in self.ppi_test])
        else:
            raise Exception("wrong dataset")
        #
        '''
        if self.args.base_model:
            actions = search_space.generate_action_base(maxN_layers)
            self.modelBase = GNNmodel(
                self.num_feats,
                self.n_classes,
                actions,
                self.search_space,
                True)
            self.modelBase.to(self.device)
            self.best_score = None
        '''

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        #
        pop.ObjV = np.zeros([Vars.shape[0], self.M])  # 把求得的目标函数值赋值给种群pop的ObjV
        #
        for i, inv in enumerate(Vars):
            actions, model, train_loss, train_acc, val_loss, val_acc, model_val_acc, test_acc, test_acc_best, dur = \
                self.eval_one_solution(inv, self.args.early_stop, self.args.base_model, self.args.epochs)
            print(actions, '; train acc: ', train_acc,
                  '; val acc: ', model_val_acc,
                  '; test acc: ', test_acc_best)
            '''
            if self.args.save_model:
                nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在
                torch.save(model.state_dict(),
                           "saved_models/" + "gnn+" + str(nowTime) + ".pt")
            '''
            pop.ObjV[i, 0] = model_val_acc  # stopper.best_score
            pop.ObjV[i, 1] = math.log(sum(p.numel() for p in model.parameters()))
            if self.M > 2:
                wm = 0.
                cnt = 0.
                md_dict = model.state_dict()
                for key in md_dict:
                    if 'bias' not in key and 'op_bn' not in key:
                        wm += torch.norm(md_dict[key], 2)
                        cnt += md_dict[key].numel()
                pop.ObjV[i, 2] = math.log(wm) - math.log(cnt)

    def aimFunc_no_train(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        #
        pop.ObjV = np.zeros([Vars.shape[0], self.M])  # 把求得的目标函数值赋值给种群pop的ObjV
        #
        for i, inv in enumerate(Vars):
            actions = self.search_space.generate_actions_4_solution(inv)
            actions = self.search_space.remove_invalid_layer(actions)
            actions[-1] = self.n_classes
            print('no_train: ', actions)
            model = GraphNet(actions, self.num_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                             batch_normal=False, state_num=self.search_space.state_num-1, flag_se=True)
            '''
            model = GNNmodel(
                self.num_feats,
                self.n_classes,
                actions,
                self.search_space)
            '''
            pop.ObjV[i, 1] = math.log(sum(p.numel() for p in model.parameters()))
            if self.M > 2:
                wm = 0.
                cnt = 0.
                md_dict = model.state_dict()
                for key in md_dict:
                    if 'bias' not in key and 'op_bn' not in key:
                        wm += torch.norm(md_dict[key], 2)
                        cnt += md_dict[key].numel()
                pop.ObjV[i, 2] = math.log(wm) - math.log(cnt)

    def eval_one_solution(self, inv, early_stop, base_model, epochs):
        actions = self.search_space.generate_actions_4_solution(inv)
        short_cut = actions[-1] if self.search_space.tag_all else False
        actions = self.search_space.remove_invalid_layer(actions)
        actions[-2] = self.n_classes
        model = GraphNet(actions, self.num_feats, self.n_classes, drop_out=self.args.in_drop, multi_label=False,
                         batch_normal=False, state_num=self.search_space.state_num-1, flag_se=True, short_cut=short_cut)
        '''
        model = GNNmodel(
            self.num_feats,
            self.n_classes,
            actions,
            self.search_space)
        '''
        if self.__DEBUG:
            print(model)
        patience = 100
        if early_stop:
            stopper = EarlyStopping(patience=patience)
        model.to(self.device)
        '''
        if base_model:
            model.copy_para(self.modelBase)
        '''

        # use optimizer
        lr = self.args.lr
        wd = self.args.weight_decay
        optimizer = torch.optim.Adam(
            [{'params': (p for name, p in model.named_parameters() if 'bias' not in name and 'op_bn' not in name)},
            {'params': (p for name, p in model.named_parameters() if 'bias' in name or 'op_bn' in name), 'weight_decay': 0.}],
            lr=lr, weight_decay=wd)
        # print(optimizer)
        # initialize graph
        test_acc_best = 0
        val_loss_min = float("inf")
        train_loss_min = float("inf")
        model_val_acc = 0
        dur = []
        for epoch in range(epochs):
            if epoch >= 3:
                t0 = time.time()
            # forward
            if self.args.dataset in ["cora", "citeseer", "pubmed"]:
                loss_fcn = torch.nn.CrossEntropyLoss()
                model.train()
                logits = model(self.g.ndata['feat'], self.g)
                loss = loss_fcn(logits[self.train_mask], self.labels[self.train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # evaluate
                model.eval()
                logits = model(self.g.ndata['feat'], self.g)
                # logits = F.log_softmax(logits, 1)

                if epoch >= 3:
                    dur.append(time.time() - t0)

                train_acc = accuracy(logits[self.train_mask], self.labels[self.train_mask])

                val_acc = accuracy(logits[self.val_mask], self.labels[self.val_mask])
                val_loss = loss_fcn(logits[self.val_mask], self.labels[self.val_mask]).item()

                test_acc = accuracy(logits[self.test_mask], self.labels[self.test_mask])
            elif self.args.dataset in ["ppi"]:
                loss_fcn = torch.nn.BCEWithLogitsLoss()
                model.train()
                loss_list = []
                for batch, subgraph in enumerate(self.train_dataloader):
                    subgraph = subgraph.to(self.device)
                    logits = model(subgraph.ndata['feat'], subgraph)
                    loss = loss_fcn(logits, subgraph.ndata['label'])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())
                loss = np.array(loss_list).mean()
                train_acc = float("inf")

                # evaluate
                score_list = []
                val_loss_list = []
                for batch, subgraph in enumerate(self.valid_dataloader):
                    subgraph = subgraph.to(self.device)
                    score, val_loss = f1_multi_label(model, subgraph, loss_fcn)
                    score_list.append(score)
                    val_loss_list.append(val_loss)
                val_acc = np.array(score_list).mean()
                val_loss = np.array(val_loss_list).mean()
                score_list = []
                val_loss_list = []
                for batch, subgraph in enumerate(self.test_dataloader):
                    subgraph = subgraph.to(self.device)
                    score, val_loss = f1_multi_label(model, subgraph, loss_fcn)
                    score_list.append(score)
                    val_loss_list.append(val_loss)
                test_acc = np.array(score_list).mean()
                test_loss = np.array(val_loss_list).mean()

                if epoch >= 3:
                    dur.append(time.time() - t0)
            else:
                raise Exception("wrong dataset")

            if val_loss < val_loss_min:  # and loss < train_loss_min
                val_loss_min = val_loss
                train_loss_min = loss
                model_val_acc = val_acc
                if test_acc > test_acc_best:
                    test_acc_best = test_acc
                '''
                if self.args.base_model and base_model:
                    if self.best_score is None:
                        self.best_score = val_acc
                        self.modelBase.save_para(model)
                    elif self.best_score < val_acc:
                        self.best_score = val_acc
                        self.modelBase.save_para(model)
                '''

            if early_stop:
                if stopper.should_stop(float(loss), train_acc, val_loss, val_acc, epoch):
                    break

            if self.__DEBUG:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                      " ValAcc {:.4f} | TestAcc {:.4f} | BestTestAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                      format(epoch, np.mean(dur), float(loss), train_acc,
                             val_acc, test_acc, test_acc_best, self.n_edges / np.mean(dur) / 1000))

        if self.args.dataset in ["ppi"]:
            score_list = []
            train_loss_list = []
            for batch, subgraph in enumerate(self.train_dataloader):
                subgraph = subgraph.to(self.device)
                score, train_loss = f1_multi_label(model, subgraph, loss_fcn)
                score_list.append(score)
                train_loss_list.append(train_loss)
            train_acc = np.array(score_list).mean()
            loss = np.array(train_loss_list).mean()

        # print()
        # if self.args.early_stop:
        #    model.load_state_dict(torch.load('es_checkpoint.pt'))
        # acc, _ = evaluate(model, loss_fcn, self.g, self.labels, self.test_mask)
        # print("Test Accuracy {:.4f}".format(acc))

        return actions, model, float(loss), train_acc, \
               val_loss, val_acc, model_val_acc, test_acc, test_acc_best, np.mean(dur)
