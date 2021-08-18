# -*- coding: utf-8 -*-
import geatpy as ea  # import geatpy
from MOP_GNN import MyProblem  # 导入自定义问题接口
from CCMOEA import CCNSGA2_archive

import sys
import os
import datetime
import argparse
import torch
import numpy as np
import random
from dgl.data import register_data_args, load_data
from search_space import MacroSearchSpace


class Logger(object):
    def __init__(self, filename="log_train.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    # sys.stdout = Logger()

    parser = argparse.ArgumentParser(description='evoGNN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--random_seed', type=int, default=81)
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8, help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1, help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128, help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6, help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6, help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2, help="the negative slope of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2, help="batch size used for training, validation and test")
    parser.add_argument('--early-stop', action='store_true', default=True, help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False, help="skip re-evaluate the validation set")
    parser.add_argument('--save_model', action="store_true", default=True, help="whether save the whole model")
    parser.add_argument('--base_model', action="store_true", default=False, help="whether use the base model")
    parser.add_argument("--aggregator-type", type=str, default="pool", help="Aggregator type: mean/sum/pool/lstm")
    parser.add_argument("--attention-type", type=str, default="gat", help="Attention type: const/gcn/gat/sym-gat/cos/linear/gen_linear")
    parser.add_argument("--activation-type", type=str, default="relu", help="Attention type: linear/elu/sigmoid/tanh/relu/relu6/softplus/leaky_relu")
    parser.add_argument("--name_surrog", type=str, default="KRG_MIXINT", help="Surrogate type: LS/QP/KPLS_squar/KPLS_abs/KRG/KPLSK/IDW/RBF")
    args = parser.parse_args()

    for dataset in ["cora", "citeseer", "pubmed"]:  #"ppi",
        args.dataset = dataset
        args.gpu = 0
        print(args)

        folder = 'Result_' + dataset

        torch.manual_seed(args.random_seed)
        if args.gpu >= 0:
            torch.cuda.manual_seed_all(args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        search_space = MacroSearchSpace(tag_all=True)

        """===============================实例化问题对象============================"""
        maxN_layers = 2
        problem = MyProblem(args, search_space, maxN_layers)  # 生成问题对象
        """==================================种群设置==============================="""
        """=================================算法参数设置============================"""
        MAXGEN = 50  # 最大进化代数
        """==================================输出结果=============================="""

        fileObj = folder + '/pop_acc' + '%04d' % (MAXGEN-1) + '/ObjV.csv'
        fileSol = folder + '/pop_acc' + '%04d' % (MAXGEN-1) + '/Phen.csv'
        objectives = np.loadtxt(open(fileObj, "r"), delimiter=",", skiprows=0)
        solutions = np.loadtxt(open(fileSol, "r"), delimiter=",", skiprows=0)

        goodId = np.argsort(-objectives[:, 0])[:10]
        solutions = solutions[goodId, :]

        num_cut = 1
        all_results = []
        for solution in solutions:
            tmp = []
            durs = []
            for _ in range(10):
                actions, model, train_loss, train_acc, val_loss, val_acc, model_val_acc, test_acc, test_acc_best, dur = \
                    problem.eval_one_solution(solution, early_stop=False, base_model=args.base_model, epochs=200)
                tmp.append(test_acc_best)
                durs.append(dur)
            print("_" * 80)
            tmp.sort()
            print(dataset, actions,
                  sum(p.numel() for p in model.parameters()), np.mean(durs), np.mean(tmp), np.std(tmp), solution)
            all_results.append([np.mean(tmp), np.std(tmp),
                                sum(p.numel() for p in model.parameters()), np.mean(durs)] + solution.tolist())
        np.savetxt(folder + '/all_test0_mean_std.csv', np.array(all_results), delimiter=',')
        print("_" * 80)
        best_ind = np.argsort(-np.array(all_results)[:, 0])[0]
        best_sol = solutions[best_ind]
        tmp = []
        durs = []
        for _ in range(100):
            actions, model, train_loss, train_acc, val_loss, val_acc, model_val_acc, test_acc, test_acc_best, dur = \
                problem.eval_one_solution(best_sol, early_stop=False, base_model=args.base_model, epochs=200)
            tmp.append(test_acc_best)
            durs.append(dur)
        print("_" * 80)
        tmp.sort()
        print('best: ', dataset, ' ', actions, '; ',
              sum(p.numel() for p in model.parameters()), ';',
              np.mean(durs), ';', np.mean(tmp), np.std(tmp), ';', best_sol)
        np.savetxt(folder + '/final_test0_mean_std.csv',
                   [sum(p.numel() for p in model.parameters()), np.mean(durs), np.mean(tmp), np.std(tmp)] + best_sol.tolist(),
                   delimiter=',')
