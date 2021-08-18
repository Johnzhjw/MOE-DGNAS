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

import matplotlib as mpl
import matplotlib.pyplot as plt


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
    mpl.use('Ps')

    overall_best_obj = []
    all_final_pop = []
    for dataset in ["cora", "citeseer", "pubmed"]:  #"ppi",
        folder = 'Result_' + dataset

        MAXGEN = 50  # 最大进化代数

        all_gen = []
        all_best_obj = []
        for i_gen in range(MAXGEN):
            fileObj = folder + '/pop_acc' + '%04d' % i_gen + '/ObjV.csv'
            fileSol = folder + '/pop_acc' + '%04d' % i_gen + '/Phen.csv'
            objectives = np.loadtxt(open(fileObj, "r"), delimiter=",", skiprows=0)
            solutions = np.loadtxt(open(fileSol, "r"), delimiter=",", skiprows=0)

            goodId = np.argsort(-objectives[:, 0])
            best_val_acc = objectives[goodId[0], :]
            all_best_obj.append(best_val_acc)
            all_gen.append(i_gen + 1)

            if i_gen == MAXGEN - 1:
                final_pop = objectives

        overall_best_obj.append(all_best_obj)
        all_final_pop.append(final_pop)

    # 创建画布
    plt.figure()
    plt.plot(all_gen, np.array(overall_best_obj[0])[:, 0], marker='.', color='black', label='Cora')
    plt.plot(all_gen, np.array(overall_best_obj[1])[:, 0], marker='.', color='r', label='Citeseer')
    plt.plot(all_gen, np.array(overall_best_obj[2])[:, 0], marker='.', color='b', label='Pubmed')
    # 显示图例（使绘制生效）
    plt.legend(loc='best')
    # 坐标名称
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    # 保存图片到本地
    plt.savefig('curve_acc.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.plot(all_gen, np.array(overall_best_obj[0])[:, 1], marker='.', color='black', label='Cora')
    plt.plot(all_gen, np.array(overall_best_obj[1])[:, 1], marker='.', color='r', label='Citeseer')
    plt.plot(all_gen, np.array(overall_best_obj[2])[:, 1], marker='.', color='b', label='Pubmed')
    # 显示图例（使绘制生效）
    plt.legend(loc='best')
    # 坐标名称
    plt.xlabel('Generation')
    plt.ylabel('Complexity')
    # 保存图片到本地
    plt.savefig('curve_comp.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.plot(all_gen, np.array(overall_best_obj[0])[:, 2], marker='.', color='black', label='Cora')
    plt.plot(all_gen, np.array(overall_best_obj[1])[:, 2], marker='.', color='r', label='Citeseer')
    plt.plot(all_gen, np.array(overall_best_obj[2])[:, 2], marker='.', color='b', label='Pubmed')
    # 显示图例（使绘制生效）
    plt.legend(loc='best')
    # 坐标名称
    plt.xlabel('Generation')
    plt.ylabel('Generality')
    # 保存图片到本地
    plt.savefig('curve_gene.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[0])[:, 0], np.array(all_final_pop[0])[:, 1])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Complexity')
    # 保存图片到本地
    plt.savefig('Cora' + '_acc_comp.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[0])[:, 0], np.array(all_final_pop[0])[:, 2])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Generality')
    # 保存图片到本地
    plt.savefig('Cora' + '_acc_gene.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[1])[:, 0], np.array(all_final_pop[1])[:, 1])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Complexity')
    # 保存图片到本地
    plt.savefig('Citeseer' + '_acc_comp.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[1])[:, 0], np.array(all_final_pop[1])[:, 2])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Generality')
    # 保存图片到本地
    plt.savefig('Citeseer' + '_acc_gene.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[2])[:, 0], np.array(all_final_pop[2])[:, 1])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Complexity')
    # 保存图片到本地
    plt.savefig('Pubmed' + '_acc_comp.eps', format='eps', bbox_inches='tight')

    # 创建画布
    plt.figure()
    plt.scatter(np.array(all_final_pop[2])[:, 0], np.array(all_final_pop[2])[:, 2])
    # 显示图例（使绘制生效）
    # plt.legend()
    # 坐标名称
    plt.xlabel('Accuracy')
    plt.ylabel('Generality')
    # 保存图片到本地
    plt.savefig('Pubmed' + '_acc_gene.eps', format='eps', bbox_inches='tight')
