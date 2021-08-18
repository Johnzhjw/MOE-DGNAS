# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from sys import path as paths
from os import path
import random
from SurroModel import get_surrogate_model

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


def remove_duplication(pop):
    x = pop.decoding()
    x_unique, ind = np.unique(x, return_index=True, axis=0)
    ind = np.sort(ind)
    pop = pop[ind]
    return pop, ind


class CCNSGA2_archive(ea.MoeaAlgorithm):

    def __init__(self, problem, population, grps, CC_flag, CC_type, surrog_flag, name_surrog, folder):
        ea.MoeaAlgorithm.__init__(self, problem, population)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'CCNSGA2-archive'
        self.grps = grps
        if population.Encoding == 'BG':
            self.bit_lens = population.Field[0].astype(np.int32)
            self.bit_offs = np.cumsum(self.bit_lens).astype(np.int32)
            self.grps = []
            for grp in grps:
                tmp_grp = []
                for i in grp:
                    for _ in range(self.bit_lens[i]):
                        tmp_grp.append(self.bit_offs[i] - self.bit_lens[i] + _)
                self.grps.append(tmp_grp)
        self.rem_grps = []
        for _ in range(len(grps)):
            rem_idx = []
            for i, grp in enumerate(self.grps):
                if i != _:
                    rem_idx += grp
            self.rem_grps.append(rem_idx)
        self.CC_flag = CC_flag
        self.CC_type = CC_type
        self.surrog_flag = surrog_flag
        self.name_surrog = name_surrog
        self.folder = folder
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        '''
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
            self.crsOper = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
        '''
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.pmOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
            self.crsOper1 = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
            self.crsOper2 = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
            self.xosp = ea.Xovsp(XOVR=1.0)
            self.xodp = ea.Xovdp(XOVR=1.0)
            self.gmOper = ea.Mutgau(Pm=1 / self.problem.Dim)
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=0.5)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
            self.crsOper1 = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
            self.crsOper2 = ea.Xovbd(XOVR=0.5, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
            self.xosp = ea.Xovsp(XOVR=1.0)
            self.xodp = ea.Xovdp(XOVR=1.0)
            self.gmOper = ea.Mutgau(Pm=None)
        else:
            raise RuntimeError('编码方式必须为''BG''或''RI''.')
        self.NP = population.sizes
        self.MAXSIZE = 10 * population.sizes  # 全局非支配解存档的大小限制，默认为10倍的种群个体数
        self.MAXSIZE_acc = population.sizes
        self.useSurrogate = False
        self.surrogate = None
        self.MAXSIZE_surrogate = 150
        self.gap_surrogate = 5
        self.ngn_surrogate = 100
        self.Pm_min = 1.0 / self.problem.Dim
        if population.Encoding == 'BG':
            self.Pm_min = 1.0 / sum(population.Field[0])
        self.Pm_max = 0.5  # 3 * self.Pm_min
        self.Pm_rng = self.Pm_min - self.Pm_min
        self.XOVR_max = 0.5
        self.XOVR_min = 0.1
        self.XOVR_rng = self.XOVR_max - self.XOVR_min
        self.Encoding = population.Encoding

    def call_aimFunc(self, pop):

        """
        使用注意:
            本函数调用的目标函数形如：aimFunc(pop), (在自定义问题类中实现)。
            其中pop为种群类的对象，代表一个种群，
            pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵。
            若不符合上述规范，则请修改算法模板或自定义新算法模板。

        描述:
            该函数调用自定义问题类中自定义的目标函数aimFunc()得到种群所有个体的目标函数值组成的矩阵，
            以及种群个体违反约束程度矩阵（假如在aimFunc()中构造了该矩阵的话）。
            该函数不返回任何的返回值，求得的目标函数值矩阵保存在种群对象的ObjV属性中，
            违反约束程度矩阵保存在种群对象的CV属性中。
        例如：population为一个种群对象，则调用call_aimFunc(population)即可完成目标函数值的计算。
             之后可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

        输入参数:
            pop : class <Population> - 种群对象。

        输出参数:
            无输出参数。

        """

        # print(pop.sizes)
        pop.Phen = pop.decoding()  # 染色体解码
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
        if self.useSurrogate:
            if len(pop) > 0:
                self.problem.aimFunc_no_train(pop)  # 调用问题类的aimFunc()
                acc_pred = self.surrogate.predict_values(pop.Phen)
                pop.ObjV[:, 0] = acc_pred.reshape(-1)
        else:
            self.problem.aimFunc(pop)  # 调用问题类的aimFunc()
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
        # 格式检查
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')

    def reinsertion(self, population, offspring, NUM, globalNDSet):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目，globalNDSet为全局非支配解存档。
        """

        # 父子两代合并
        population = population + offspring
        globalNDSet = population + globalNDSet  # 将population与全局存档合并
        # 非支配排序分层
        [levels, criLevel] = self.ndSort(globalNDSet.ObjV, None, None, globalNDSet.CV, self.problem.maxormins)
        # 更新全局存档
        globalNDSet = globalNDSet[np.where(levels == 1)[0]]
        if globalNDSet.CV is not None:  # CV不为None说明有设置约束条件
            globalNDSet = globalNDSet[np.where(np.all(globalNDSet.CV <= 0, 1))[0]]  # 排除非可行解
        if globalNDSet.sizes > self.MAXSIZE:
            dis = ea.crowdis(globalNDSet.ObjV, np.ones(globalNDSet.sizes))  # 计算拥挤距离
            globalNDSet = globalNDSet[np.argsort(-dis)[:self.MAXSIZE]]  # 根据拥挤距离选择符合个数限制的解保留在存档中
        # 选择个体保留到下一代
        levels = levels[: population.sizes]  # 得到与population个体对应的levels
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag], globalNDSet

    def reinsertion_update(self, population, archive_acc, NUM, globalNDSet):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目，globalNDSet为全局非支配解存档。
        """

        # 父子两代合并
        population = archive_acc
        globalNDSet = archive_acc  # 将population与全局存档合并
        # 非支配排序分层
        [levels, criLevel] = self.ndSort(globalNDSet.ObjV, None, None, globalNDSet.CV, self.problem.maxormins)
        # 更新全局存档
        globalNDSet = globalNDSet[np.where(levels == 1)[0]]
        if globalNDSet.CV is not None:  # CV不为None说明有设置约束条件
            globalNDSet = globalNDSet[np.where(np.all(globalNDSet.CV <= 0, 1))[0]]  # 排除非可行解
        if globalNDSet.sizes > self.MAXSIZE or population.sizes > self.NP:
            if globalNDSet.sizes > self.MAXSIZE:
                dis = ea.crowdis(globalNDSet.ObjV, np.ones(globalNDSet.sizes))  # 计算拥挤距离
                globalNDSet = globalNDSet[np.argsort(-dis)[:self.MAXSIZE]]  # 根据拥挤距离选择符合个数限制的解保留在存档中
            if population.sizes > NUM:
                dis = ea.crowdis(population.ObjV, np.ones(population.sizes))  # 计算拥挤距离
                population.FitnV = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort').reshape(-1,
                                                                                                              1)  # 计算适应度
                chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
                population = population[chooseFlag]
        return population, globalNDSet

    def reinsertion_surrogate(self, population):

        """
        描述:
            surrogate 训练存档更新
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
        """

        # 父子两代合并
        population, ind = remove_duplication(population)
        if population.sizes > self.MAXSIZE_surrogate:
            # 非支配排序分层
            [levels, criLevel] = self.ndSort(population.ObjV, None, None, population.CV, self.problem.maxormins)
            # 选择个体保留到下一代
            dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
            population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
            chooseFlag = ea.selecting('dup', population.FitnV,
                                      self.MAXSIZE_surrogate)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
            population = population[chooseFlag]
        return population

    def reinsertion_acc(self, pop_acc):

        """
        描述:
            acc 训练存档更新
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
        """

        # 父子两代合并
        globalACCSet = pop_acc
        if globalACCSet.sizes > self.MAXSIZE_acc:
            globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)
            chooseFlag = ea.selecting('dup', globalACCSet.FitnV, self.MAXSIZE_acc)
            globalACCSet = globalACCSet[chooseFlag]
        return globalACCSet

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵
        self.call_aimFunc(population)  # 计算种群的目标函数值
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                         self.problem.maxormins)  # 对NIND个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort').reshape(-1, 1)  # 计算适应度
        globalNDSet = population[np.where(levels == 1)[0]]  # 创建全局存档，该全局存档贯穿进化始终，随着进化不断更新
        if globalNDSet.CV is not None:  # CV不为None说明有设置约束条件
            globalNDSet = globalNDSet[np.where(np.all(globalNDSet.CV <= 0, 1))[0]]  # 排除非可行解
        population.save(self.folder + '/pop' + '%04d' % self.currentGen)
        globalNDSet.save(self.folder + '/nds' + '%04d' % self.currentGen)
        train_arc = population
        train_arc = self.reinsertion_surrogate(train_arc)
        train_arc.save(self.folder + '/train_arc' + '%04d' % self.currentGen)
        pop_acc, ind = remove_duplication(population)
        globalACCSet = self.reinsertion_acc(pop_acc)
        pop_acc.save(self.folder + '/pop_acc' + '%04d' % self.currentGen)
        globalACCSet.save(self.folder + '/glb_acc' + '%04d' % self.currentGen)
        # ===========================开始进化============================
        if self.surrog_flag:
            population, globalNDSet = self.evo_with_surrogate(population, NIND, globalNDSet,
                                                              globalACCSet, train_arc, pop_acc)
        else:
            population, globalNDSet = self.evo_without_surrogate(population, NIND, globalNDSet,
                                                                 globalACCSet, train_arc, pop_acc)
        return self.finishing(population, globalNDSet)  # 调用finishing完成后续工作并返回结果

    def evo_without_surrogate(self, population, NIND, globalNDSet, globalACCSet, train_arc, pop_acc):
        while self.terminated(population) == False:
            rate_gen = self.currentGen / self.MAXGEN
            XOVR = self.XOVR_max  # - self.XOVR_rng * rate_gen
            Pm = self.Pm_min  # + self.Pm_rng * (1 - rate_gen) * (1 - rate_gen)
            if self.Encoding == 'BG':
                self.recOper.XOVR = XOVR
                self.mutOper.Pm_min = Pm
                self.gmOper.Pm_min = Pm
            else:
                self.pmOper.Pm_min = Pm
                self.gmOper.Pm_min = Pm
            if True:  # self.currentGen % self.gap_surrogate:
                if True:  # self.currentGen % 2:
                    population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                    globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                if self.CC_flag:
                    for _ in range(len(self.grps)):
                        population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                        globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                        offspring = self.evo_one_grp(population, NIND, _, pop_acc, globalACCSet, False)
                        all_offspring = offspring if _ == 0 else all_offspring + offspring
                        population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                    offspring, ind = remove_duplication(all_offspring)
                else:
                    offspring = self.evo_all_var_no_mut0(population, NIND, pop_acc, globalACCSet, False)
                    population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                pop_acc = pop_acc + offspring
                globalACCSet = self.reinsertion_acc(pop_acc)
            else:
                self.get_surrogate(train_arc)
                population_bkp, globalNDSet_bkp = population, globalNDSet
                for ig in range(self.ngn_surrogate):
                    if ig % 2:
                        population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                        globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                    if self.CC_flag:
                        for _ in range(len(self.grps)):
                            offspring = self.evo_one_grp(population, NIND, _, pop_acc, globalACCSet, True)
                            population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                    else:
                        offspring = self.evo_all_var_no_mut0(population, NIND, pop_acc, globalACCSet, True)
                        population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in population.Phen])
                offspring = population[not_duplicate]
                offspring, ind = remove_duplication(offspring)
                self.useSurrogate = False
                self.call_aimFunc(offspring)
                pop_acc = pop_acc + offspring
                globalACCSet = self.reinsertion_acc(pop_acc)
                population, globalNDSet = population_bkp, globalNDSet_bkp
                population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
            population.save(self.folder + '/pop' + '%04d' % self.currentGen)
            globalNDSet.save(self.folder + '/nds' + '%04d' % self.currentGen)
            train_arc = self.reinsertion_surrogate(train_arc + offspring)
            train_arc.save(self.folder + '/train_arc' + '%04d' % self.currentGen)
            pop_acc.save(self.folder + '/pop_acc' + '%04d' % self.currentGen)
            globalACCSet.save(self.folder + '/glb_acc' + '%04d' % self.currentGen)
        return population, globalNDSet

    def evo_with_surrogate(self, population, NIND, globalNDSet, globalACCSet, train_arc, pop_acc):
        while self.terminated(population) == False:
            rate_gen = self.currentGen / self.MAXGEN
            XOVR = self.XOVR_max  # - self.XOVR_rng * rate_gen
            Pm = self.Pm_min  # + self.Pm_rng * (1 - rate_gen) * (1 - rate_gen)
            if self.Encoding == 'BG':
                self.recOper.XOVR = XOVR
                self.mutOper.Pm_min = Pm
                self.gmOper.Pm_min = Pm
            else:
                self.pmOper.Pm_min = Pm
                self.gmOper.Pm_min = Pm
            if self.currentGen % self.gap_surrogate:
                population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                if self.CC_flag:
                    self.get_surrogate(train_arc)
                    for _ in range(len(self.grps)):
                        tmp_offspring = self.evo_one_grp(population, NIND, _, pop_acc, globalACCSet, True)
                        offspring = tmp_offspring if _ == 0 else offspring + tmp_offspring
                    offspring, ind = remove_duplication(offspring)
                    offspring.FitnV = offspring.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                    chooseFlag = ea.selecting('dup', offspring.FitnV, (NIND + 1) // 2)
                    rem_ind = [_ for _ in [i for i in range(offspring.sizes)] if _ not in chooseFlag]
                    pop_rem = offspring[rem_ind]
                    if len(pop_acc) > 0:
                        vrt_pred = self.surrogate.predict_variances(pop_rem.Phen)
                        pop_rem.FitnV = vrt_pred.reshape(-1, 1)  # 计算适应度
                        chooseFlag2 = ea.selecting('dup', pop_rem.FitnV, NIND // 2)
                        chooseFlag = np.concatenate((chooseFlag, np.array(rem_ind)[chooseFlag2]))
                    chooseFlag = np.unique(chooseFlag)
                    offspring = offspring[chooseFlag]
                    self.useSurrogate = False
                    self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
                else:
                    offspring = self.evo_all_var_no_mut0(population, NIND, pop_acc, globalACCSet, False)
                population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                pop_acc = pop_acc + offspring
                globalACCSet = self.reinsertion_acc(pop_acc)
            else:
                self.get_surrogate(train_arc)
                population_bkp, globalNDSet_bkp = population, globalNDSet
                for ig in range(self.ngn_surrogate):
                    if self.CC_flag:
                        for _ in range(len(self.grps)):
                            # if ig % 2:
                            population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                            globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                            offspring = self.evo_one_grp(population, NIND, _, pop_acc, globalACCSet, True)
                            population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                    else:
                        # if ig % 2:
                        population.FitnV = population.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                        globalACCSet.FitnV = globalACCSet.ObjV[:, 0].reshape(-1, 1)  # 计算适应度
                        offspring = self.evo_all_var_no_mut0(population, NIND, pop_acc, globalACCSet, True)
                        population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
                not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in population.Phen])
                offspring = population[not_duplicate]
                offspring, ind = remove_duplication(offspring)
                self.useSurrogate = False
                self.call_aimFunc(offspring)
                pop_acc = pop_acc + offspring
                globalACCSet = self.reinsertion_acc(pop_acc)
                population, globalNDSet = population_bkp, globalNDSet_bkp
                population, globalNDSet = self.reinsertion(population, offspring, NIND, globalNDSet)
            population.save(self.folder + '/pop' + '%04d' % self.currentGen)
            globalNDSet.save(self.folder + '/nds' + '%04d' % self.currentGen)
            train_arc = self.reinsertion_surrogate(train_arc + offspring)
            train_arc.save(self.folder + '/train_arc' + '%04d' % self.currentGen)
            pop_acc.save(self.folder + '/pop_acc' + '%04d' % self.currentGen)
            globalACCSet.save(self.folder + '/glb_acc' + '%04d' % self.currentGen)
        return population, globalNDSet

    def evo_one_grp(self, population, NIND, i_grp, pop_acc, globalACCSet, useSurrogate):
        _ = i_grp
        # 选择个体参与进化
        if self.Encoding == 'BG':
            offspring = population[ea.selecting('tour', population.FitnV, NIND)]
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
        else:
            r0 = ea.selecting('ecs', population.FitnV, NIND)  # 得到基向量索引
            offspring = population.copy()  # 存储子代种群
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field, [r0])  # 变异
        if self.CC_type == 'CC_copy':
            offspring.Chrom[:, self.rem_grps[_]] = population.Chrom[:, self.rem_grps[_]]
        elif self.CC_type == 'CC_cross':
            tmp_pop01 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_pop02 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_Chrom = self.crsOper2.do(np.vstack([tmp_pop01.Chrom, tmp_pop02.Chrom]))  # 重组
            tmp_Chrom = self.crsOper1.do(np.vstack([population.Chrom, tmp_Chrom]))  # 重组
            offspring.Chrom[:, self.rem_grps[_]] = tmp_Chrom[:, self.rem_grps[_]]
        elif self.CC_type == 'CC_toBest':
            ind_candid = np.argsort(population.ObjV[:, 0])[:2].tolist()
            tmp_Chrom0 = population[random.choices(ind_candid, k=NIND)].Chrom
            offspring.Chrom[:, self.rem_grps[_]] = tmp_Chrom0[:, self.rem_grps[_]]
        if self.Encoding == 'BG':
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        else:
            offspring.Chrom = self.pmOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def evo_one_grp_no_mut(self, population, NIND, i_grp, pop_acc, globalACCSet, useSurrogate):
        _ = i_grp
        # 选择个体参与进化
        r0 = ea.selecting('ecs', population.FitnV, NIND)  # 得到基向量索引
        offspring = population.copy()  # 存储子代种群
        if self.CC_type == 'CC_copy':
            ind_candid = np.argsort(population.ObjV[:, 0])[:2].tolist()
            tmp_Chrom0 = population[random.choices(ind_candid, k=NIND)].Chrom
            offspring.Chrom[:, self.rem_grps[_]] = tmp_Chrom0[:, self.rem_grps[_]]
        elif self.CC_type == 'CC_cross':
            tmp_pop01 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_pop02 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_Chrom = self.crsOper2.do(np.vstack([tmp_pop01.Chrom, tmp_pop02.Chrom]))  # 重组
            tmp_Chrom = self.crsOper1.do(np.vstack([population.Chrom, tmp_Chrom]))  # 重组
            offspring.Chrom[:, self.rem_grps[_]] = tmp_Chrom[:, self.rem_grps[_]]
        elif self.CC_type == 'CC_toBest':
            tmp_pop = population[ea.selecting('ecs', population.FitnV, NIND)]
            offspring.Chrom[:, self.rem_grps[_]] = tmp_pop.Chrom[:, self.rem_grps[_]]
        offspring.Chrom = self.pmOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def evo_all_var_no_mut(self, population, NIND, pop_acc, globalACCSet, useSurrogate):
        if self.Encoding == 'BG':
            offspring = population[ea.selecting('tour', population.FitnV, NIND)]
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
        else:
            offspring = globalACCSet  # [ea.selecting('ecs', globalACCSet.FitnV, NIND)]
            tmp_pop01 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_pop02 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_Chrom = self.crsOper2.do(np.vstack([tmp_pop01.Chrom, tmp_pop02.Chrom]))  # 重组
            tmp_Chrom = self.crsOper1.do(np.vstack([offspring.Chrom, tmp_Chrom]))  # 重组
            offspring.Chrom = self.pmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        if self.Encoding == 'BG':
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        else:
            offspring.Chrom = self.pmOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def evo_all_var_no_mut0(self, population, NIND, pop_acc, globalACCSet, useSurrogate):
        if self.Encoding == 'BG':
            offspring = population[ea.selecting('tour', population.FitnV, NIND)]
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
        else:
            offspring = population[ea.selecting('ecs', population.FitnV, NIND)]
            tmp_pop01 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_pop02 = population[ea.selecting('tour', population.FitnV, NIND)]
            tmp_Chrom = self.crsOper2.do(np.vstack([tmp_pop01.Chrom, tmp_pop02.Chrom]))  # 重组
            tmp_Chrom = self.crsOper1.do(np.vstack([offspring.Chrom, tmp_Chrom]))  # 重组
            offspring.Chrom = self.pmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        if self.Encoding == 'BG':
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        else:
            offspring.Chrom = self.pmOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def evo_all_var_dp(self, population, NIND, pop_acc, globalACCSet, useSurrogate):
        offspring = globalACCSet.copy()  # 存储子代种群
        tmp_Chrom = self.xodp.do(offspring.Chrom)  # 重组
        tmp_Chrom = self.pmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        offspring.Chrom = self.gmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def evo_all_var_sp(self, population, NIND, pop_acc, globalACCSet, useSurrogate):
        offspring = globalACCSet.copy()  # 存储子代种群
        tmp_Chrom = self.xosp.do(offspring.Chrom)  # 重组
        tmp_Chrom = self.pmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        offspring.Chrom = self.gmOper.do(offspring.Encoding, tmp_Chrom, offspring.Field)  # 变异
        offspring.Phen = offspring.decoding()
        not_duplicate = np.logical_not([any(all(x == a) for a in pop_acc.Phen) for x in offspring.Phen])
        offspring = offspring[not_duplicate]
        offspring, ind = remove_duplication(offspring)
        self.useSurrogate = useSurrogate
        self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
        return offspring

    def get_surrogate(self, train_arc):
        if self.name_surrog in ['KRG_MIXINT']:
            xt = np.concatenate([self.problem.search_space.generate_surrogate_actions_4_solution(_)
                                 for _ in train_arc.Phen])
            self.surrogate = get_surrogate_model(self.name_surrog, xt, train_arc.ObjV[:, 0],
                                                 self.problem.maxN_layers,
                                                 self.problem.Dim, self.problem.xlimits)
        else:
            self.surrogate = get_surrogate_model(self.name_surrog, train_arc.Phen, train_arc.ObjV[:, 0],
                                                 self.problem.maxN_layers,
                                                 self.problem.Dim, self.problem.xlimits)
