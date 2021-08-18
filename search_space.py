import torch
import random
import numpy as np


class MacroSearchSpace(object):
    def __init__(self, search_space=None, dim_type=None, search_space2=None, dim_type2=None, tag_all=False):
        if search_space and dim_type:
            self.search_space = search_space
            self.dim_type = dim_type
            self.search_space2 = search_space2
            self.dim_type2 = dim_type2
        else:
            # Define operators in search space
            self.search_space = {
                'layer_flag': [False, True],
                'attention_type': ["linear", "gen_linear", "cos", "const", "gcn", "gat", "sym-gat"],  # "ggcn",
                'aggregator_type': ["mean", "sum", "pool_mean", "pool_max", "mlp"],  # "gru", "lstm", "none",
                'combinator_type': ["mlp", "identity", "none"],  # "gru", "lstm", "none", "lstm",
                'activate_function': ["linear", "elu", "sigmoid", "tanh",
                                      "relu", "relu6", "softplus", "leaky_relu"],
                'number_of_heads': [1, 2, 4, 6, 8, 16],  #
                'hidden_units': [4, 8, 16, 32, 64, 128],
                # 'attention_dim': [4, 8, 16, 32, 64, 128, 256],  #
                # 'pooling_dim': [4, 8, 16, 32, 64, 128, 256],
                # 'feature_dropout_rate': [0.05, 0.6],  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], #
                # 'attention_dropout_rate': [0.05, 0.6],  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], #
                # 'negative_slope': [0.01, 0.49],  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], #
                # 'residual': [False, True],
                # 'dropout_rate': [0.2, 0.4, 0.6, 0.8],
                'se_layer': [False, True],
            }
            self.dim_type = [
                "bool",
                "discrete",
                "discrete",
                "discrete",
                "discrete",
                "discrete",
                "discrete",
                # "discrete",
                # "discrete",
                # "float",
                # "float",
                # "float",
                # "bool"
                # "discrete",
                "bool"
            ]
            self.search_space2 = {
                # 'weight_decay_rate': [5e-4, 8e-4, 1e-3, 4e-3],  # [5e-4, 8e-4, 1e-3, 4e-3], #
                # 'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],  # [5e-4, 1e-2], #
                'short_cut': [False, True],  # [5e-4, 1e-2], #
            }
            self.dim_type2 = [
                # "discrete",
                # "discrete",
                "bool",
            ]
        if len(self.search_space) != len(self.dim_type):
            raise RuntimeError("Wrong Input: unmatchable input lengths")
        if len(self.search_space2) != len(self.dim_type2):
            raise RuntimeError("Wrong Input2: unmatchable input lengths")
        self.action_names = list(self.search_space.keys())
        self.action_lens = []
        for key in self.action_names:
            self.action_lens.append(len(self.search_space[key]))
        self.state_num = len(self.search_space)
        self.action_names2 = list(self.search_space2.keys())
        self.action_lens2 = []
        for key in self.action_names2:
            self.action_lens2.append(len(self.search_space2[key]))
        self.state_num2 = len(self.search_space2)
        self.tag_all = tag_all

    def get_search_space(self):
        return self.search_space

    def get_search_space2(self):
        return self.search_space2

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers=2):
        action_list = []
        for _ in range(num_of_layers):
            for act in self.action_names:
                action_list.append(act)
        if self.tag_all:
            for act in self.action_names2:
                action_list.append(act)
        return action_list

    def generate_solution(self, num_of_layers=2):
        lb = self.get_lb(num_of_layers)
        ub = self.get_ub(num_of_layers)
        solution = []
        for vl, vu in zip(lb, ub):
            solution.append(random.uniform(vl, vu))
        return solution

    def generate_action_solution(self, num_of_layers=2):
        action_list = self.action_names * num_of_layers
        actions = []
        for i, key in enumerate(action_list):
            k = i % self.state_num
            if self.dim_type[k] == "float":
                actions.append(random.uniform(self.search_space[key][0], self.search_space[key][1]))
            else:
                ind = (int)(self.action_lens[k] * torch.rand(1))
                actions.append(self.search_space[key][ind])
        if self.tag_all:
            for k, key in enumerate(self.action_names2):
                if self.dim_type2[k] == "float":
                    actions.append(random.uniform(self.search_space2[key][0], self.search_space2[key][1]))
                else:
                    ind = (int)(self.action_lens2[k] * torch.rand(1))
                    actions.append(self.search_space2[key][ind])
        return actions

    def generate_action_solution_4_surrogate(self, num_of_layers=2):
        action_list = self.action_names * num_of_layers
        actions = []
        for i, key in enumerate(action_list):
            k = i % self.state_num
            if self.dim_type[k] == "float":
                actions.append(random.uniform(self.search_space[key][0], self.search_space[key][1]))
            else:
                ind = (int)(self.action_lens[k] * torch.rand(1))
                actions.append(ind)
        if self.tag_all:
            for k, key in enumerate(self.action_names2):
                if self.dim_type2[k] == "float":
                    actions.append(random.uniform(self.search_space2[key][0], self.search_space2[key][1]))
                else:
                    ind = (int)(self.action_lens2[k] * torch.rand(1))
                    actions.append(ind)
        return np.array(actions).reshape(1, -1)

    def generate_action_base(self, num_of_layers=2):
        action_list = self.action_names * num_of_layers
        actions = []
        for i, key in enumerate(action_list):
            k = i % self.state_num
            if self.dim_type[k] == "float":
                actions.append(random.uniform(self.search_space[key][0], self.search_space[key][1]))
            elif self.dim_type[k] == "bool":
                actions.append(True)
            else:
                if key in ['number_of_heads', 'hidden_units', 'attention_dim', 'pooling_dim']:
                    ind = (int)(self.action_lens[k] - 1)
                else:
                    ind = (int)(self.action_lens[k] * torch.rand(1))
                actions.append(self.search_space[key][ind])
        if self.tag_all:
            for k, key in enumerate(self.action_names2):
                if self.dim_type2[k] == "float":
                    actions.append(random.uniform(self.search_space2[key][0], self.search_space2[key][1]))
                else:
                    ind = (int)(self.action_lens2[k] * torch.rand(1))
                    actions.append(self.search_space2[key][ind])
        return actions

    def generate_actions_4_solution(self, solution):
        state_length = len(solution)
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        actions = []
        for i in range(state_length):
            val = solution[i]
            k = i % self.state_num
            key = self.action_names[k]
            if self.dim_type[k] == "float":
                actions.append(val)
            else:
                ind = (int)(val)
                actions.append(self.search_space[key][ind])
        if self.tag_all:
            num_layers = state_length // self.state_num
            for k, key in enumerate(self.action_names2):
                i = num_layers * self.state_num + k
                val = solution[i]
                if self.dim_type2[k] == "float":
                    actions.append(val)
                else:
                    ind = (int)(val)
                    actions.append(self.search_space2[key][ind])
        return actions

    def generate_surrogate_actions_4_solution(self, solution):
        state_length = len(solution)
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        actions = []
        for i in range(state_length):
            val = solution[i]
            k = i % self.state_num
            key = self.action_names[k]
            if self.dim_type[k] == "float":
                actions.append(val)
            else:
                ind = (int)(val)
                actions.append(ind)
        if self.tag_all:
            num_layers = state_length // self.state_num
            for k, key in enumerate(self.action_names2):
                i = num_layers * self.state_num + k
                val = solution[i]
                if self.dim_type2[k] == "float":
                    actions.append(val)
                else:
                    ind = (int)(val)
                    actions.append(ind)
        return np.array(actions).reshape(1, -1)

    def get_lb(self, num_of_layers=2):
        action_list = self.action_names * num_of_layers
        lb = []
        for i, key in enumerate(action_list):
            k = i % self.state_num
            if self.dim_type[k] == "float":
                lb.append(self.search_space[key][0])
            else:
                lb.append(0)
        if self.tag_all:
            for k, key in enumerate(self.action_names2):
                if self.dim_type2[k] == "float":
                    lb.append(self.search_space2[key][0])
                else:
                    lb.append(0)
        return lb

    def get_ub(self, num_of_layers=2):
        action_list = self.action_names * num_of_layers
        ub = []
        for i, key in enumerate(action_list):
            k = i % self.state_num
            if self.dim_type[k] == "float":
                ub.append(self.search_space[key][1])
            else:
                ub.append(self.action_lens[k] - 1e-6)
        if self.tag_all:
            for k, key in enumerate(self.action_names2):
                if self.dim_type2[k] == "float":
                    ub.append(self.search_space2[key][1])
                else:
                    ub.append(self.action_lens2[k] - 1e-6)
        return ub

    def get_last_layer_ind(self, actions):
        state_length = len(actions)
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise Exception("wrong action length")
        maxN_layers = state_length // self.state_num
        ind_last_layer = 0
        for _ in range(maxN_layers):
            tmp_offset = _ * self.state_num
            if _ == 0 or actions[tmp_offset] == True:
                ind_last_layer = _
        return ind_last_layer

    def remove_invalid_layer(self, actions):
        state_length = len(actions)
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise Exception("wrong action length")
        maxN_layers = state_length // self.state_num
        action_list = []
        for _ in range(maxN_layers):
            tmp_offset = _ * self.state_num
            if _ == 0 or actions[tmp_offset] == True:
                for i in range(self.state_num):
                    if i > 0:
                        action_list.append(actions[tmp_offset + i])
        return action_list

    def get_dim_num(self, num_of_layers=2):
        nDim = num_of_layers * self.state_num
        if self.tag_all:
            nDim += self.state_num2
        return nDim

    def get_varTypes(self, num_of_layers=2):
        varTypes_list = []
        for _ in range(num_of_layers):
            for ty in self.dim_type:
                varTypes_list.append(ty)
        if self.tag_all:
            for ty in self.dim_type2:
                varTypes_list.append(ty)
        return varTypes_list

    def get_layer_num(self, vals):
        state_length = len(vals)
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        num_layer = state_length // self.state_num
        return num_layer

    def statistics_solution(self, solutions):
        state_length = len(solutions[0])
        if self.tag_all:
            state_length -= self.state_num2
        if state_length % self.state_num != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        statistics1 = self.search_space
        statistics2 = self.search_space2
        for key in self.action_names:
            statistics1[key] = []
        for key in self.action_names2:
            statistics2[key] = []
        for solution in solutions:
            for i in range(state_length):
                val = solution[i]
                k = i % self.state_num
                key = self.action_names[k]
                if self.dim_type[k] == "float":
                    statistics1[key].append(val)
                else:
                    ind = (int)(val)
                    statistics1[key].append(self.search_space[key][ind])
            if self.tag_all:
                num_layers = state_length // self.state_num
                for k, key in enumerate(self.action_names2):
                    i = num_layers * self.state_num + k
                    val = solution[i]
                    if self.dim_type2[k] == "float":
                        statistics2[key].append(val)
                    else:
                        ind = (int)(val)
                        statistics2[key].append(self.search_space2[key][ind])
        for k, key in enumerate(self.action_names):
            if self.dim_type[k] == "float":
                statistics1[key].append(val)
            else:
                ind = (int)(val)
                statistics1[key].append(self.search_space[key][ind])
            statistics1[key] = []
        for k, key in enumerate(self.action_names2):
            if self.dim_type2[k] == "float":
                statistics2[key].append(val)
            else:
                ind = (int)(val)
                statistics2[key].append(self.search_space2[key][ind])
            statistics2[key] = []
        return statistics1, statistics2
