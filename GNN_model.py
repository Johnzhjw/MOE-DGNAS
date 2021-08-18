import torch as th
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.nn.pytorch.utils import Identity
from dgl.base import DGLError

from collections import OrderedDict


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        num_hidden = channel // reduction if channel // reduction > 0 else 1
        self.fc = nn.Sequential(
            nn.Linear(channel, num_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc[0].weight, gain=gain)
        nn.init.xavier_normal_(self.fc[2].weight, gain=gain)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def get_head_index_list(tar, src):
    if tar > src:
        raise Exception("tar should not be larger than src.")
    lst = [_ for _ in range(0, src, src//tar)]
    return lst


class GNNmodel(nn.Module):
    def __init__(self,
                 in_dim,
                 num_classes,
                 actions,
                 search_space,
                 flag_base=False):
        super(GNNmodel, self).__init__()
        self.flag_se = False
        self.num_layers = 0
        state_num = search_space.state_num
        self.short_cut = actions[-1] if search_space.tag_all else False
        self.maxN_layers = search_space.get_layer_num(actions)
        self.gnn_layers = nn.ModuleList()
        num_in_pre = 0
        self.ind_last_layer = search_space.get_last_layer_ind(actions)
        n_out_all = 0
        if flag_base:
            self.ind_last_layer = self.maxN_layers - 1
        for _ in range(self.maxN_layers):
            tmp_offset = _ * state_num
            if _ != 0 and actions[tmp_offset] == False:
                continue
            attention_type = actions[tmp_offset + 1]
            aggregator_type = actions[tmp_offset + 2]
            combinator_type = actions[tmp_offset + 3]
            activation = actions[tmp_offset + 4]
            heads = actions[tmp_offset + 5]
            num_hidden = actions[tmp_offset + 6]
            feat_drop = 0.6  # actions[tmp_offset + 7]
            attn_drop = 0.6  # actions[tmp_offset + 7]
            negative_slope = 0.2  # actions[tmp_offset + 9]
            # residual = actions[tmp_offset + 7]
            se_layer = actions[tmp_offset + 7]
            if activation == "linear":
                fn_act = Identity()
            elif activation == "elu":
                fn_act = F.elu
            elif activation == "sigmoid":
                fn_act = th.sigmoid
            elif activation == "tanh":
                fn_act = th.tanh
            elif activation == "relu":
                fn_act = F.relu
            elif activation == "relu6":
                fn_act = F.relu6
            elif activation == "softplus":
                fn_act = F.softplus
            elif activation == "leaky_relu":
                fn_act = F.leaky_relu
            else:
                raise Exception("wrong activate function")
            #
            if _ == 0:
                num_in = in_dim
                residual = False
            else:
                num_in = num_in_pre
            if _ == self.ind_last_layer and not self.short_cut:
                num_out = num_classes
                fn_act = None
            else:
                num_out = num_hidden
            if flag_base:
                self.gnn_layers.append(GNNlayerBase(
                    in_feats=num_in, out_feats=num_out, aggregator_type=aggregator_type, attention_type=attention_type,
                    combinator_type=combinator_type, num_heads=heads,
                    feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                    activation=fn_act, flag_se=True))
            else:
                self.gnn_layers.append(GNNlayer(
                    in_feats=num_in, out_feats=num_out, aggregator_type=aggregator_type, attention_type=attention_type,
                    combinator_type=combinator_type, num_heads=heads,
                    feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
                    activation=fn_act, flag_se=se_layer))
            num_in_pre = num_out * heads
            self.num_layers += 1
            n_out_all += num_in_pre
        if self.short_cut:
            self.final_lin = nn.Linear(n_out_all, num_classes, bias=False)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(self.final_lin.weight, gain=gain)

    def forward(self, g):
        h = g.ndata['feat']
        if self.short_cut:
            for l in range(self.num_layers):
                h = self.gnn_layers[l](g, h).flatten(1)
                h_all = h if l == 0 else th.cat((h_all, h), dim=1)
            logits = self.final_lin(h_all)
        else:  #
            for l in range(self.num_layers - 1):
                h = self.gnn_layers[l](g, h).flatten(1)
            # output projection
            logits = self.gnn_layers[-1](g, h).mean(1)
        return logits

    def copy_para(self, modelBase):
        stateBase = modelBase.state_dict()
        stateMine = self.state_dict()
        #
        for l in range(0, self.ind_last_layer + 1):
            sub_str1 = 'gnn_layers.' + str(l)
            ll = l
            sub_str2 = 'gnn_layers.' + str(ll)
            nh_s = self.gnn_layers[l]._num_heads
            nh_l = modelBase.gnn_layers[ll]._num_heads
            nh_st = nh_l // nh_s
            nfo_s = self.gnn_layers[l]._out_feats
            nfo_l = modelBase.gnn_layers[ll]._out_feats
            nfo_st = nfo_l // nfo_s
            nfi_s = self.gnn_layers[l]._in_src_feats
            nfi_l = modelBase.gnn_layers[ll]._in_src_feats
            nfi_st = nfi_l // nfi_s
            for key in stateMine:
                if key.find(sub_str1) == 0:
                    key2 = key.replace(sub_str1, sub_str2)
                    shp = stateMine[key].shape
                    dim = len(shp)
                    if dim == 1:
                        if key.split('.')[2] == 'bias':
                            '''
                            print(stateMine[key].shape)
                            print(nh_s, nfo_s)
                            print(l, self.ind_last_layer)
                            print(stateBase[key2].shape)
                            print(nh_l, nfo_l)
                            print(ll)
                            '''
                            stateMine[key] = \
                                stateBase[key2].view(nh_l, nfo_l)[0:nh_l:nh_st, 0:nfo_l:nfo_st].\
                                    contiguous().flatten()[:shp[0]]
                        elif key.split('.')[2] in ['agg_pool', 'comb_mlp'] and \
                                key.split('.')[3] == 'dec' and \
                                key.split('.')[4] == 'bias':
                            stateMine[key] = \
                                stateBase[key2].view(nh_l, nfo_l)[0:nh_l:nh_st, 0:nfo_l:nfo_st].\
                                    contiguous().flatten()[:shp[0]]
                        else:
                            stateMine[key] = stateBase[key2][:shp[0]]
                    elif dim == 2:
                        if key.split('.')[2] in ['comb_mlp'] and \
                                key.split('.')[3] == 'enc' and \
                                key.split('.')[4] == 'weight':
                            stateMine[key] = \
                                stateBase[key2][:, 0:nfi_l:nfi_st].contiguous()[:shp[0], :shp[1]]
                        if key.split('.')[2] in ['agg_pool'] and \
                                key.split('.')[3] == 'enc' and \
                                key.split('.')[4] == 'weight':
                            stateMine[key] = \
                                stateBase[key2].view(-1, nh_l, nfo_l)[:, 0:nh_l:nh_st, 0:nfo_l:nfo_st].\
                                    contiguous().flatten(1)[:shp[0], :shp[1]]
                        elif key.split('.')[2] in ['agg_pool', 'comb_mlp'] and \
                                key.split('.')[3] == 'dec' and \
                                key.split('.')[4] == 'weight':
                            stateMine[key] = \
                                stateBase[key2].view(nh_l, nfo_l, -1)[0:nh_l:nh_st, 0:nfo_l:nfo_st, :].\
                                    contiguous().flatten(0, 1)[:shp[0], :shp[1]]
                        elif key.split('.')[2] in ['fc', 'fc_src', 'fc_dst', 'res_fc', 'comb_idn'] and \
                                key.split('.')[3] == 'weight':
                            stateMine[key] = \
                                stateBase[key2].view(nh_l, nfo_l, -1)[0:nh_l:nh_st, 0:nfo_l:nfo_st, 0:nfi_l:nfi_st].\
                                    contiguous().flatten(0, 1)[:shp[0], :shp[1]]
                        elif key.split('.')[2] == 'se' and \
                                key.split('.')[3] == 'fc' and \
                                key.split('.')[5] == 'weight':
                            nhi_s = self.gnn_layers[l].se.fc[0].weight.shape[0]
                            nhi_l = modelBase.gnn_layers[ll].se.fc[0].weight.shape[0]
                            nhi_st = nhi_l // nhi_s
                            if key.split('.')[4] == '0':
                                stateMine[key] = stateBase[key2][0:nhi_l:nhi_st, 0:nh_l:nh_st][:shp[0], :shp[1]]
                            if key.split('.')[4] == '2':
                                stateMine[key] = stateBase[key2][0:nh_l:nh_st, 0:nhi_l:nhi_st][:shp[0], :shp[1]]
                        else:
                            stateMine[key] = stateBase[key2][:shp[0], :shp[1]]
                    elif dim == 3:
                        if key.split('.')[2] in ['attn_l', 'attn_r']:
                            stateMine[key] = stateBase[key2][:, 0:nh_l:nh_st, 0:nfo_l:nfo_st][:shp[0], :shp[1], :shp[2]]
                        elif key.split('.')[2] in ['attn_w', 'attn_p_a', 'attn_p_b', 'attn_p_lmd']:
                            stateMine[key] = stateBase[key2][:, 0:nh_l:nh_st, :][:shp[0], :shp[1], :shp[2]]
                        else:
                            stateMine[key] = stateBase[key2][:shp[0], :shp[1], :shp[2]]
        self.load_state_dict(stateMine)

    def save_para(self, modelBetter):
        stateBetter = modelBetter.state_dict()
        stateBase = self.state_dict()
        #
        for l in range(0, modelBetter.ind_last_layer + 1):
            sub_str1 = 'gnn_layers.' + str(l)
            ll = l
            sub_str2 = 'gnn_layers.' + str(ll)
            nh_s = modelBetter.gnn_layers[l]._num_heads
            nh_l = self.gnn_layers[ll]._num_heads
            nh_st = nh_l // nh_s
            nfo_s = modelBetter.gnn_layers[l]._out_feats
            nfo_l = self.gnn_layers[ll]._out_feats
            nfo_st = nfo_l // nfo_s
            nfi_s = modelBetter.gnn_layers[l]._in_src_feats
            nfi_l = self.gnn_layers[ll]._in_src_feats
            nfi_st = nfi_l // nfi_s
            for key in stateBetter:
                if key.find(sub_str1) == 0:
                    key2 = key.replace(sub_str1, sub_str2)
                    shp = stateBetter[key].shape
                    dim = len(shp)
                    if dim == 1:
                        if key.split('.')[2] == 'bias':
                            stateBase[key2].view(nh_l, nfo_l)[0:nh_l:nh_st, 0:nfo_l:nfo_st][:nh_s, :nfo_s] = \
                                stateBetter[key].view(nh_s, nfo_s)
                        elif key.split('.')[2] in ['agg_pool', 'comb_mlp'] and \
                                key.split('.')[3] == 'dec' and \
                                key.split('.')[4] == 'bias':
                            stateBase[key2].view(nh_l, nfo_l)[0:nh_l:nh_st, 0:nfo_l:nfo_st][:nh_s, :nfo_s] = \
                                stateBetter[key].view(nh_s, nfo_s)
                        else:
                            stateBase[key2][:shp[0]] = stateBetter[key]
                    elif dim == 2:
                        if key.split('.')[2] in ['comb_mlp'] and \
                                key.split('.')[3] == 'enc' and \
                                key.split('.')[4] == 'weight':
                            stateBase[key2][:, 0:nfi_l:nfi_st][:, :nfi_s] = stateBetter[key]
                        if key.split('.')[2] in ['agg_pool'] and \
                                key.split('.')[3] == 'enc' and \
                                key.split('.')[4] == 'weight':
                            stateBase[key2].view(-1, nh_l, nfo_l)[:, 0:nh_l:nh_st, 0:nfo_l:nfo_st][:, :nh_s, :nfo_s] = \
                                stateBetter[key].view(-1, nh_s, nfo_s)
                        elif key.split('.')[2] in ['agg_pool', 'comb_mlp'] and \
                                key.split('.')[3] == 'dec' and \
                                key.split('.')[4] == 'weight':
                            stateBase[key2].view(nh_l, nfo_l, -1)[0:nh_l:nh_st, 0:nfo_l:nfo_st, :][:nh_s, :nfo_s, :] = \
                                stateBetter[key].view(nh_s, nfo_s, -1)
                        elif key.split('.')[2] in ['fc', 'fc_src', 'fc_dst', 'res_fc', 'comb_idn'] and \
                                key.split('.')[3] == 'weight':
                            stateBase[key2].view(nh_l, nfo_l, -1)[0:nh_l:nh_st, 0:nfo_l:nfo_st, 0:nfi_l:nfi_st][:nh_s, :nfo_s, :nfi_s] = \
                                stateBetter[key].view(nh_s, nfo_s, -1)
                        elif key.split('.')[2] == 'se' and \
                             key.split('.')[3] == 'fc' and \
                             key.split('.')[5] == 'weight':
                            nhi_s = modelBetter.gnn_layers[l].se.fc[0].weight.shape[0]
                            nhi_l = self.gnn_layers[ll].se.fc[0].weight.shape[0]
                            nhi_st = nhi_l // nhi_s
                            if key.split('.')[4] == '0':
                                stateBase[key2][0:nhi_l:nhi_st, 0:nh_l:nh_st][:shp[0], :shp[1]] = stateBetter[key]
                            if key.split('.')[4] == '2':
                                stateBase[key2][0:nh_l:nh_st, 0:nhi_l:nhi_st][:shp[0], :shp[1]] = stateBetter[key]
                        else:
                            stateBase[key2][:shp[0], :shp[1]] = stateBetter[key]
                    elif dim == 3:
                        if key.split('.')[2] in ['attn_l', 'attn_r']:
                            stateBase[key2][:, 0:nh_l:nh_st, 0:nfo_l:nfo_st][:shp[0], :shp[1], :shp[2]] = stateBetter[key]
                        elif key.split('.')[2] in ['attn_w', 'attn_p_a', 'attn_p_b', 'attn_p_lmd']:
                            stateBase[key2][:, 0:nh_l:nh_st, :][:shp[0], :shp[1], :shp[2]] = stateBetter[key]
                        else:
                            stateBase[key2][:shp[0], :shp[1], :shp[2]] = stateBetter[key]
        self.load_state_dict(stateBase)


class GNNlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 attention_type,
                 combinator_type,
                 num_heads,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 norm=None,
                 flag_se=True,
                 batch_normal=False):
        super(GNNlayer, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._attn_type = attention_type
        self._comb_type = combinator_type
        self._num_heads = num_heads
        self.norm = norm
        self.flag_SE = flag_se
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = activation
        self.pool_dim = 128
        # aggregator type: mean/sum/pool
        if aggregator_type in ['pool_max', 'pool_mean', 'mlp']:
            self.agg_pool = nn.Sequential(OrderedDict([
                ('enc', nn.Linear(out_feats * num_heads, self.pool_dim)),
                ('act1', nn.LeakyReLU()),
                ('dec', nn.Linear(self.pool_dim, out_feats * num_heads)),
                ('act2', nn.LeakyReLU()),
            ]))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        self._allow_zero_in_degree = allow_zero_in_degree
        self.batch_normal = batch_normal
        if isinstance(in_feats, tuple):
            if self.batch_normal:
                self.op_bn_src = nn.BatchNorm1d(self._in_src_feats, momentum=0.5)
                self.op_bn_dst = nn.BatchNorm1d(self._in_dst_feats, momentum=0.5)
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            if self.batch_normal:
                self.op_bn = nn.BatchNorm1d(self._in_src_feats, momentum=0.5)
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        if self._aggre_type != "none":
            if attention_type == 'const':
                pass
            elif attention_type == 'gcn':
                pass
            elif attention_type == 'gat':
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            elif attention_type == 'sym-gat':
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            elif attention_type == 'cos':
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            elif attention_type == 'linear':
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            elif attention_type == 'gen_linear':
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_w = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            elif attention_type == 'ggcn':
                self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_p_a = nn.Parameter(th.FloatTensor(size=(1, num_heads, 1)))
                self.attn_p_b = nn.Parameter(th.FloatTensor(size=(1, num_heads, 3)))
                self.attn_p_lmd = nn.Parameter(th.FloatTensor(size=(1, num_heads, 2)))
            else:
                raise Exception("wrong attention type")
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if self._comb_type == 'mlp':
            self.comb_mlp = nn.Sequential(OrderedDict([
                ('enc', nn.Linear(self._in_dst_feats, self.pool_dim)),
                ('act1', nn.LeakyReLU()),
                ('dec', nn.Linear(self.pool_dim, out_feats * num_heads)),
                ('act2', nn.LeakyReLU()),
            ]))
        elif self._comb_type == 'identity':
            if self._in_dst_feats != out_feats:
                self.comb_idn = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.comb_idn = Identity()
        elif self._comb_type == 'none':
            pass
        else:
            raise Exception("wrong combinator type")
        if self.flag_SE and self._num_heads > 1:
            self.se = SELayer(num_heads)
        else:
            self.register_buffer('se', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type in ['pool_max', 'pool_mean', 'mlp']:
            nn.init.xavier_uniform_(self.agg_pool.enc.weight, gain=gain)
            nn.init.constant_(self.agg_pool.enc.bias, 0)
            nn.init.xavier_uniform_(self.agg_pool.dec.weight, gain=gain)
            nn.init.constant_(self.agg_pool.dec.bias, 0)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        if self._aggre_type != "none":
            if self._attn_type == 'const':
                pass
            elif self._attn_type == 'gcn':
                pass
            elif self._attn_type == 'gat':
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
            elif self._attn_type == 'sym-gat':
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
            elif self._attn_type == 'cos':
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
            elif self._attn_type == 'linear':
                nn.init.xavier_normal_(self.attn_r, gain=gain)
            elif self._attn_type == 'gen_linear':
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
                nn.init.xavier_normal_(self.attn_w, gain=gain)
            elif self._attn_type == 'ggcn':
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
                nn.init.xavier_normal_(self.attn_p_a, gain=gain)
                nn.init.xavier_normal_(self.attn_p_b, gain=gain)
                nn.init.xavier_normal_(self.attn_p_lmd, gain=gain)
        if self._comb_type == 'mlp':
            nn.init.xavier_normal_(self.comb_mlp.enc.weight, gain=gain)
            nn.init.constant_(self.comb_mlp.enc.bias, 0)
            nn.init.xavier_normal_(self.comb_mlp.dec.weight, gain=gain)
            nn.init.constant_(self.comb_mlp.dec.bias, 0)
        elif self._comb_type == 'identity':
            if self._in_dst_feats != self._out_feats:
                nn.init.xavier_normal_(self.comb_idn.weight, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.se is not None:
            self.se.reset_parameters()

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                if self.batch_normal:
                    feat_ori_src = self.op_bn_src(feat[0])
                    feat_ori_dst = self.op_bn_dst(feat[1])
                else:
                    feat_ori_src = feat[0]
                    feat_ori_dst = feat[1]
                feat_src = self.feat_drop(feat_ori_src)
                feat_dst = self.feat_drop(feat_ori_dst)
                if not hasattr(self, 'fc_src'):
                    feat_src_fc = self.fc(feat_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst_fc = self.fc(feat_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src_fc = self.fc_src(feat_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst_fc = self.fc_dst(feat_dst).view(-1, self._num_heads, self._out_feats)
            else:
                if self.batch_normal:
                    feat_ori = self.op_bn(feat)
                else:
                    feat_ori = feat
                feat_src = feat_dst = self.feat_drop(feat_ori)
                feat_src_fc = feat_dst_fc = self.fc(feat_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    feat_dst_fc = feat_src_fc[:graph.number_of_dst_nodes()]

            if self._aggre_type != "none":
                if self._attn_type == 'const':
                    graph.srcdata.update({'ft': feat_src_fc})
                elif self._attn_type == 'gcn':
                    degs = graph.in_degrees().to(feat_dst_fc)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': degs})
                    graph.dstdata.update({'er': degs})
                    graph.apply_edges(fn.u_mul_v('el', 'er', 'e'))
                    e = th.rsqrt(graph.edata.pop('e'))
                elif self._attn_type == 'gat':
                    el = (feat_src_fc * self.attn_l).sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                    e = self.leaky_relu(graph.edata.pop('e'))
                elif self._attn_type == 'sym-gat':
                    el = (feat_src_fc * self.attn_l).sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                    e1 = graph.edata.pop('e')
                    # opposite
                    # if isinstance(feat, tuple):
                    #    if not hasattr(self, 'fc_src'):
                    #        feat_src_fc_oppo = feat_src_fc
                    #        feat_dst_fc_oppo = feat_dst_fc
                    #    else:
                    #        feat_src_fc_oppo = self.fc_dst(feat_src).view(-1, self._num_heads, self._out_feats)
                    #        feat_dst_fc_oppo = self.fc_src(feat_dst).view(-1, self._num_heads, self._out_feats)
                    # else:
                    #    feat_src_fc_oppo = feat_dst_fc_oppo = feat_src_fc
                    #    if graph.is_block:
                    #        feat_dst_fc_oppo = feat_src_fc_oppo[:graph.number_of_dst_nodes()]
                    # el_oppo = (feat_src_fc_oppo * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    # er_oppo = (feat_dst_fc_oppo * self.attn_l).sum(dim=-1).unsqueeze(-1)
                    el_oppo = er
                    er_oppo = el
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el_oppo})  # 'ft' how to assign, what
                    graph.dstdata.update({'er': er_oppo})
                    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                    e2 = graph.edata.pop('e')
                    e = self.leaky_relu(e1 + e2)
                elif self._attn_type == 'cos':
                    el = (feat_src_fc * self.attn_l)  # .sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r)  # .sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_mul_v('el', 'er', 'e'))
                    e = self.leaky_relu(graph.edata.pop('e').sum(dim=-1).unsqueeze(-1))
                elif self._attn_type == 'linear':
                    el = th.zeros_like(feat_src_fc).sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                    e = th.tanh(graph.edata.pop('e'))
                elif self._attn_type == 'gen_linear':
                    el = (feat_src_fc * self.attn_l)  # .sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r)  # .sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                    e = (th.tanh(graph.edata.pop('e')) * self.attn_w).sum(dim=-1).unsqueeze(-1)
                elif self._attn_type == 'ggcn':
                    el = (feat_src_fc * self.attn_l)  # .sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst_fc * self.attn_r)  # .sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'ft': feat_src_fc, 'el': el})
                    graph.dstdata.update({'er': er})
                    graph.apply_edges(fn.u_mul_v('el', 'er', 'e'))
                    e1 = graph.edata.pop('e').sum(dim=-1).unsqueeze(-1)
                    attn_p_b_n = th.softmax(self.attn_p_b, 2)
                    ind_p = e1 >= 0
                    ind_n = e1 < 0
                    e2 = th.empty_like(e1)
                    e2[ind_p] = (e1 * attn_p_b_n[:, :, 1:2])[ind_p]
                    e2[ind_n] = (e1 * attn_p_b_n[:, :, 2:3])[ind_n]
                    degs = graph.in_degrees().to(feat_dst_fc) + 1
                    graph.srcdata.update({'dl': degs})
                    graph.dstdata.update({'dr': degs})
                    graph.apply_edges(fn.u_div_v('dl', 'dr', 'd'))
                    d = F.softplus(
                        self.attn_p_lmd[:, :, 0:1] * (th.rsqrt(graph.edata.pop('d')) - 1).unsqueeze(-1).unsqueeze(
                            -1) + self.attn_p_lmd[:, :, 1:2])
                    e = e2 * d

            # compute softmax
            if self._aggre_type != "none":
                if self._attn_type in ['const']:
                    msg_fn = fn.copy_src('ft', 'm')
                elif self._attn_type in ['gcn']:
                    graph.edata['a'] = e
                    msg_fn = fn.u_mul_e('ft', 'a', 'm')
                elif self._attn_type in ['gat', 'sym-gat', 'cos', 'linear', 'gen_linear']:
                    graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                    msg_fn = fn.u_mul_e('ft', 'a', 'm')
                elif self._attn_type in ['ggcn']:
                    graph.edata['a'] = self.attn_drop(e)
                    msg_fn = fn.u_mul_e('ft', 'a', 'm')

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Message Passing
            if self._aggre_type == 'mean':
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'sum':
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'pool_max':
                graph.srcdata['ft'] = self.agg_pool(feat_src_fc.flatten(-2)).view(-1, self._num_heads, self._out_feats)
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'pool_mean':
                graph.srcdata['ft'] = self.agg_pool(feat_src_fc.flatten(-2)).view(-1, self._num_heads, self._out_feats)
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'mlp':
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                h_neigh = self.agg_pool(h_neigh.flatten(-2)).view(-1, self._num_heads, self._out_feats)
            elif self._aggre_type == 'none':
                h_neigh = feat_src_fc
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            if self._comb_type == 'mlp':
                rst = h_neigh + self.comb_mlp(feat_dst).view(-1, self._num_heads, self._out_feats)
            elif self._comb_type == 'identity':
                rst = h_neigh + self.comb_idn(feat_dst).view(feat_dst.shape[0], -1, self._out_feats)
            else:
                rst = h_neigh

            if self._attn_type == 'ggcn' and self._aggre_type != 'none':
                rst = F.softplus(self.attn_p_a[:, :, 0:1]) * (rst + graph.srcdata['ft'] * attn_p_b_n[:, :, 0:1])

            # bias term
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            if self.se is not None:
                rst = self.se(rst)

            if get_attention and \
                    self._attn_type in ['gcn', 'gat', 'sym-gat', 'cos', 'linear', 'gen_linear', 'ggcn'] and \
                    self._aggre_type != 'none':
                return rst, graph.edata['a']
            else:
                return rst


"""
"""


class GNNlayerBase(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 attention_type,
                 combinator_type,
                 num_heads,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 negative_slope=0.2,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 flag_se=True,
                 batch_normal=True):
        super(GNNlayerBase, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._attn_type = attention_type
        self._comb_type = combinator_type
        self._num_heads = num_heads
        self.norm = norm
        self.flag_SE = flag_se
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = activation
        self.pool_dim = 128
        # aggregator type: mean/sum/pool
        self.agg_pool = nn.Sequential(OrderedDict([
            ('enc', nn.Linear(out_feats * num_heads, self.pool_dim)),
            ('act1', nn.LeakyReLU()),
            ('dec', nn.Linear(self.pool_dim, out_feats * num_heads)),
            ('act2', nn.LeakyReLU()),
        ]))
        self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        self.batch_normal = batch_normal
        if isinstance(in_feats, tuple):
            if self.batch_normal:
                self.op_bn_src = nn.BatchNorm1d(self._in_src_feats, momentum=0.5)
                self.op_bn_dst = nn.BatchNorm1d(self._in_dst_feats, momentum=0.5)
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            if self.batch_normal:
                self.op_bn = nn.BatchNorm1d(self._in_src_feats, momentum=0.5)
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_w = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_p_a = nn.Parameter(th.FloatTensor(size=(1, num_heads, 1)))
        self.attn_p_b = nn.Parameter(th.FloatTensor(size=(1, num_heads, 3)))
        self.attn_p_lmd = nn.Parameter(th.FloatTensor(size=(1, num_heads, 2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.comb_mlp = nn.Sequential(OrderedDict([
            ('enc', nn.Linear(self._in_dst_feats, self.pool_dim)),
            ('act1', nn.LeakyReLU()),
            ('dec', nn.Linear(self.pool_dim, out_feats * num_heads)),
            ('act2', nn.LeakyReLU()),
        ]))
        self.comb_idn = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        self.se = SELayer(num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.agg_pool.enc.weight, gain=gain)
        nn.init.constant_(self.agg_pool.enc.bias, 0)
        nn.init.xavier_uniform_(self.agg_pool.dec.weight, gain=gain)
        nn.init.constant_(self.agg_pool.dec.bias, 0)
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_w, gain=gain)
        nn.init.xavier_normal_(self.attn_p_a, gain=gain)
        nn.init.xavier_normal_(self.attn_p_b, gain=gain)
        nn.init.xavier_normal_(self.attn_p_lmd, gain=gain)
        nn.init.xavier_normal_(self.comb_mlp.enc.weight, gain=gain)
        nn.init.constant_(self.comb_mlp.enc.bias, 0)
        nn.init.xavier_normal_(self.comb_mlp.dec.weight, gain=gain)
        nn.init.constant_(self.comb_mlp.dec.bias, 0)
        nn.init.xavier_normal_(self.comb_idn.enc.weight, gain=gain)
        nn.init.constant_(self.bias, 0)
        self.se.reset_parameters()

    def forward(self, graph, feat, get_attention=False):
        pass
