import math
import random
from typing import List, Tuple

import torch
from torch import nn, Tensor


from .base import Batch_GraphConv

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support

class repeat_conv(nn.Module):
    def __init__(self,dg_hidden_size, sample_num):
        super(repeat_conv, self).__init__()
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.sample_num = sample_num


    def forward(self, y, shape):
        repeat_shape = tuple([1 for _ in range(len(shape))])
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        support = support.repeat((self.sample_num,) + repeat_shape)
        support = gumbel_softmax(support, temperature=0.1, hard=False, device=y.device)
        return support

class soft_conv(nn.Module):
    def __init__(self,dg_hidden_size, sample_num, permute_shape):
        super(soft_conv, self).__init__()
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)
        self.sample_num = sample_num
        self.fc4 = nn.Linear(dg_hidden_size, sample_num)
        self.permute_shape = permute_shape

    def forward(self, y, shape):
        support = self.fc4(torch.relu(self.fc2(y))).reshape(shape+(-1,)).permute(self.permute_shape)
        support = gumbel_softmax(support, temperature=0.1, hard=False, device=y.device)
        return support


class uni_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(uni_conv, self).__init__()
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)
        self.fc3 = nn.Linear(dg_hidden_size, 2)

    def forward(self, y, shape):
        support = self.fc3(torch.relu(self.fc2(y))).reshape(-1, 2)
        support = gumbel_max(support, temperature=0.1, hard=False, device=y.device)
        support = support[:, 0].clone().reshape(shape)
        return support

class DynamicGraphGRU(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int, sample_num: int, conv_method: str):
        super(DynamicGraphGRU, self).__init__()
        self.sample_num = sample_num
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        if conv_method == 'softmax':
            self.conv = soft_conv(dg_hidden_size, sample_num, [3,0,1,2])
        elif conv_method == 'uniform':
            self.conv = uni_conv(dg_hidden_size)
        elif conv_method == 'repeat':
            self.conv = repeat_conv(dg_hidden_size, sample_num)
        elif conv_method == 'normal':
            self.conv = normal_conv(dg_hidden_size)
            self.conv2 = normal_conv(dg_hidden_size)
        else:
            raise ValueError(f'not such conv: {conv_method}')




    def forward(self, inputs: Tensor, states):
        """
        :param inputs: inputs to cal dynamic relations   [B,N,C]
        :param states: recurrent state [B, N,C]
        :return:  graph[B,N,N]       states[B,N,C]
        """
        b,n,c = states.shape
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_


        dy_sent = torch.unsqueeze(torch.relu(new_state), dim=-2).repeat(1, 1, n, 1)
        dy_revi = dy_sent.transpose(1,2)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        states = self.conv(y, (b, n, n))
        mask = self.conv2(y,(b,n,n))
        support = states * torch.sigmoid(mask)




        return support, new_state


class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int):
        super(DCGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.ru_gate_g_conv = Batch_GraphConv(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop)
        self.candidate_g_conv = Batch_GraphConv(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop)

    def forward(self, inputs: Tensor, supports: List[Tensor], states) -> Tuple[Tensor, Tensor]:
        """

        :param inputs:  B ,N ,C
        :param supports:
        :param states:  B,N,C
        :return:
        """

        r_u = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c

        return outputs, new_state




class DCRNNEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,hidden_size_st:int,  hidden_size_dy: int, num_node: int, n_supports: int, k_hop: int, n_layers: int):
        super(DCRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_dy = hidden_size_dy
        self.encoder = nn.ModuleList()
        self.encoder.append(DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.encoder.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))
        self.predictor = nn.Sequential(
            nn.Linear(n_layers * hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.linear_s2d = nn.Linear(hidden_size_st, hidden_size_dy)

    def forward(self, inputs: Tensor, supports: List[Tensor],node_fea_st:Tensor, dygraph_generator: nn.Module) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, input_size]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [n_layers, B, N, hidden_size]
        """
        b, t, n, _ = inputs.shape
        dv, dt = inputs.device, inputs.dtype

        node_fea_dy = self.linear_s2d(node_fea_st).unsqueeze(0).unsqueeze(0)
        states_dy = list(node_fea_dy.repeat(len(self.encoder), b, 1, 1))

        states = list(torch.zeros(len(self.encoder), b, n, self.hidden_size, device=dv, dtype=dt))
        inputs = list(inputs.transpose(0, 1))
        dy_graph_list = list()

        for i_layer, cell in enumerate(self.encoder):
            for i_t in range(t):
                dy_graph, states_dy[i_layer] = dygraph_generator(inputs[i_t][..., :1], states_dy[i_layer])
                dy_graph_list.append(dy_graph)
                inputs[i_t], states[i_layer] = cell(inputs[i_t], [dy_graph], states[i_layer])
        out = torch.stack(states)
        out = out.permute([1,2,3,0]).reshape(b,n,-1)
        return self.predictor(out), torch.stack(dy_graph_list)


def gumbel_softmax_sample(logits, temperature, device, eps=1e-10):
    U = torch.rand(logits.size()).to(device)
    sample = -torch.log(-torch.log(U + eps) + eps)
    y = logits + sample
    return torch.softmax(y / temperature, dim=-1)
def gumbel_max(logits, temperature, hard=False, device=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, device=device, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)

        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)

        y = (y_hard-y_soft).clone().detach().requires_grad_(True) + y_soft
    else:
        y = y_soft
    return y
def gumbel_softmax(logits, temperature,hard=False, device=False, eps=1e-10):

    y_soft = gumbel_softmax_sample(logits, temperature=temperature, device=device, eps=eps)
    y_soft = torch.mean(y_soft, dim=0)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)

        y_hard = torch.zeros(*shape).to(device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)

        y = (y_hard - y_soft).clone().detach().requires_grad_(True) + y_soft
    else:
        y = y_soft
    return y



class nf_encode(nn.Module):
    def __init__(self, hidden_size_st, sample_num, conv_method, fc_dim):
        super(nf_encode, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  # .to(device)
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1)  # .to(device)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size_st)

        self.fc = torch.nn.Linear(fc_dim, hidden_size_st)
        if conv_method == 'softmax':
            self.conv = soft_conv(hidden_size_st, sample_num, [2,0,1])
        elif conv_method == 'uniform':
            self.conv = uni_conv(hidden_size_st)
        elif conv_method == 'repeat':
            self.conv = repeat_conv(hidden_size_st, sample_num)
        elif conv_method == 'normal':
            self.conv = normal_conv(hidden_size_st)
        else:
            raise ValueError(f'not such conv: {conv_method}')


    def forward(self, node_fea, batch_size):
        t, n = node_fea.shape
        x = node_fea.transpose(1, 0).view(n, 1, -1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.bn3(x)

        x_sent = x.unsqueeze(1).repeat(1, n, 1)
        x_revi = x_sent.transpose(0, 1)
        y = torch.cat([x_sent, x_revi], dim=-1)
        support = self.conv(y, (n, n))

        d = support.sum(1)
        d_inv = torch.pow(d, -1).flatten()
        d_mat_inv = torch.diag(d_inv)
        support = d_mat_inv.mm(support)



        batch_support = support.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, n, n)
        return batch_support, x


class regular_cal(nn.Module):
    def __init__(self, hidden_size, max_step = 3):
        super(regular_cal, self).__init__()

        self.conv = Batch_GraphConv(input_dim=1 , output_dim =hidden_size, num_nodes = 111, n_supports = 2, max_step =max_step)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, graph_dy, graph_st=None, inputs=None):
        """
        cal the regular term
        :param graph_dy: (t, b, n, n)
        :param graph_st: (b, n, n)
        :param inputs: (t, b, n, 1)
        :return: regular term (t-1, b, n, 1)
        """


        t, b, n, n_ = graph_dy.shape
        conv_out = list()
        for g, i in zip(graph_dy, inputs):
            states = self.conv(i, [graph_st, g])
            conv_out.append(self.out(states))  # (b, n, 1)
        conv_out.pop()
        conv_out = torch.stack(conv_out).reshape(t - 1, b, n, -1)  # (t-1,b, n, c, dy/st)

        return conv_out, inputs[1:, ...]


class EGC_Stage2_Single(nn.Module):
    def __init__(self,
                 n_pred: int,
                 hidden_size: int,
                 hidden_size_dy: int,
                 hidden_size_st: int,
                 hidden_size_simple: int,
                 num_nodes: int,
                 sample_num_dy: int,
                 sample_num_st: int,
                 st_conv_method: str,
                 dy_conv_method: str,
                 n_supports: int,
                 k_hop: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 output_dim: int,
                 fc_dim: int,
                 node_fea: Tensor,
                 cl_decay_steps: int):
        super(EGC_Stage2_Single, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        self.node_encode = nf_encode(hidden_size_st, sample_num_st, st_conv_method, fc_dim)
        self.dygraph_generator = DynamicGraphGRU(1, hidden_size_dy, sample_num_dy, dy_conv_method)
        self.encoder = DCRNNEncoder( input_dim, hidden_size, hidden_size_st,hidden_size_dy, num_nodes,
                                    n_supports, k_hop,
                                    n_rnn_layers)

        self.node_fea = node_fea
        self.regular_cal = regular_cal(hidden_size_simple, max_step = k_hop)
        self.output_dim = output_dim


    def forward(self, inputs: Tensor, i_targets: Tensor = None, targets: Tensor = None,
                batch_seen: int = None) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, output_dim]
        """
        support,x = self.node_encode(self.node_fea, inputs.shape[0])

        outputs, dygraph = self.encoder(inputs, support,x, self.dygraph_generator)

        regular_term, regular_label = self.regular_cal(graph_dy = dygraph,
                                                       graph_st = support,
                                                        inputs = inputs[..., :1].transpose(0,1)
                                            )
        self.support = support
        return outputs, regular_term, regular_label

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))
