import math
import random
from typing import List, Tuple

import torch
from torch import nn, Tensor

from .base import GraphConv_dense
from .base import TemporalConvNet


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






class DCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int):
        super(DCGRUCell, self).__init__()
        self.hidden_size = hidden_size

        self.ru_gate_g_conv = GraphConv_dense(input_size + hidden_size, hidden_size * 2, num_node, n_supports, k_hop)
        self.candidate_g_conv = GraphConv_dense(input_size + hidden_size, hidden_size, num_node, n_supports, k_hop)

    def forward(self, inputs: Tensor, supports: List[Tensor], states) -> Tuple[Tensor, Tensor]:
        """
        :param inputs:  B ,N ,C
        :param supports: B
        :param states:
        :return:
        """
        r_u = torch.sigmoid(self.ru_gate_g_conv(torch.cat([inputs, states], -1), supports))
        r, u = r_u.split(self.hidden_size, -1)
        c = torch.tanh(self.candidate_g_conv(torch.cat([inputs, r * states], -1), supports))
        outputs = new_state = u * states + (1 - u) * c

        return outputs, new_state


class DCRNNEncoder(nn.ModuleList):
    def __init__(self, input_size: int, hidden_size: int, num_node: int, n_supports: int, k_hop: int, n_layers: int):
        super(DCRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.append(DCGRUCell(input_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))

    def forward(self, inputs: Tensor, supports: List[Tensor]) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, input_size]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [n_layers, B, N, hidden_size]
        """
        b, t, n, _ = inputs.shape
        dv, dt = inputs.device, inputs.dtype

        states = list(torch.zeros(len(self), b, n, self.hidden_size, device=dv, dtype=dt))
        inputs = list(inputs.transpose(0, 1))

        for i_layer, cell in enumerate(self):
            for i_t in range(t):
                inputs[i_t], states[i_layer] = cell(inputs[i_t], supports, states[i_layer])
        return torch.stack(states)


class DCRNNDecoder(nn.ModuleList):
    def __init__(self, output_size: int, hidden_size: int, num_node: int,
                 n_supports: int, k_hop: int, n_layers: int, n_preds: int):
        super(DCRNNDecoder, self).__init__()
        self.output_size = output_size
        self.n_preds = n_preds
        self.append(DCGRUCell(output_size, hidden_size, num_node, n_supports, k_hop))
        for _ in range(1, n_layers):
            self.append(DCGRUCell(hidden_size, hidden_size, num_node, n_supports, k_hop))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, supports: List[Tensor], states: Tensor, first_states:Tensor,
                targets: Tensor = None, teacher_force: bool = 0.5) -> Tensor:
        """
        :param supports: list of sparse tensors, each of shape [N, N]
        :param states: tensor, [n_layers, B, N, hidden_size]
        :param targets: None or tensor, [B, T, N, output_size]
        :param teacher_force: random to use targets as decoder inputs
        :return: tensor, [B, T, N, output_size]
        """
        n_layers, b, n, _ = states.shape

        # inputs = torch.zeros(b, n, self.output_size, device=states.device, dtype=states.dtype)
        inputs = first_states
        states = list(states)
        assert len(states) == n_layers

        new_outputs = list()
        for i_t in range(self.n_preds):
            for i_layer in range(n_layers):
                inputs, states[i_layer] = self[i_layer](inputs, supports, states[i_layer])
            inputs = self.out(inputs)
            new_outputs.append(inputs)
            if targets is not None and random.random() < teacher_force:
                inputs = targets[:, i_t]

        return torch.stack(new_outputs, 1)


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

        self.conv = repeat_conv(hidden_size_st, sample_num)



    def forward(self, node_fea, batch_size=None):
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

        return support



class EGC_Stage1(nn.Module):
    def __init__(self,
                 n_pred: int,
                 hidden_size: int,
                 num_nodes: int,
                 hidden_size_st: int,
                 sample_num_st: int,
                 st_conv_method: str,
                 fc_dim: int,
                 n_supports: int,
                 k_hop: int,
                 n_rnn_layers: int,
                 input_dim: int,
                 node_fea: Tensor,
                 output_dim: int,
                 cl_decay_steps: int):
        super(EGC_Stage1, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        self.output_dim = output_dim
        self.encoder = DCRNNEncoder(input_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers)
        self.decoder = DCRNNDecoder(output_dim, hidden_size, num_nodes, n_supports, k_hop, n_rnn_layers, n_pred)
        self.node_fea = node_fea
        self.node_encode = nf_encode(hidden_size_st, sample_num_st, st_conv_method, fc_dim)

    def forward(self, inputs: Tensor, targets: Tensor = None, batch_seen: int = None) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param inputs: [B, n_hist, N, input_dim]
        :param supports: list of tensors, each tensor is with shape [N, N]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, output_dim]
        """

        support = self.node_encode(self.node_fea)
        states = self.encoder(inputs, [support])
        outputs = self.decoder([support], states, inputs[:,-1,:,:self.output_dim],targets, self._compute_sampling_threshold(batch_seen))
        return outputs

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

    def cal(x,y):
        return x*math.log((1/y -1)*x)