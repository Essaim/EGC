from typing import List

import torch
from torch import nn, Tensor, sparse
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class SelfAttention(nn.Module):
    def __init__(self, n_hidden: int):
        super(SelfAttention, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )

    def forward(self, inputs: Tensor):
        """
        :param inputs: tensor, [B, n_hist, N, n_hidden]
        :return: tensor, [B, N, n_hidden]
        """
        energy = self.projector(inputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        outputs = (inputs * weights.unsqueeze(-1)).sum(1)
        return outputs


class Batch_GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(Batch_GraphConv, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [B, N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape
        x0 = inputs
        x = x0.unsqueeze(dim=0)  # (1, batch_size, num_nodes, input_dim)
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.bmm(support, x0)       #  (batch_size, num_nodes, input_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.bmm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = x.permute([1,2,3,0])  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)



class GraphConv_dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv_dense, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape
        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = x0.unsqueeze(dim=0)  # (1, num_nodes, input_dim * batch_size)
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = support.mm(x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * support.mm(x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)

class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_nodes: int, n_supports: int, max_step: int):
        super(GraphConv, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(input_dim * num_metrics, output_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,
                inputs: Tensor,
                supports: List[Tensor]):
        """
        :param inputs: tensor, [B, N, input_dim]
        :param supports: list of sparse tensors, each of shape [N, N]
        :return: tensor, [B, N, output_dim]
        """
        b, n, input_dim = inputs.shape

        x = inputs
        x0 = x.permute([1, 2, 0]).reshape(n, -1)  # (num_nodes, input_dim * batch_size)
        x = x0.unsqueeze(dim=0)  # (1, num_nodes, input_dim * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = x.reshape(-1, n, input_dim, b).transpose(0, 3)  # (batch_size, num_nodes, input_dim, num_matrices)
        x = x.reshape(b, n, -1)  # (batch_size, num_nodes, input_dim * num_matrices)

        return self.out(x)  # (batch_size, num_nodes, output_dim)


class ChebNet(nn.Module):
    def __init__(self, f_in: int, f_out: int, n_matrices: int):
        super(ChebNet, self).__init__()
        self.out = nn.Linear(n_matrices * f_in, f_out)

    def forward(self, signals: Tensor, supports: Tensor) -> Tensor:
        """
        implement of ChebNet
        :param signals: input signals, Tensor, [*, N, F_in]
        :param supports: pre-calculated Chebychev polynomial filters, Tensor, [N, n_matrices, N]
        :return: Tensor, [B, N, F_out]
        """
        # shape => [B, N, K, F_in]
        tmp = supports.matmul(signals.unsqueeze(-3))
        # shape => [B, N, F_out]
        return self.out(tmp.reshape(tmp.shape[:-2] + (-1,)))


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, inputs):
        """
        :param inputs: tensor, [N, C_{in}, L_{in}]
        :return: tensor, [N, C_{out}, L_{out}]
        """
        outputs = super(CausalConv1d, self).forward(F.pad(inputs, [self.__padding, 0]))
        return outputs[:, :, :outputs.shape[-1] - self.__padding]


class Embedding(nn.Module):
    def __init__(self, d_input, d_embed):
        super().__init__()
        self.input_linear = nn.Linear(1, d_embed, bias=True)
        self.time_linear = nn.Linear(1, d_embed, bias= True)

    def forward(self, x):
        if x.shape[-1] == 2:
            out = self.input_linear(x[...,0].unsqueeze(-1)) + self.time_linear(x[...,1].unsqueeze(-1))
        else:
            out = self.time_linear(x)
        return out






class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        ?????????????????????????????????????????????????????????padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        ???????????????Residual block

        :param n_inputs: int, ???????????????
        :param n_outputs: int, ???????????????
        :param kernel_size: int, ???????????????
        :param stride: int, ??????????????????1
        :param dilation: int, ????????????
        :param padding: int, ????????????
        :param dropout: float, dropout??????
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # ??????conv1????????????size?????????(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # ?????????????????????padding?????????????????????????????????seq_len
        self.relu1 = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.dropout1 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1,  self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        ???????????????

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN?????????paper?????????TCN????????????????????????????????????????????????????????????sequence?????????
        ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        ?????????????????????????????????????????????????????????????????????????????????

        :param num_inputs: int??? ???????????????
        :param num_channels: list????????????hidden_channel????????????[25,25,25,25]?????????4??????????????????hidden_channel??????25
        :param kernel_size: int, ???????????????
        :param dropout: float, drop_out??????
        """
        super(TemporalConvNet, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # ???????????????1???2???4???8??????
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # ?????????????????????????????????
            out_channels = num_channels[i]  # ?????????????????????????????????
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        # self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        ??????x??????????????????RNN?????????RNN???size???(Batch, seq_len, channels)??????(seq_len, Batch, channels)???
        ?????????seq_len??????channels??????????????????????????????????????????????????????Conv1d??????????????????????????????????????????????????????
        ?????????????????????

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        out = list()
        for layer in self.layers:
            x = layer(x)
            out.append(x)
            print(x.shape)

