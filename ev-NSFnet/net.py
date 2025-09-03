# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Zhicheng Wang, Hui Xiang
# Created: 08.03.2023
import torch
import torch.nn as nn
from collections import OrderedDict

# neural network
class FCNet(torch.nn.Module):
    def __init__(self, num_ins=3,
                 num_outs=3,
                 num_layers=10,
                 hidden_size=50,
                 hidden_sizes=None,
                 activation=torch.nn.Tanh):
        super(FCNet, self).__init__()

        # 支援兩種配置模式：
        # 1. hidden_size: 所有隱藏層使用相同神經元數量
        # 2. hidden_sizes: 每層神經元數量列表
        if hidden_sizes is not None:
            if len(hidden_sizes) != num_layers:
                raise ValueError(f"hidden_sizes長度({len(hidden_sizes)})必須等於num_layers({num_layers})")
            layers = [num_ins] + hidden_sizes + [num_outs]
        else:
            layers = [num_ins] + [hidden_size] * num_layers + [num_outs]
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = activation

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            # 嘗試以工廠方式建立可帶參數activation（相容LAAF與Tanh）
            act = None
            try:
                act = self.activation()
            except TypeError:
                # 若activation需要形狀資訊，可於此傳入（目前Layer-wise LAAF不需要）
                act = self.activation()
            layer_list.append((f'activation_{i}', act))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
        # 應用針對tanh的Xavier初始化（不做額外縮放；縮放由外部依配置完成）
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization tailored for tanh without extra scaling.

        External scaling for first/last layers is applied by the solver
        based on configuration to keep initialization concerns separated.
        """
        gain = nn.init.calculate_gain('tanh')
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.layers(x)
        return out
