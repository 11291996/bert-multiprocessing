import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter, init
from torch.nn import functional as F
import math

class ParallelLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dtype=None) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.left_device = torch.device("cuda:1")
        self.right_device = torch.device("cuda:0")

        factory_kwargs = {'device': torch.device("cpu"), 'dtype': dtype}
        factory_kwargs_left = {'device': self.left_device, 'dtype': dtype}
        factory_kwargs_right = {'device': self.right_device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.left_weight = Parameter(torch.empty((out_features, in_features // 2), **factory_kwargs_left))
        self.right_weight = Parameter(torch.empty((out_features, in_features // 2 + in_features % 2), **factory_kwargs_right))

        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def divide_weight(self):
        self.left_weight = Parameter(torch.tensor(self.weight[...,:self.in_features // 2], device=self.left_device))
        self.right_weight = Parameter(torch.tensor(self.weight[...,self.in_features // 2:], device=self.right_device))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.left_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.right_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.left_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # devide
        left_input = input[...,:(self.in_features // 2)].to(self.left_device)
        right_input = input[...,(self.in_features // 2):].to(self.right_device)

        # partial linear operation
        left_product = F.linear(left_input, self.left_weight)
        right_product = F.linear(right_input, self.right_weight)
        
        # summation
        left_product = left_product.to(torch.device("cpu"))
        right_product = right_product.to(torch.device("cpu"))
    
        return left_product + right_product + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )