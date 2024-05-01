import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.utils import transpose
from peft.tuners.tuners_utils import BaseTunerLayer

class SloraLayer(BaseTunerLayer):
    def __init__(self, num_loras: int, negated_adapters: list, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.n = num_loras
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleList([nn.ModuleDict({}) for i in range(self.n)])
        self.lora_A = nn.ModuleList([nn.ModuleDict({}) for i in range(self.n)])
        self.lora_B = nn.ModuleList([nn.ModuleDict({}) for i in range(self.n)])
        self.negated_adapters = negated_adapters
        self.layer_num = 0
        # For Embedding layer
        self.lora_embedding_A = nn.ModuleList([nn.ParameterDict({}) for i in range(self.n)])
        self.lora_embedding_B = nn.ModuleList([nn.ParameterDict({}) for i in range(self.n)])
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, slora_idx, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[slora_idx].update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[slora_idx].update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B[slora_idx].update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(slora_idx, adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(self, adapter_name, slora_idx, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[slora_idx].update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            weight_A = torch.randn((r, self.in_features), dtype=self.weight.dtype, device=self.weight.device)
            weight_B = torch.randn((self.out_features, r), dtype=self.weight.dtype, device=self.weight.device)
            self.lora_embedding_A[slora_idx].update(nn.ParameterDict({adapter_name: nn.Parameter(weight_A)}))
            self.lora_embedding_B[slora_idx].update(nn.ParameterDict({adapter_name: nn.Parameter(weight_B)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(slora_idx, adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, slora_idx, adapter_name):
        if adapter_name in self.lora_A[slora_idx].keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[slora_idx][adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[slora_idx][adapter_name].weight)
        if adapter_name in self.lora_embedding_A[slora_idx].keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[slora_idx][adapter_name])
            nn.init.normal_(self.lora_embedding_B[slora_idx][adapter_name])

class SloraLinear(nn.Linear, SloraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SloraLayer.__init__(self, n, negated_adapters, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        for i in range(0, n):
            self.update_layer(adapter_name, i, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name
            self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )
    
class SloraSumLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                if i not in self.negated_adapters:
                    result += (
                        self.lora_B[i][self.active_adapter](
                            self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                else:
                    result -= (
                        self.lora_B[i][self.active_adapter](
                            self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )

        result = result.to(previous_dtype)

        return result
    
class SloraLoraHubLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        hub_A = torch.zeros_like(self.lora_A[0][self.active_adapter].weight.T)
        hub_B = torch.zeros_like(self.lora_B[0][self.active_adapter].weight.T)

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                if i not in self.negated_adapters:
                    hub_A += self.lora_A[i][self.active_adapter].weight.T
                    hub_B += self.lora_B[i][self.active_adapter].weight.T
                else:
                    hub_A -= self.lora_A[i][self.active_adapter].weight.T
                    hub_B -= self.lora_B[i][self.active_adapter].weight.T

        result += torch.matmul(torch.matmul(x, hub_A), hub_B) * self.scaling[self.active_adapter]

        result = result.to(previous_dtype)

        return result

class SloraAvgLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                if i not in self.negated_adapters:
                    result += (
                        self.lora_B[i][self.active_adapter](
                            self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter] / self.n
                    )
                else:
                    raise BaseException("Negative Adapters Not Supported! " \
                                        f"Ensure negated_adapters = [] in the config")

        result = result.to(previous_dtype)

        return result

class SloraAbsMaxLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        outs = []

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                outs.append(
                    self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )

        max_idx = 0
        max_abs_sum = 0

        if i not in self.negated_adapters:
            for i in range(0, self.n):
                abs_sum = torch.abs(outs[i]).sum()

                if abs_sum > max_abs_sum:
                    max_abs_sum = abs_sum
                    max_idx = i

            result += outs[max_idx]

        else:
            raise BaseException("Negative Adapters Not Supported! " \
                                f"Ensure negated_adapters = [] in the config")

        result = result.to(previous_dtype)

        return result
    

class SloraMaxSumLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        outs = []

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                

                outs.append(
                    self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )

        max_idx = 0
        max_abs_sum = 0

        if i not in self.negated_adapters:
            for i in range(0, self.n):
                abs_sum = outs[i].sum()

                if abs_sum > max_abs_sum:
                    max_abs_sum = abs_sum
                    max_idx = i

            result += outs[max_idx]
        else:
            raise BaseException("Negative Adapters Not Supported! " \
                                f"Ensure negated_adapters = [] in the config")

        result = result.to(previous_dtype)

        return result
    

class SloraMaxElemLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        outs = []

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                outs.append(
                    self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )

        if i not in self.negated_adapters:
            max_elem = torch.ones_like(result) * -1
            for i in range(0, self.n):
                max_elem = torch.maximum(max_elem, outs[i])

            result += max_elem
        else:
            raise BaseException("Negative Adapters Not Supported! " \
                                f"Ensure negated_adapters = [] in the config")
        
        result = result.to(previous_dtype)

        return result
    
class SloraMaxDiffElemLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        outs = []

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                outs.append(
                    self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )

        # Propogate max element-wise value for conjuncted adapters
        max_elem = torch.zeros_like(result)
        for i in range(0, self.n):
            if i not in self.negated_adapters:
                out = torch.zeros_like(max_elem)
                
                c = torch.zeros_like(max_elem)
                c[torch.abs(max_elem) > torch.abs(outs[i])] = 1
                out = c * max_elem

                c = torch.zeros_like(max_elem)
                c[torch.abs(max_elem) < torch.abs(outs[i])] = 1
                out += c * outs[i]

                max_elem = out        

        # Then subtract min element-wise value for negated adapters
        for i in range(0, self.n):
            if i in self.negated_adapters:
                out = torch.zeros_like(max_elem)

                c = torch.zeros_like(max_elem)
                c[torch.abs(max_elem) < torch.abs(outs[i])] = 1
                out = c * max_elem

                c = torch.zeros_like(max_elem)
                c[torch.abs(max_elem) > torch.abs(outs[i])] = 1
                out -= c * outs[i]

                max_elem = out   
        
        result += max_elem            
        
        result = result.to(previous_dtype)

        return result
    
class SloraConcatLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[0][self.active_adapter].weight.dtype)
        if i not in self.negated_adapters:
            dropout = torch.transpose(self.lora_dropout[0][self.active_adapter](x), 1, 2)

            #print(dropout.shape)

            A = self.lora_A[0][self.active_adapter].weight
            B = self.lora_B[0][self.active_adapter].weight

            for i in range(1, self.n):
                if self.active_adapter not in self.lora_A[i].keys():
                    raise(f"Active lora adapter {i} not found!")
                    return result
                if self.disable_adapters:
                    raise(f"Lora adapter {i} disabled!")
                    if self.r[self.active_adapter] > 0 and self.merged:
                        self.unmerge()
                elif self.r[self.active_adapter] > 0 and not self.merged:
                    
                    A = torch.cat((A, self.lora_A[i][self.active_adapter].weight), 0)
                    B = torch.cat((B, self.lora_B[i][self.active_adapter].weight), 1)

            batch_size = dropout.shape[0]

            A = A.repeat(batch_size, 1, 1)
            B = B.repeat(batch_size, 1, 1)

            #print(A.shape)
            #print(B.shape)

            out = torch.bmm(B, torch.bmm(A, dropout))        

            result += torch.transpose(out, 1, 2)
        else:
            raise BaseException("Negative Adapters Not Supported! " \
                                f"Ensure negated_adapters = [] in the config")
        
        result = result.to(previous_dtype)

        return result
    
class SloraLSELinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        adapter_list = []
        
        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                
                adapter_list.append(self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                    )
                    * self.scaling[self.active_adapter]
                )

        if i not in self.negated_adapters:
            stacked_adapter = torch.stack(adapter_list, dim=0)
            lse_adapter = torch.sum(stacked_adapter, dim=0) + torch.logsumexp(-stacked_adapter, dim=0)

            result += lse_adapter
        else:
            raise BaseException("Negative Adapters Not Supported! " \
                                f"Ensure negated_adapters = [] in the config")
        
        result = result.to(previous_dtype)

        return result
    

class SloraSMLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        #print(self.layer_num)

        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        adapter = []

        if self.layer_num == 0:
            #x_new = torch.stack((x_new, x_new), dim=0)
            #print(x_new.shape)If
            pass

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                if i not in self.negated_adapters:
                    mask = torch.ones_like(x)
                    mask[torch.arange(mask.size(0)) % self.n == i] = 0

                    result += (
                        self.lora_B[i][self.active_adapter](
                            self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter] * mask
                    )
                else:
                    raise BaseException("Negative Adapters Not Supported! " \
                                        f"Ensure negated_adapters = [] in the config")

        #adapter_output = torch.stack(adapter, dim=0)
        #print(adapter_output.shape)
        #result = x_new + adapter_output

        #result[(i+1) % self.n] = 0

        return result.to(previous_dtype)
    
class SloraArgMaxLogitLinear(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
        
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # print(f'Layer ID: {self.layer_num}')
        # print(x[0].shape)

        # result has size (adapter_num, bsz, embedding)
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        adapter_outputs = []

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                # adapter output has size (1, bsz, embedding)
                adapter_outputs.append(
                    (self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x[i]))
                    )
                    * self.scaling[self.active_adapter]).unsqueeze(dim=0)
                )

        # All adapters are run idependently of one another until logits are compared in the SloraBaseModel forward() call
        result += torch.cat(adapter_outputs, dim=0)

        return result.to(previous_dtype)

class SloraConsensus(SloraLinear):
    def __init__(
        self,
        adapter_name: str,
        n: int,
        negated_adapters: list,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        super().__init__(adapter_name, 
                         n,
                         negated_adapters, 
                         in_features, 
                         out_features,
                         r=r,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         fan_in_fan_out=fan_in_fan_out,
                         is_target_conv_1d_layer=is_target_conv_1d_layer, 
                         **kwargs,
                         )
    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # print(f'Layer ID: {self.layer_num}')
        # Increase size of x 
        #x = torch.cat((x, x[0:1]), dim=0)

        # result has size (adapter_num+1, bsz, embedding)
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        adapter_outputs = [torch.zeros_like(x[0]).unsqueeze(dim=0)]

        for i in range(0, self.n):
            if self.active_adapter not in self.lora_A[i].keys():
                raise(f"Active lora adapter {i} not found!")
                return result
            if self.disable_adapters:
                raise(f"Lora adapter {i} disabled!")
                if self.r[self.active_adapter] > 0 and self.merged:
                    self.unmerge()
            elif self.r[self.active_adapter] > 0 and not self.merged:
                x = x.to(self.lora_A[i][self.active_adapter].weight.dtype)

                # adapter output has size (1, bsz, embedding)
                adapter_outputs.append(
                    (self.lora_B[i][self.active_adapter](
                        self.lora_A[i][self.active_adapter](self.lora_dropout[i][self.active_adapter](x[i+1]))
                    )
                    * self.scaling[self.active_adapter]).unsqueeze(dim=0)
                )

        # All adapters are run idependently of one another until logits are compared in the SloraBaseModel forward() call

        result += torch.cat(adapter_outputs, dim=0)

        return result.to(previous_dtype)

if __name__ == '__main__':
    model = SloraConsensus()