"""이 프로젝트가 쓰는 계층들입니다. 순전파만 구현합니다."""

from collections import OrderedDict
import copy

import numpy as np

from arr_fun.tensor import Tensor, _tensor_to_array


class Module:
    def __init__(self):
        # self.name = orderdict
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        """보관함이랑 속성 두곳에 다 저장(복사본x)"""
        if name in ("_modules", "_parameters", "_buffers", "training"):
            object.__setattr__(self, name, value)
            return
        if "_modules" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            self._parameters[name] = value
            self._modules.pop(name, None)
            self._buffers.pop(name, None)
        elif isinstance(value, Module):
            self._modules[name] = value
            self._parameters.pop(name, None)
            self._buffers.pop(name, None)
        elif name in self._buffers:
            self._buffers[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        for parameter in self._parameters.values():
            yield parameter # 값을 하나씩 내보냄 
        for child_module in self._modules.values():
            yield from child_module.parameters()

    def buffers(self):
        for buffer in self._buffers.values():
            yield buffer
        for child_module in self._modules.values():
            yield from child_module.buffers()

    def eval(self):
        self.training = False
        for child_module in self._modules.values():
            child_module.eval()
        return self

    def train(self, mode=True):
        self.training = mode
        for child_module in self._modules.values():
            child_module.train(mode)
        return self

    def add_module(self, name, module):
        setattr(self, name, module)

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def modules(self):
        yield self
        for child_module in self._modules.values():
            yield from child_module.modules()

    def __deepcopy__(self, memo):
        module_class = self.__class__
        copied_module = object.__new__(module_class)
        memo[id(self)] = copied_module
        object.__setattr__(copied_module, "_modules", OrderedDict())
        object.__setattr__(copied_module, "_parameters", OrderedDict())
        object.__setattr__(copied_module, "_buffers", OrderedDict())
        object.__setattr__(copied_module, "training", self.training)
        for attribute_name, attribute_value in self.__dict__.items():
            if attribute_name in ("_modules", "_parameters", "_buffers"):
                continue
            setattr(copied_module, attribute_name, copy.deepcopy(attribute_value, memo))
        return copied_module


class Parameter(Tensor):
    """학습 대상인 텐서입니다. make_model이 여기서 가중치를 모아 초기화합니다."""

    def __repr__(self):
        # print() 하면 __repr__이 나옴
        return "Parameter(" + super().__repr__() + ")"

    def __deepcopy__(self, memo):
        cloned = Parameter(_array=self._array.copy())
        memo[id(self)] = cloned
        return cloned


class Sequential(Module):
    # model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2))
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        for layer_index, module in enumerate(layers):

            # 각 층이 0, 1, 라는 이름으로 _modules 보관함에 저장됨
            self.add_module(str(layer_index), module) 

    def __getitem__(self, layer_index):
        if layer_index < 0:
            layer_index = len(self._modules) + layer_index
        return self._modules[str(layer_index)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for layer_index in range(len(self._modules)):
            yield self[layer_index]

    def forward(self, input_tensor):
        current_output = input_tensor
        for layer in self:
            current_output = layer(current_output)
        return current_output

class ModuleList(Module):
    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            for layer_index, module in enumerate(modules):
                self.add_module(str(layer_index), module)

    def __getitem__(self, layer_index):
        if layer_index < 0:
            layer_index = len(self) + layer_index
        return self._modules[str(layer_index)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for layer_index in range(len(self)):
            yield self[layer_index]

class Dropout(Module):
    """일부 값을 무작위로 0으로 끄는 클래스 """
    def __init__(self, probability=0.5):
        super(Dropout, self).__init__()
        self.p = probability

    def forward(self, input_tensor):
        if not self.training or self.p == 0.0:
            return input_tensor
        input_array = _tensor_to_array(input_tensor)
        # 0과 1사이 무작위 소수로 같은 모양 표를 만듬 
        # 그 이후 살아남을 확률이 1-p
        keep_mask = (np.random.rand(*input_array.shape) >= self.p).astype(input_array.dtype)
        # 학습 때 끄는 칸이 있어도 평균 크기를 유지하려고 1-p로 나눔
        return Tensor(_array=input_array * keep_mask / (1.0 - self.p))
