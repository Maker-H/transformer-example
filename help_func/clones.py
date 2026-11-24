import copy

import arr_fun as nn


def clones(module, num_copies):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])
