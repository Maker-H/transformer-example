import inspect

import arr_fun as torch

RUN_EXAMPLES = True


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(example_function, args=[]):
    caller_frame = inspect.currentframe().f_back
    if RUN_EXAMPLES and caller_frame and caller_frame.f_globals.get("__name__") == "__main__":
        return example_function(*args)


def execute_example(example_function, args=[]):
    caller_frame = inspect.currentframe().f_back
    if RUN_EXAMPLES and caller_frame and caller_frame.f_globals.get("__name__") == "__main__":
        example_function(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
