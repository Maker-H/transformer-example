"""학습할 때 가중치를 고치는 Optimizer와, 학습률을 단계마다 바꾸는 LambdaLR이에요.
arr_fun에는 아직 backward(기울기 계산)가 없어서, 기울기(grad)가 없는 가중치는 그대로 둬요.
"""

from arr_fun.tensor import _tensor_to_array
from help_func.trace import log


class Optimizer:
    """가중치를 '가중치 - 학습률 x 기울기' 로 조금씩 고치는 가장 단순한 방식(경사 하강법)이에요.
    Annotated Transformer는 torch.optim.Adam을 쓰는데, Adam은 이 식에 기울기 평균 같은 보정을 더한 것이에요.
    """

    def __init__(self, params=None, defaults=None, lr=0.0):
        # 고칠 가중치(Parameter) 목록이에요. model.parameters()를 받아서 리스트로 저장해 둬요.
        self.parameters = list(params) if params is not None else []
        self.defaults = defaults or {}
        # torch와 같은 모양이에요. 학습률은 param_groups[0]["lr"]에 들어 있고, scheduler가 이 값을 바꿔요.
        self.param_groups = [{"lr": lr, "initial_lr": lr}]

    def step(self):
        log("기울기(grad)를 보고 가중치를 조금씩 고치는 단계예요. 식은 '새 가중치 = 가중치 - 학습률 x 기울기' 예요.")
        learning_rate = self.param_groups[0]["lr"]
        log(f"    지금 학습률은 {learning_rate:.6f}이에요.")

        log("    먼저 기울기가 구해진 가중치 묶음만 골라요.")
        parameters_with_grad = [parameter for parameter in self.parameters if getattr(parameter, "grad", None) is not None]
        log(f"    가중치 묶음 {len(self.parameters)}개 중 기울기가 있는 묶음은 {len(parameters_with_grad)}개예요.")

        if not parameters_with_grad:
            log("    arr_fun에는 아직 backward가 없어서 기울기가 하나도 없어요. 그래서 이번에는 가중치를 하나도 안 고쳐요.")
            return None

        log("    기울기가 있는 묶음마다 식대로 가중치를 고쳐요.")
        for parameter in parameters_with_grad:
            parameter._array = parameter._array - learning_rate * _tensor_to_array(parameter.grad)
        log(f"    가중치 묶음 {len(parameters_with_grad)}개를 고쳤어요.")
        return None

    def zero_grad(self, set_to_none=False):
        log("다음 묶음의 기울기와 섞이지 않게, 쌓여 있던 기울기를 비워요.")
        cleared_count = 0
        for parameter in self.parameters:
            if getattr(parameter, "grad", None) is not None:
                parameter.grad = None if set_to_none else parameter.grad * 0
                cleared_count += 1
        log(f"    기울기를 비운 가중치 묶음은 {cleared_count}개예요.")
        return None


class LambdaLR:
    """torch.optim.lr_scheduler.LambdaLR을 흉내 낸 것이에요.
    학습률 = 처음 학습률 x lr_lambda(지금까지 scheduler.step을 부른 횟수)
    """

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.lr_lambda = lr_lambda
        self.base_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
        # torch처럼 만들자마자 0단계 학습률을 넣어 둬요.
        self.last_step = 0
        self.optimizer.param_groups[0]["lr"] = self.base_lr * self.lr_lambda(0)

    def step(self):
        log("학습률을 다음 단계 값으로 바꿔요. 식은 '학습률 = 처음 학습률 x rate(단계 번호)' 예요.")
        self.last_step += 1
        factor = self.lr_lambda(self.last_step)
        new_learning_rate = self.base_lr * factor
        self.optimizer.param_groups[0]["lr"] = new_learning_rate
        log(f"    {self.last_step}단계 학습률 = {self.base_lr} x {factor:.6f} = {new_learning_rate:.6f}")

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]
