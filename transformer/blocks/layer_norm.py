import arr_fun as torch
import arr_fun as nn
from help_func import trace
from help_func.trace import fmt, log, to_array


class LayerNorm(nn.Module):
    """단어 벡터 하나 안의 값들을 평균 0, 퍼진 정도 1로 맞춰서 값이 너무 커지거나 치우치지 않게 함"""
    def __init__(self, features, zero_division_guard=1e-6):
        super(LayerNorm, self).__init__()
        
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        # 원본 코드의 eps입니다. 표준편차가 0일 때 0으로 나누지 않도록 더하는 아주 작은 값입니다.
        self.zero_division_guard = zero_division_guard
        # 로그에서 들어오는 값과 나가는 값을 부르는 이름이에요. 계산에는 영향이 없어요.
        self.input_label = "x"
        self.output_label = "LayerNorm 결과"

    def forward(self, input_tensor):
        # 로그는 따라가는 단어 하나의 평균, 표준편차, 칸마다 정규화 식을 보여줘요.
        # 설명을 먼저 찍고, 그 줄을 계산하고, 결과를 바로 찍어요. 계산은 원래와 같아요.
        if trace.LOG:
            x = to_array(input_tensor)
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[1]
            f = trace.FOLLOW if trace.FOLLOW < x.shape[0] else 0
            me = trace.word_name(f)
            trace.section("LayerNorm: 단어 하나의 칸들을 평균 0, 퍼진 정도 1로 맞춤", "layer_norm.py")
            log("  한 줄 요약: 단어 벡터 하나 안의 칸 값들을 평균 0, 퍼진 정도 1로 맞춰요.")
            log("  왜: 값이 너무 커지거나 한쪽으로 치우치면 다음 계산이 불안정해져서, 블록에 넣기 전에 먼저 크기를 정리해요.")
            x_name = trace.ref(self.input_label, me)
            y_name = trace.result(self.output_label, me)
            log(f"  {x_name} = {fmt(x[f])}")

        mean = input_tensor.mean(-1, keepdim=True)
        if trace.LOG:
            m = to_array(mean).reshape(-1)
            log(f"  평균 = ({' + '.join(f'{v:.2f}' for v in x[f])}) / {n} = {m[f]:.2f}")

        std = input_tensor.std(-1, keepdim=True)
        if trace.LOG:
            sd = to_array(std).reshape(-1)
            squares = " + ".join(f"({v:.2f}-{m[f]:.2f})^2" for v in x[f])
            log(f"  표준편차 = sqrt(({squares}) / {n - 1}) = {sd[f]:.2f}   (표본 표준편차라서 칸 수에서 1을 뺀 값으로 나눠요)")

        # 평균을 빼서 평균 0, 표준편차로 나눠서 퍼진 정도 1로 맞춘 뒤 scale을 곱하고 shift를 더해서 리턴
        # 원래는 바로 return 했는데, 로그를 찍으려고 output에 담았어요. 계산은 같아요.
        if trace.LOG:
            log(f"  칸마다 (x - 평균) / (표준편차 + 0.000001) * scale + shift")
        output = self.scale * (input_tensor - mean) / (std + self.zero_division_guard) + self.shift
        if trace.LOG:
            y = to_array(output)
            y = y.reshape(-1, y.shape[-1])
            a = to_array(self.scale)
            b = to_array(self.shift)
            for c in range(n):
                log(f"    {y_name}[{c}] = ({x[f, c]:.2f} - {m[f]:.2f}) / {sd[f]:.2f} * {a[c]:.2f} + {b[c]:.2f} = {y[f, c]:.2f}")
            log(f"  {y_name} = {fmt(y[f])}")
            others = [f"{trace.result(self.output_label, trace.word_name(i))} = {fmt(y[i])}" for i in range(x.shape[0]) if i != f]
            if others:
                log("  다른 단어: " + ",  ".join(others))
        return output
