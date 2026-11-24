"""출력을 (입력 @ 가중치의 전치) + 편향 으로 계산합니다."""

import numpy as np

from arr_fun.modules import Module, Parameter
from arr_fun.tensor import matmul
from help_func import trace
from help_func.trace import fmt, log, to_array


class Linear(Module):
    """
    in_feature가 512고 out_feature가 2048이면 
    길이가 512인 벡터를 받아서 2048 벡터로 바꿔서 내보냄 
    """

    def __init__(self, in_features, out_features, bias=True):
        # 1. 부모인 Module의 __init__을 실행해서 보관함 네 개를 먼저 만들어요.
        #    이 줄이 있어야 아래에서 self.weight를 저장할 때 _parameters에 등록돼요.
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 로그에 찍을 이름이에요. MultiHeadedAttention처럼 Linear를 여러 개 쓰는 곳에서
        # "query_projection"처럼 바꿔 주면 어느 Linear인지 구분돼요.
        self.log_name = "Linear"
        # 로그에서 들어온 값과 나간 값을 부르는 이름이에요. 예: x_A 를 넣으면 Q_A 가 나옴
        self.input_label = "x"
        self.output_label = "y"
        # 결과에 붙여 줄 짧은 별명이에요. 예: "Q"면 결과 줄 끝에 (줄여서 부르는 이름: Q_A)를 붙여요. 없으면 None이에요.
        self.output_alias = None
        # 단계 제목 바로 아래에 찍을 설명 줄이에요. 이 Linear를 쓰는 곳에서 "무엇을, 왜 하는지"를 넣어 줘요.
        self.log_about = []
        # 들어오는 단어가 Q 쪽 문장인지 K, V 쪽 문장인지예요. 로그에서 단어 이름을 고를 때만 써요.
        self.name_side = "q"

        # 3. 행은 out_features개(2), 열은 in_features개(3)
        weight_table = np.empty((out_features, in_features), dtype=np.float32)

        #    가중치 표를 Parameter로 감싸서 저장
        #    Module의 __setattr__이 이 값을 _parameters["weight"]에 등록
        self.weight = Parameter(weight_table)

        # 4. 편향을 만들어요. bias가 True면 출력 칸마다 하나씩, 길이 2인 리스트를 만들어요.
        if bias:
            bias_list = []
            for _ in range(out_features):
                bias_list.append(0.0)

            self.bias = Parameter(bias_list)
        else:
            self.bias = None

            # 5. 가중치와 편향에 작은 무작위 시작값을 넣어요.
            #    모두 0이면 학습이 제대로 시작되지 않기 때문에 이 단계가 필요해요.
        self._initialize_weight_and_bias()


    def _initialize_weight_and_bias(self):
        from arr_fun.init import xavier_uniform_
        xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias._array.fill(0)

    def forward(self, input_tensor):
        # 로그는 따라가는 단어 하나가 표 W와 어떻게 곱해지는지 칸마다 식으로 보여줘요.
        # 설명(표 W와 들어온 값)을 먼저 찍고, 곱셈을 계산하고, 칸마다 식과 결과를 바로 찍어요.
        if trace.LOG:
            inputs = to_array(input_tensor).reshape(-1, self.in_features)
            weights = to_array(self.weight)
            biases = to_array(self.bias) if self.bias is not None else np.zeros(self.out_features)
            name_of = trace.kv_name if self.name_side == "kv" else trace.word_name
            f = trace.FOLLOW if trace.FOLLOW < inputs.shape[0] else 0
            me = name_of(f)
            trace.section(self.log_name, "linear.py")
            for line in self.log_about:
                log(f"  {line}")
            x_name = trace.ref(self.input_label, me)
            y_name = trace.result(self.output_label, me)
            alias = f"   (줄여서 부르는 이름: {self.output_alias}_{me})" if self.output_alias else ""
            if trace.COMPACT:
                log(f"  {x_name} = {fmt(inputs[f])}")
                log(f"  표 W ({self.out_features}줄 x {self.in_features}칸).  {y_name}[j] = ({trace._short(x_name)} 칸 * W j번째 줄 칸) 을 칸마다 곱해서 다 더함 + 편향 b")
            else:
                log(f"  표 W ({self.out_features}줄 x {self.in_features}칸)")
                for row in range(weights.shape[0]):
                    log(f"    {row}번째 줄 {fmt(weights[row])}")
                log(f"  {x_name} {fmt(inputs[f])} 와 W의 줄을 칸끼리 곱해서 더함 (+ 편향 b)")

        weight_transposed = self.weight.transpose(0, 1)
        output = matmul(input_tensor, weight_transposed)
        if self.bias is not None:
            output = output + self.bias

        if trace.LOG:
            outputs = to_array(output).reshape(-1, self.out_features)
            if trace.COMPACT:
                for row in range(weights.shape[0]):
                    terms = " + ".join(f"{inputs[f, c]:.2f}*{weights[row, c]:.2f}" for c in range(self.in_features))
                    log(f"    W {row}번째 줄 {fmt(weights[row])}   {y_name}[{row}] = {terms} + {biases[row]:.2f} = {outputs[f, row]:.2f}")
                # Q, K, V는 단어 하나만이 아니라 모든 단어 줄을 가진 행렬로 보여줘요. 간단 모드에서도 보여요.
                if self.output_alias in ("Q", "K", "V"):
                    trace.matrix(f"{_section_number(y_name)}번 {self.output_alias}", outputs, [name_of(i) for i in range(inputs.shape[0])])
                log(f"  {y_name} = {fmt(outputs[f])}{alias}")
                others = [f"{trace.result(self.output_label, name_of(i))} = {fmt(outputs[i])}" for i in range(inputs.shape[0]) if i != f]
                if others:
                    log("  다른 단어도 같은 표로: " + ",  ".join(others))
            else:
                for row in range(weights.shape[0]):
                    terms = " + ".join(f"{inputs[f, c]:.2f}*{weights[row, c]:.2f}" for c in range(self.in_features))
                    log(f"    {y_name}의 {row}번째 칸 = {terms} + {biases[row]:.2f} = {outputs[f, row]:.2f}")
                if self.output_alias in ("Q", "K", "V"):
                    trace.matrix(f"{_section_number(y_name)}번 {self.output_alias}", outputs, [name_of(i) for i in range(inputs.shape[0])])
                log(f"  결과 (다른 단어도 같은 표로 계산)")
                for index in range(inputs.shape[0]):
                    mark = f"   (따라가는 단어){alias}" if index == f else ""
                    log(f"    {trace.result(self.output_label, name_of(index))} = {fmt(outputs[index])}{mark}")
        return output


def _section_number(value_name):
    """"39번 결과_시작"에서 단계 번호 39를 꺼내요. 로그 이름에만 써요."""
    return value_name.split("번")[0]
