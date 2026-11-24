import arr_fun as nn
import arr_fun as torch
import math

from help_func import trace
from help_func.trace import fmt, log, to_array

class PositionalEncoding(nn.Module):
    "implement the pe function"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(probability=dropout)


        # compute the positional encodings once in log space
        position_table = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        position_table[:, 0::2] = torch.sin(position * div_term)
        position_table[:, 1::2] = torch.cos(position * div_term)
        position_table = position_table.unsqueeze(0)
        self.register_buffer("pe", position_table)
        # 로그에서 쓰는 이름이에요. 계산에는 영향이 없어요.
        self.input_label = "임베딩"
        self.output_label = "위치 더한 값"


    def forward(self, embedded_tokens):
        # 자리마다 정해진 pe 줄을 칸끼리 더하는 걸 보여줘요. 설명을 먼저 찍고, 더하고, 결과를 바로 찍어요.
        if trace.LOG:
            x = to_array(embedded_tokens)[0]
            pe = to_array(self.pe)[0]
            f = trace.FOLLOW if trace.FOLLOW < x.shape[0] else 0
            me = trace.word_name(f)
            trace.section("위치 정보 더하기: 단어 자리마다 정해진 pe 줄을 더함", "positional_encoding.py")
            log("  한 줄 요약: 단어 자리(0번째, 1번째, ...)마다 정해진 위치 값을 단어 벡터에 칸끼리 더해요.")
            log("  왜: 어텐션은 단어 순서를 모르기 때문에, 몇 번째 자리인지를 값에 직접 넣어 줘야 순서를 구분할 수 있어요.")
            log("  pe 표는 학습하지 않는 고정값이에요. 짝수 칸은 sin(자리 * 주기), 홀수 칸은 cos(자리 * 주기)")
            log(f"  {trace.word_label(me)}는 {f}번째 자리라서 pe {f}번째 줄을 씀: 위치_{me} = {fmt(pe[f])}")

        # 원래는 embedded_tokens에 바로 더했는데, 더하기 전 값을 로그로 보려고 이름을 나눴어요. 계산은 같아요.
        added = embedded_tokens + self.pe[:, : embedded_tokens.size(1)].requires_grad_(False)
        if trace.LOG:
            y = to_array(added)[0]
            input_name = trace.ref(self.input_label, me)
            output_name = trace.result(self.output_label, me)
            others = [f"{trace.result(self.output_label, trace.word_name(i))} = {fmt(y[i])}" for i in range(x.shape[0]) if i != f]
            trace.add_formula(input_name, x[f], f"위치_{me}", pe[f], output_name, y[f], others)
        return self.dropout(added)
