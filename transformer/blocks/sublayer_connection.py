import arr_fun as nn

from transformer.blocks.layer_norm import LayerNorm
from help_func import trace
from help_func.trace import fmt, log, to_array


class SublayerConnection(nn.Module):
    """층의 결과를 원래 입력에 더해 줌(잔차 연결)"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # 층에 넣기 전에 단어 벡터 값을 평균 0, 퍼진 정도 1로 맞추는 LayerNorm
        self.norm = LayerNorm(size)
        # 층의 결과 일부를 학습 중에 무작위로 끄는 dropout
        self.dropout = nn.Dropout(dropout)
        # 로그에서 쓰는 이름이에요. 이 블록을 쓰는 레이어가 set_log_labels로 정해 줘요. 계산에는 영향이 없어요.
        self.input_label = "입력"
        self.sublayer_label = "층결과"
        self.output_label = "출력"
        self.log_title = "SublayerConnection"
        # 이 블록 안의 층 이름이에요. 예: "셀프 어텐션". 잔차 연결 설명에만 써요.
        self.block_name = "층"
        # 이 블록의 단계 순서예요. 예: trace.ENCODER_SELF_STEPS. 블록이 시작할 때 "순서: ..." 한 줄로 찍어요.
        self.log_steps = None

    def set_log_labels(self, title, input_label, normed_label, sublayer_label, output_label, block_name="층", steps=None):
        self.log_title = title
        self.log_steps = steps
        self.block_name = block_name
        self.input_label = input_label
        self.norm.input_label = input_label
        self.norm.output_label = normed_label
        self.sublayer_label = sublayer_label
        self.output_label = output_label

    def forward(self, input_tensor, sublayer):
        # 입력을 LayerNorm으로 맞추고, 받은 층(sublayer)에 넣고, dropout을 한 뒤, 원래 입력에 더해서 리턴
        # 결과를 누적한다는 의미에서 원래 입력에서 계산한 값을 더함
        # 원래 한 줄: return input_tensor + self.dropout(sublayer(self.norm(input_tensor)))
        # 중간 값을 로그로 보려고 나눴고 계산은 같아요.
        with trace.place(self.log_title, self.log_steps):
            normed = self.norm(input_tensor)
            sublayer_output = sublayer(normed)
            dropped = self.dropout(sublayer_output)

            # 잔차 연결: 층에 넣기 전 값에 층의 결과를 칸끼리 더하는 걸 보여줘요.
            if trace.LOG:
                trace.section("잔차 연결: 층에 넣기 전 값에 층의 결과를 더함", "sublayer_connection.py")
                log(f"  한 줄 요약: {self.block_name} 층이 찾은 추가 정보를, 이 블록에 들어오기 전 원래 단어 값에 칸끼리 더해요.")
                log(f"  왜: 원래 단어 정보를 잃지 않고 새로 알게 된 것만 덧붙이려고요. 레이어를 여러 개 쌓아도 원래 정보가 남아요. 여기부터는 {self.block_name} 계산이 아니고, 그 결과를 처리하는 단계예요.")
                log("  (eval 모드라 dropout은 아무것도 끄지 않아요. 층의 결과가 그대로 더해져요.)")
            output = input_tensor + dropped
            if trace.LOG:
                x = to_array(input_tensor)
                x = x.reshape(-1, x.shape[-1])
                d = to_array(dropped).reshape(x.shape)
                y = to_array(output).reshape(x.shape)
                f = trace.FOLLOW if trace.FOLLOW < x.shape[0] else 0
                me = trace.word_name(f)
                input_name = trace.ref(self.input_label, me)
                sublayer_name = trace.ref(self.sublayer_label, me)
                output_name = trace.result(self.output_label, me)
                others = [f"{trace.result(self.output_label, trace.word_name(i))} = {fmt(y[i])}" for i in range(x.shape[0]) if i != f]
                trace.add_formula(input_name, x[f], sublayer_name, d[f], output_name, y[f], others)
                log(f"  여기서 {self.block_name} 블록이 끝났어요. 이 블록의 결과는 {trace._short(output_name)} 값이에요.")
        return output
