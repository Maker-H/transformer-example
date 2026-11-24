import arr_fun as nn
from help_func import trace
from help_func.trace import fmt, log, to_array


class PositionwiseFeedForward(nn.Module):
    """단어 벡터를 넓혀 특징을 많이 만들고, 필요 없는 특징은 relu로 끈 뒤 다시 원래 크기로 줄임"""

    def __init__(self, model_dim, hidden_dim, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.expand_layer = nn.Linear(model_dim, hidden_dim)
        self.shrink_layer = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # 로그에서 쓰는 이름이에요. 계산에는 영향이 없어요.
        self.expand_layer.log_name = "expand_layer (칸 수 넓히기)"
        self.expand_layer.output_label = "피드포워드 넓힌 값"
        self.shrink_layer.log_name = "shrink_layer (원래 칸 수로 줄이기)"
        self.shrink_layer.input_label = "ReLU 결과"
        self.shrink_layer.output_label = "피드포워드 출력"
        self.expand_layer.log_about = [
            f"한 줄 요약: 단어 값 {model_dim}칸에 학습 가중치 표 W를 곱해서 {hidden_dim}칸으로 넓혀요.",
            "왜: 칸을 넓히면 단어마다 더 많은 특징을 따로 계산할 수 있어요. 피드포워드는 어텐션과 달리 다른 단어를 안 보고, 단어마다 따로 계산해요.",
        ]
        self.shrink_layer.log_about = [
            f"한 줄 요약: {hidden_dim}칸으로 넓혔던 값에 학습 가중치 표 W를 곱해서 다시 {model_dim}칸으로 줄여요.",
            f"왜: 다음 잔차 연결에서 원래 단어 값({model_dim}칸)에 더하려면 칸 수가 같아야 해요.",
        ]

    def set_log_labels(self, input_label, output_label):
        self.expand_layer.input_label = input_label
        self.shrink_layer.output_label = output_label

    def forward(self, input_tensor):
        # 입력을 weight와 행렬곱하고 bias를 더해 칸 수를 hidden_dim으로 넓힌 뒤 relu로 음수는 0으로 만들어서 꺼버림
        # 원래 한 줄: expanded = self.expand_layer(input_tensor).relu()
        # relu 전 값을 로그로 보려고 나눴고 계산은 같아요.
        before_relu = self.expand_layer(input_tensor)

        # ReLU에서 음수라서 0이 된 칸을 보여줘요. 설명을 먼저 찍고, relu를 계산하고, 결과를 바로 찍어요.
        if trace.LOG:
            x = to_array(before_relu)
            x = x.reshape(-1, x.shape[-1])
            f = trace.FOLLOW if trace.FOLLOW < x.shape[0] else 0
            me = trace.word_name(f)
            trace.section("ReLU: 음수 칸은 0으로, 양수 칸은 그대로", "positionwise_feed_forward.py")
            log("  한 줄 요약: 넓힌 값에서 음수 칸은 0으로 바꾸고, 양수 칸은 그대로 둬요.")
            log("  왜: 쓸모없는 특징(음수)을 꺼 줘야, 곱하고 더하기만으로는 못 만드는 복잡한 관계를 표현할 수 있어요.")
            log(f"  {trace.ref(self.expand_layer.output_label, me)} = {fmt(x[f])}")
        expanded = before_relu.relu()
        if trace.LOG:
            y = to_array(expanded).reshape(x.shape)
            zeroed = [str(c) for c in range(x.shape[1]) if x[f, c] < 0]
            if zeroed:
                log(f"  음수라서 0이 된 칸: {', '.join(zeroed)}번째 ({', '.join(f'{x[f, int(c)]:.2f}' for c in zeroed)})")
            else:
                log("  음수인 칸이 없어서 그대로예요.")
            log(f"  {trace.result(self.shrink_layer.input_label, me)} = {fmt(y[f])}")
            others = [f"{trace.result(self.shrink_layer.input_label, trace.word_name(i))} = {fmt(y[i])}" for i in range(x.shape[0]) if i != f]
            if others:
                log("  다른 단어: " + ",  ".join(others))

        # 학습 중이면 각 칸을 dropout rate 확률로 0으로 만들어서 끔
        expanded = self.dropout(expanded)

        # 넓힌 걸 weight와 행렬곱하고 bias를 더해 칸 수를 model_Dim으로 줄인뒤 리턴
        return self.shrink_layer(expanded)
