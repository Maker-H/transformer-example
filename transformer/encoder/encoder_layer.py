import arr_fun as nn

from help_func.clones import clones
from transformer.blocks.sublayer_connection import SublayerConnection
from help_func import trace
from help_func.trace import log


class EncoderLayer(nn.Module):
    """인코더 층 하나로, self attention과 feed forward를 차례로 함
    self attention으로 입력 문장의 단어들끼리 서로 참고하고, feed forward로 단어마다 따로 가공함
    두 단계 모두 SublayerConnection으로 감싸서 결과를 원래 입력에 누적함
    """

    def __init__(self, model_dim, self_attention, feed_forward, dropout_rate):
        super(EncoderLayer, self).__init__()
        # 입력 문장의 단어들끼리 서로 참고하는 MultiHeadedAttention
        self.self_attention = self_attention
        # 단어마다 따로 가공하는 PositionwiseFeedForward
        self.feed_forward = feed_forward
        # self attention용, feed forward용 SublayerConnection 2개
        connections = clones(SublayerConnection(model_dim, dropout_rate), 2)
        self.attention_connection = connections[0]
        self.feed_forward_connection = connections[1]
        # Encoder가 마지막 LayerNorm 크기를 정할 때 씀
        self.model_dim = model_dim
        # 몇 번째 인코더 레이어인지예요. Encoder가 1, 2, ... 로 정해 줘요. 로그 제목에만 써요.
        self.layer_number = 1

    def forward(self, source_hidden, source_mask):
        if trace.LOG:
            self._set_log_labels()
            log("")
            log(f"[인코더 레이어 {self.layer_number} 시작] 인코더 레이어 하나는 아래 두 묶음을 차례로 해요.")
            log("    셀프 어텐션 블록: LayerNorm, 멀티헤드 어텐션(Q, K, V 만들기, head로 쪼개기, head마다 attention, head 합치기, W_O), 잔차 더하기", level=trace.DETAIL)
            log("    피드포워드 블록: LayerNorm, 피드포워드(넓히기, ReLU, 줄이기), 잔차 더하기", level=trace.DETAIL)
        # 1) self attention: query, key, value 모두 같은 입력 문장을 씀
        source_hidden = self.attention_connection(
            source_hidden,
            lambda normed_source: self.self_attention(normed_source, normed_source, normed_source, source_mask),
        )
        # 2) feed forward
        return self.feed_forward_connection(source_hidden, self.feed_forward)

    def _set_log_labels(self):
        """로그에서 값마다 붙일 이름을 정해요. 계산에는 영향이 없어요."""
        n = self.layer_number
        where = f"인코더 레이어 {n} (encoder_layer.py)"
        # 값 이름은 로그에서 "12번 결과_A (셀프 어텐션 출력)"처럼, 그 값을 만든 단계 번호와 역할로 바뀌어 찍혀요.
        # 레이어 1의 입력은 위치 정보를 더한 값이고, 레이어 2부터는 앞 레이어의 결과예요.
        layer_input = "인코더 레이어 1 입력" if n == 1 else f"인코더 레이어 {n - 1}의 최종 결과"
        block1_result = f"인코더 레이어 {n} 블록 1(셀프 어텐션) 결과"
        self.attention_connection.set_log_labels(
            f"{where} / 블록 1 셀프 어텐션", layer_input, "LayerNorm 결과", "셀프 어텐션 출력", block1_result, "셀프 어텐션",
            trace.ENCODER_SELF_STEPS,
        )
        self.self_attention.set_log_labels("LayerNorm 결과", "LayerNorm 결과", "셀프 어텐션 출력")
        self.feed_forward_connection.set_log_labels(
            f"{where} / 블록 2 피드포워드", block1_result, "LayerNorm 결과", "피드포워드 출력", f"인코더 레이어 {n}의 최종 결과", "피드포워드",
            trace.FEED_FORWARD_STEPS,
        )
        self.feed_forward.set_log_labels("LayerNorm 결과", "피드포워드 출력")
