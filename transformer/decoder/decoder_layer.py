import arr_fun as nn

from help_func.clones import clones
from transformer.blocks.sublayer_connection import SublayerConnection
from help_func import trace
from help_func.trace import fmt, log, to_array


class DecoderLayer(nn.Module):
    """디코더 층 하나로, self attention, source attention, feed forward를 차례로 함
    self attention으로 출력 문장의 단어들끼리 참고하되, target_mask로 아직 안 나온 뒤 단어는 못 보게 함
    source attention으로 인코더 결과(입력 문장)를 참고하고, feed forward로 단어마다 따로 가공함
    세 단계 모두 SublayerConnection으로 감싸서 결과를 원래 입력에 누적함
    """

    def __init__(self, model_dim, self_attention, source_attention, feed_forward, dropout_rate):
        super(DecoderLayer, self).__init__()
        # Decoder가 마지막 LayerNorm 크기를 정할 때 씀
        self.model_dim = model_dim
        # 출력 문장의 단어들끼리 참고하는 MultiHeadedAttention
        self.self_attention = self_attention
        # 출력 문장의 단어가 입력 문장(인코더 결과)을 참고하는 MultiHeadedAttention (원본 코드의 src_attn)
        self.source_attention = source_attention
        # 단어마다 따로 가공하는 PositionwiseFeedForward
        self.feed_forward = feed_forward
        # self attention용, source attention용, feed forward용 SublayerConnection 3개
        connections = clones(SublayerConnection(model_dim, dropout_rate), 3)
        self.self_attention_connection = connections[0]
        self.source_attention_connection = connections[1]
        self.feed_forward_connection = connections[2]
        # 몇 번째 디코더 레이어인지예요. Decoder가 1, 2, ... 로 정해 줘요. 로그 제목에만 써요.
        self.layer_number = 1

    def forward(self, target_hidden, encoder_output, source_mask, target_mask):
        if trace.LOG:
            self._set_log_labels()
            log("")
            log(f"[디코더 레이어 {self.layer_number} 시작] 디코더 레이어 하나는 아래 세 묶음을 차례로 해요.")
            log("    셀프 어텐션 블록: LayerNorm, 멀티헤드 어텐션(디코더 단어끼리, 뒤 단어는 마스크로 가림), 잔차 더하기", level=trace.DETAIL)
            log("    소스 어텐션 블록: LayerNorm, 멀티헤드 어텐션(Q는 디코더 단어, K와 V는 인코더 결과 memory), 잔차 더하기", level=trace.DETAIL)
            log("    피드포워드 블록: LayerNorm, 피드포워드(넓히기, ReLU, 줄이기), 잔차 더하기", level=trace.DETAIL)
        # 1) self attention: query, key, value 모두 출력 문장을 쓰고, target_mask로 뒤 단어를 가림
        target_hidden = self.self_attention_connection(
            target_hidden,
            lambda normed_target: self.self_attention(normed_target, normed_target, normed_target, target_mask),
        )
        # 2) source attention: query는 출력 문장, key와 value는 인코더 결과를 씀
        #    계산하기 전에 Q와 K, V가 어디서 오는지 먼저 보여줘요.
        if trace.LOG:
            n = self.layer_number
            me = trace.word_name(trace.FOLLOW)
            block1_result = f"디코더 레이어 {n} 블록 1(마스크 셀프 어텐션) 결과"
            first_kv = trace.WORD_NAMES[0] if trace.WORD_NAMES else 0
            with trace.place(self.source_attention_connection.log_title, self.source_attention_connection.log_steps):
                trace.section("재료 준비: Q는 블록 1 결과에서, K와 V는 인코더 결과(memory)에서", "decoder_layer.py")
                log("  한 줄 요약: 이제 디코더 단어가 입력 문장(인코더 결과)을 참고하는 소스 어텐션 블록을 시작해요. 여기서는 계산 없이 재료가 어디서 오는지만 보여 줘요.")
                log("  왜: 디코더가 다음 단어를 고르려면, 번역할 원래 문장에 어떤 단어가 있었는지 봐야 해요.")
                log(f"  Q 재료 = 디코더 레이어 {n} 블록 1(마스크 셀프 어텐션)의 결과({trace.name_of(block1_result, me)}) = {fmt(to_array(target_hidden)[0][trace.FOLLOW])}", level=trace.SIMPLE)
                log(f"  K, V 재료 = memory = 인코더 최종 결과({trace.name_of('memory, 인코더 최종 결과', first_kv)} 등 입력 문장 단어 {len(to_array(encoder_output)[0])}개)", level=trace.SIMPLE)
                log(f"  Q는 디코더 쪽 {trace.word_label(me)}로 만들어요. {trace.ref(block1_result, me)}를 다음 단계에서 LayerNorm으로 정리한 값이에요.")
                log("  K와 V는 인코더가 만든 memory로 만들어요. 그래서 입력 문장 단어를 참고할 수 있어요.")
                for index, row in enumerate(to_array(encoder_output)[0]):
                    name = trace.WORD_NAMES[index] if index < len(trace.WORD_NAMES) else index
                    log(f"    {trace.ref('memory, 인코더 최종 결과', name)} = {fmt(row)}")
        # 로그에서 K, V 쪽 단어 이름을 입력 문장 단어(A, B, C)로 바꿔 둬요. 계산에는 영향이 없어요.
        saved_kv_names = trace.KV_NAMES
        trace.KV_NAMES = trace.WORD_NAMES
        target_hidden = self.source_attention_connection(
            target_hidden,
            lambda normed_target: self.source_attention(normed_target, encoder_output, encoder_output, source_mask),
        )
        trace.KV_NAMES = saved_kv_names
        # 3) feed forward
        return self.feed_forward_connection(target_hidden, self.feed_forward)

    def _set_log_labels(self):
        """로그에서 값마다 붙일 이름을 정해요. 계산에는 영향이 없어요."""
        n = self.layer_number
        where = f"디코더 레이어 {n} (decoder_layer.py)"
        block1_result = f"디코더 레이어 {n} 블록 1(마스크 셀프 어텐션) 결과"
        block2_result = f"디코더 레이어 {n} 블록 2(소스 어텐션) 결과"
        # 값 이름은 로그에서 "47번 결과_시작 (셀프 어텐션 출력)"처럼, 그 값을 만든 단계 번호와 역할로 바뀌어 찍혀요.
        # 레이어 1의 입력은 위치 정보를 더한 값이고, 레이어 2부터는 앞 레이어의 결과예요.
        layer_input = "디코더 레이어 1 입력" if n == 1 else f"디코더 레이어 {n - 1}의 최종 결과"
        self.self_attention_connection.set_log_labels(
            f"{where} / 블록 1 마스크 셀프 어텐션", layer_input, "LayerNorm 결과", "셀프 어텐션 출력", block1_result, "셀프 어텐션",
            trace.DECODER_SELF_STEPS,
        )
        self.self_attention.set_log_labels("LayerNorm 결과", "LayerNorm 결과", "셀프 어텐션 출력")
        self.source_attention_connection.set_log_labels(
            f"{where} / 블록 2 소스 어텐션", block1_result, "LayerNorm 결과", "소스 어텐션 출력", block2_result, "소스 어텐션",
            trace.SOURCE_STEPS,
        )
        self.source_attention.set_log_labels("LayerNorm 결과", "memory, 인코더 최종 결과", "소스 어텐션 출력", source=True)
        self.feed_forward_connection.set_log_labels(
            f"{where} / 블록 3 피드포워드", block2_result, "LayerNorm 결과", "피드포워드 출력", f"디코더 레이어 {n}의 최종 결과", "피드포워드",
            trace.FEED_FORWARD_STEPS,
        )
        self.feed_forward.set_log_labels("LayerNorm 결과", "피드포워드 출력")
