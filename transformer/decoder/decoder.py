import arr_fun as nn

from help_func.clones import clones
from transformer.blocks.layer_norm import LayerNorm
from help_func import trace
from help_func.trace import fmt, log, to_array


class Decoder(nn.Module):
    """DecoderLayer를 num_layers개 쌓아서, 지금까지 만든 출력 문장의 단어 벡터들을 차례로 통과시킴
    각 층에서 인코더 결과(encoder_output)를 참고하고, 마지막에 LayerNorm으로 한 번 더 정리함
    """

    def __init__(self, decoder_layer, num_layers):
        super(Decoder, self).__init__()
        # 같은 구조의 DecoderLayer를 num_layers개 복사해서 쌓음
        self.layers = clones(decoder_layer, num_layers)
        # 모든 층을 지난 뒤 마지막으로 값을 정리하는 LayerNorm
        self.final_norm = LayerNorm(decoder_layer.model_dim)
        # 로그 제목에 몇 번째 레이어인지 보이게 번호를 붙여요. 계산에는 영향이 없어요.
        for number, layer in enumerate(self.layers, start=1):
            layer.layer_number = number

    def forward(self, target_hidden, encoder_output, source_mask, target_mask):
        # 단어 벡터들을 첫 번째 층부터 마지막 층까지 차례로 통과시킴
        for layer in self.layers:
            if trace.LOG and layer.layer_number > 1:
                follow = trace.FOLLOW if trace.FOLLOW < target_hidden.size(1) else 0
                me = trace.word_name(follow)
                n = layer.layer_number
                hidden = to_array(target_hidden)[0]
                trace.section(f"디코더 레이어 {n}로 넘어감", "decoder.py")
                log(f"  한 줄 요약: 디코더 레이어 {n - 1}의 마지막 결과를 디코더 레이어 {n}에 그대로 넣어요. 새로 계산하는 건 없어요.")
                log("  왜: 같은 구조의 레이어를 여러 번 쌓아서, 단어 정보를 여러 번 다듬으려고요.")
                log(f"  디코더 레이어 {n} 입력 = 디코더 레이어 {n - 1}의 최종 결과({trace.name_of(f'디코더 레이어 {n - 1}의 최종 결과', me)}) = {fmt(hidden[follow])}", level=trace.SIMPLE)
            target_hidden = layer(target_hidden, encoder_output, source_mask, target_mask)
        # 원래 한 줄: return self.final_norm(target_hidden)
        # 로그 이름을 붙이려고 나눴고 계산은 같아요.
        self.final_norm.input_label = f"디코더 레이어 {len(self.layers)}의 최종 결과"
        self.final_norm.output_label = "디코더 최종 결과"
        with trace.place("디코더 최종 정규화 (decoder.py)"):
            output = self.final_norm(target_hidden)
        return output
