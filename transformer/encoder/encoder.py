import arr_fun as nn

from help_func.clones import clones
from transformer.blocks.layer_norm import LayerNorm
from help_func import trace
from help_func.trace import fmt, log, to_array


class Encoder(nn.Module):
    """EncoderLayer를 num_layers개 쌓아서, 입력 문장의 단어 벡터들을 차례로 통과시킴
    마지막에 LayerNorm으로 한 번 더 정리해서, 디코더가 참고할 결과(memory)를 만듦
    """

    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        # 같은 구조의 EncoderLayer를 num_layers개 복사해서 쌓음
        self.layers = clones(encoder_layer, num_layers)
        # 모든 층을 지난 뒤 마지막으로 값을 정리하는 LayerNorm
        self.final_norm = LayerNorm(encoder_layer.model_dim)
        # 로그 제목에 몇 번째 레이어인지 보이게 번호를 붙여요. 계산에는 영향이 없어요.
        for number, layer in enumerate(self.layers, start=1):
            layer.layer_number = number

    def forward(self, source_hidden, source_mask):
        # 단어 벡터들을 첫 번째 층부터 마지막 층까지 차례로 통과시킴
        for layer in self.layers:
            if trace.LOG and layer.layer_number > 1:
                me = trace.word_name(trace.FOLLOW)
                n = layer.layer_number
                hidden = to_array(source_hidden)[0]
                trace.section(f"인코더 레이어 {n}로 넘어감", "encoder.py")
                log(f"  한 줄 요약: 인코더 레이어 {n - 1}의 마지막 결과를 인코더 레이어 {n}에 그대로 넣어요. 새로 계산하는 건 없어요.")
                log("  왜: 같은 구조의 레이어를 여러 번 쌓아서, 단어 정보를 여러 번 다듬으려고요.")
                log(f"  인코더 레이어 {n} 입력 = 인코더 레이어 {n - 1}의 최종 결과({trace.name_of(f'인코더 레이어 {n - 1}의 최종 결과', me)}) = {fmt(hidden[trace.FOLLOW])}", level=trace.SIMPLE)
            source_hidden = layer(source_hidden, source_mask)
        # 원래 한 줄: return self.final_norm(source_hidden)
        # 로그 이름을 붙이려고 나눴고 계산은 같아요.
        self.final_norm.input_label = f"인코더 레이어 {len(self.layers)}의 최종 결과"
        self.final_norm.output_label = "memory, 인코더 최종 결과"
        with trace.place("인코더 최종 정규화 (encoder.py)"):
            memory = self.final_norm(source_hidden)
        if trace.LOG:
            trace.section("인코더 끝: memory = 인코더 결과. 디코더가 source attention에서 이걸 참고해요", "encoder.py")
            log(f"  한 줄 요약: {trace.step_of('memory, 인코더 최종 결과')}번 단계에서 마지막으로 정리한 인코더 결과를 memory라고 불러요. 여기서 인코더가 끝났어요.")
            log("  왜: 디코더가 source attention에서 이 memory로 K와 V를 만들어서, 입력 문장의 단어들을 참고해요.")
            for index, row in enumerate(to_array(memory)[0]):
                me = trace.word_name(index)
                log(f"    memory_{me} = 인코더 최종 결과({trace.name_of('memory, 인코더 최종 결과', me)}) = {fmt(row)}", level=trace.SIMPLE if index == trace.FOLLOW else None)
        return memory
