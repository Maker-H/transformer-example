import arr_fun as nn

import math

from help_func import trace
from help_func.trace import fmt, log, to_array


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        # 로그에서 결과를 부르는 이름이에요. 계산에는 영향이 없어요.
        self.output_label = "임베딩"

    def forward(self, token_ids):
        # 토큰 번호로 표 lut의 몇 번째 줄을 꺼내는지, sqrt(d_model)을 곱하는 걸 보여줘요.
        # 설명을 먼저 찍고, 그 줄을 계산하고, 결과를 바로 찍어요.
        root = math.sqrt(self.d_model)
        if trace.LOG:
            ids = to_array(token_ids).reshape(-1)
            f = trace.FOLLOW if trace.FOLLOW < len(ids) else 0
            me = trace.word_name(f)
            table = to_array(self.lut.weight)
            trace.section(f"임베딩: 토큰 번호로 표 lut에서 줄을 꺼내고 sqrt({self.d_model}) = {root:.2f} 곱하기", "embeddings.py")
            log(f"  한 줄 요약: 토큰 번호를 표 lut에서 찾아서, 숫자 {self.d_model}칸짜리 단어 벡터로 바꿔요.")
            log(f"  왜: 모델은 번호 자체로는 계산할 수 없어서, 단어마다 학습되는 숫자 벡터로 바꿔서 써요. sqrt({self.d_model})를 곱하는 건 다음 단계에서 더할 위치 값보다 너무 작아지지 않게 크기를 맞추려는 거예요.")
            log(f"  표 lut 모양: {table.shape[0]}줄 x {table.shape[1]}칸 (단어장 단어마다 한 줄)")
            log("  단어마다 토큰 번호: " + ",  ".join(f"{trace.word_label(trace.word_name(i))} = 토큰 {int(t)}번" for i, t in enumerate(ids)))

        # 원래 한 줄: return self.lut(token_ids) * math.sqrt(self.d_model)
        # 표에서 꺼낸 값을 로그로 보려고 나눴고 계산은 같아요.
        looked_up = self.lut(token_ids)
        if trace.LOG:
            rows = to_array(looked_up).reshape(len(ids), -1)
            log(f"  {trace.word_label(me)}는 토큰 {int(ids[f])}번이라 lut {int(ids[f])}번째 줄을 꺼냄: {fmt(rows[f])}")

        output = looked_up * math.sqrt(self.d_model)
        if trace.LOG:
            out = to_array(output).reshape(len(ids), -1)
            terms = ", ".join(f"{a:.2f}*{root:.2f}={b:.2f}" for a, b in zip(rows[f], out[f]))
            log(f"    칸마다: {terms}")
            log(f"  {trace.result(self.output_label, me)} = {fmt(out[f])}")
            others = [f"{trace.result(self.output_label, trace.word_name(i))} = {fmt(out[i])}" for i in range(len(ids)) if i != f]
            if others:
                log("  다른 단어: " + ",  ".join(others))
        return output
