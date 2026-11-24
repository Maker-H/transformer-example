import numpy as np

import arr_fun as nn
from arr_fun.functional import log_softmax
from help_func import trace
from help_func.trace import fmt, log, to_array


class Generator(nn.Module):
    """디코더 결과 벡터를 단어장 크기의 점수로 바꾼 뒤, log_softmax로 단어마다 확률(로그 값)을 구함
    이 확률이 가장 큰 단어가 다음에 올 단어로 뽑힘
    """

    def __init__(self, model_dim, vocab_size):
        super(Generator, self).__init__()
        # model_dim 칸 벡터를 단어장 단어 수(vocab_size)만큼의 점수로 바꾸는 Linear
        self.output_projection = nn.Linear(model_dim, vocab_size)
        # 로그에서 쓰는 이름이에요. 계산에는 영향이 없어요.
        self.output_projection.log_name = f"Generator output_projection (단어장 {vocab_size}개 토큰마다 점수 만들기)"
        self.output_projection.input_label = "디코더 최종 결과"
        self.output_projection.output_label = "단어장 점수"
        self.output_projection.log_about = [
            f"한 줄 요약: 디코더 최종 결과 {model_dim}칸에 학습 가중치 표 W를 곱해서, 단어장 {vocab_size}개 토큰마다 점수를 하나씩 만들어요.",
            "왜: 점수가 클수록 그 토큰이 다음 단어일 가능성이 커요. 다음 단계에서 이 점수를 확률로 바꿔요.",
        ]

    def forward(self, decoder_output):
        # 마지막 축 기준으로 log_softmax를 해서 단어마다 확률의 로그 값을 구함
        # 원래 한 줄: return log_softmax(self.output_projection(decoder_output), dim=-1)
        # 점수를 로그로 보려고 나눴고 계산은 같아요.
        # 로그에서 "다음 단어 점수" 블록으로 묶어서, 순서와 지금 위치를 보여줘요. 계산에는 영향이 없어요.
        with trace.place("다음 단어 점수 (generator.py)", trace.GENERATOR_STEPS):
            scores = self.output_projection(decoder_output)

            # 점수가 log_softmax로 어떻게 바뀌는지 토큰마다 보여줘요.
            # 설명(빼는 값 m, e^점수, log(합))을 먼저 찍고, log_softmax를 계산하고, 토큰마다 결과를 바로 찍어요.
            if trace.LOG:
                s = to_array(scores).reshape(-1, to_array(scores).shape[-1])[0]
                m = float(s.max())
                exps = np.exp(s - m)
                log_sum = float(np.log(exps.sum()))
                trace.section("log_softmax: 점수를 확률의 로그 값으로 바꿈", "generator.py / functional.py")
                me = trace.word_name(0)
                log(f"  한 줄 요약: {trace.ref(self.output_projection.output_label, me)}를 토큰마다 확률(의 로그 값)로 바꿔요. 확률은 다 더하면 100%예요.")
                log("  왜: 점수는 크기에 정해진 범위가 없어서, 합이 100%인 확률로 바꿔야 토큰끼리 비교하고 학습 때 손실을 계산하기 쉬워요.")
                log(f"  가장 큰 점수 m = {m:.2f}  (e^점수가 너무 커지지 않게 먼저 빼요. 결과는 같아요)")
                log("  e^(점수 - m): " + ", ".join(f"토큰{t} {v:.2f}" for t, v in enumerate(exps)))
                log(f"  log(합) = log({exps.sum():.2f}) = {log_sum:.2f}")
                log(f"  토큰마다 로그확률 = 점수 - m - log(합), 확률 = e^로그확률")
            log_probs = log_softmax(scores, dim=-1)
            if trace.LOG:
                lp = to_array(log_probs).reshape(-1, s.shape[0])[0]
                for t in range(s.shape[0]):
                    log(f"    토큰{t}: {s[t]:.2f} - {m:.2f} - {log_sum:.2f} = {lp[t]:.2f}   (확률 {np.exp(lp[t]) * 100:.1f}%)")
                top = np.argsort(-lp)[:3]
                log(f"  {trace.result('로그확률', me)} = 위 토큰마다 로그확률")
                log("  로그확률 큰 순서 3개: " + ",  ".join(f"토큰{t} {lp[t]:.2f}" for t in top), level=trace.SIMPLE)
            return log_probs
