import math
import os
import sys

# 이 파일을 실행 버튼으로 바로 돌려도 프로젝트 폴더를 찾을 수 있게, 프로젝트 폴더를 경로에 넣어요.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arr_fun as torch
from arr_fun.tensor import Tensor
from help_func import trace
from help_func.trace import log


class SimpleLossCompute:
    """Annotated Transformer의 SimpleLossCompute를 옮긴 것이에요.
    디코더 출력을 Generator로 단어 확률로 바꾸고, 정답 단어를 얼마나 못 맞혔는지 손실을 계산해요.
    원본은 criterion으로 LabelSmoothing을 쓰는데, smoothing이 0이면 아래처럼
    '정답 토큰 로그확률에 마이너스를 붙여서 다 더한 값'과 같아요. 그래서 그 계산을 직접 썼어요.
    """

    def __init__(self, generator, pad_id):
        self.generator = generator
        self.pad_id = pad_id

    def __call__(self, decoder_output, target_tokens, num_target_tokens):
        log("Generator로 디코더 출력을 단어장 토큰마다 로그확률로 바꿔요. (Generator 안쪽 로그는 길어서 꺼 둬요.)")
        with trace.quiet():
            log_probs = self.generator(decoder_output)
        log(f"    log_probs 모양 {tuple(log_probs.size())}: 문장 수, 자리 수, 단어장 토큰 수 순서예요.")

        log("자리마다 정답 토큰의 로그확률을 꺼내서 마이너스를 붙이면 그 자리의 손실이에요.")
        log("    정답 확률이 100%면 손실이 0이고, 확률이 낮을수록 손실이 커져요. 패딩 자리는 빼요.")
        log_prob_rows = log_probs.numpy().tolist()
        target_rows = target_tokens.numpy().tolist()
        loss_sum = 0.0
        for sentence_index, (sentence_log_probs, sentence_targets) in enumerate(zip(log_prob_rows, target_rows)):
            pieces = []
            for position_log_probs, target_token in zip(sentence_log_probs, sentence_targets):
                target_token = int(target_token)
                if target_token == self.pad_id:
                    pieces.append(f"{target_token}(패딩이라 뺌)")
                    continue
                token_loss = -position_log_probs[target_token]
                loss_sum += token_loss
                pieces.append(f"{target_token}(확률 {math.exp(-token_loss) * 100:.0f}%, 손실 {token_loss:.2f})")
            log(f"    문장 {sentence_index + 1} 정답 토큰: " + "  ".join(pieces))
        log(f"    손실 합계 = {loss_sum:.2f}")

        log("손실 합계를 정답 토큰 수로 나눠서, 토큰 하나당 평균 손실로 만들어요.")
        log("    이렇게 나눠야 문장이 길든 짧든 같은 기준으로 비교할 수 있고, 가중치를 고치는 크기도 일정해져요.")
        loss_per_token = Tensor(loss_sum / num_target_tokens)
        log(f"    {loss_sum:.2f} / {num_target_tokens} = {float(loss_per_token):.4f}")

        # 원본처럼 (손실 합계, 역전파에 쓸 평균 손실) 두 개를 돌려줘요.
        return loss_sum, loss_per_token


def loss_for_correct_score(correct_score, criterion):
    """원본의 loss(x, crit) 예요. penalization_visualization에서 써요.
    정답(1번 단어) 점수를 correct_score, 나머지 단어 점수를 1로 두고 확률로 바꾼 뒤, criterion으로 손실을 구해요.
    correct_score가 클수록 모델이 정답을 더 확신하는 거예요.
    원본 이름: correct_score는 x, criterion은 crit, denominator는 d, predicted_probs는 predict예요.
    """
    log(f"[1] 분모 = 정답 점수 + 나머지 단어 3개 점수(1씩) = {correct_score} + 3 * 1. 점수를 다 더한 값으로 나눠야 확률 합이 1이 돼요.")
    denominator = correct_score + 3 * 1
    log(f"    1번 결과_분모 = {denominator}")

    log("[2] 예측 확률을 만들어요. 0번(패딩) 칸은 0, 1번(정답) 칸은 정답 점수 / 분모, 나머지는 1 / 분모예요.")
    predicted_probs = torch.FloatTensor([[0, correct_score / denominator, 1 / denominator, 1 / denominator, 1 / denominator]])
    log(f"    1번 칸 = {correct_score} / {denominator} = {correct_score / denominator:.4f},  나머지 = 1 / {denominator} = {1 / denominator:.4f}")
    log("    2번 결과_예측확률:")
    log(predicted_probs)

    log("[3] 예측 확률에 로그를 취해서 criterion(LabelSmoothing)에 넣어요. 정답 단어는 1번이에요.")
    loss = criterion(predicted_probs.log(), torch.LongTensor([1])).data
    log(f"    3번 결과_손실 = {float(loss):.4f}")
    return loss


def main():
    from train_func.label_smoothing import LabelSmoothing
    from transformer.model.generator import Generator

    # 실행할 때마다 같은 숫자가 나오게 고정해요.
    torch.manual_seed(0)

    log("[예제 1] SimpleLossCompute: 단어장 7개, 패딩 0번, 디코더 출력 4칸짜리 Generator로 손실을 구해요.")
    pad_id = 0
    generator = Generator(model_dim=4, vocab_size=7)
    loss_compute = SimpleLossCompute(generator, pad_id)
    log("    문장 1개, 자리 3개예요. 마지막 자리는 정답이 0(패딩)이라 손실에서 빠져요.")
    decoder_output = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]])
    target_tokens = torch.LongTensor([[3, 5, 0]])
    loss_sum, loss_per_token = loss_compute(decoder_output, target_tokens, num_target_tokens=2)
    log(f"    예제 1 결과: 손실 합계 {loss_sum:.4f}, 토큰 하나당 {float(loss_per_token):.4f}")

    log("")
    log("[예제 2] loss_for_correct_score: smoothing 0.1인 LabelSmoothing으로 정답 점수 5일 때 손실을 구해요.")
    criterion = LabelSmoothing(vocab_size=5, pad_id=0, smoothing=0.1)
    loss = loss_for_correct_score(5, criterion)
    log(f"    예제 2 결과: 손실 {float(loss):.4f}")


if __name__ == "__main__":
    main()
