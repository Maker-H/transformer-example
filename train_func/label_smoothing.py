import math
import os
import sys

# 이 파일을 실행 버튼으로 바로 돌려도 프로젝트 폴더를 찾을 수 있게, 프로젝트 폴더를 경로에 넣어요.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arr_fun as torch
from arr_fun.functional import kl_div
from arr_fun.modules import Module
from help_func import trace
from help_func.helper import show_example
from help_func.trace import log


class LabelSmoothing(Module):
    """Annotated Transformer의 LabelSmoothing을 옮긴 것이에요.
    정답을 '정답 칸 100%, 나머지 0%' 로 주지 않고, 정답 칸에 confidence(= 1 - smoothing)를 주고
    남은 smoothing을 다른 칸에 고르게 나눠 줘요. 모델이 정답 하나만 100% 믿지 않게 하려는 거예요.
    손실은 KL divergence(두 분포가 얼마나 다른지)를 모두 더한 값이에요. (원본: nn.KLDivLoss(reduction="sum"))
    원본 이름: vocab_size는 size, pad_id는 padding_idx예요.
    """

    def __init__(self, vocab_size, pad_id, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # 원본의 self.criterion = nn.KLDivLoss(reduction="sum") 자리예요. 같은 계산을 하는 kl_div를 써요.
        self.criterion = lambda predicted_log_probs, target_probs: kl_div(predicted_log_probs, target_probs, reduction="sum")
        self.pad_id = pad_id
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        # 마지막으로 만든 정답 분포를 남겨 둬요. 원본과 같은 이름(true_dist)이라 예제에서 꺼내 볼 수 있어요.
        self.true_dist = None

    def forward(self, predicted_log_probs, target_ids):
        """원본 이름: predicted_log_probs는 x, target_ids는 target, true_distribution은 true_dist예요."""
        row_count = predicted_log_probs.size(0)
        follow_row = trace.FOLLOW if trace.FOLLOW < row_count else 0
        target_list = [int(target_id) for target_id in target_ids.numpy().tolist()]

        log("")
        log(f"[1] 크기 확인: 예측의 칸 수가 단어장 크기 {self.vocab_size}와 같은지 봐요. 칸 하나가 단어 하나라서 같아야 해요.")
        assert predicted_log_probs.size(1) == self.vocab_size
        log(f"    예측 모양 {tuple(predicted_log_probs.size())}: 자리 {row_count}개, 자리마다 단어 {self.vocab_size}칸이에요. 정답 단어는 {target_list}예요.")

        log(f"[2] 정답 분포를 담을 표를 예측과 똑같은 모양으로 복사해서 만들어요. 값은 다음 단계에서 전부 덮어써요.")
        true_distribution = predicted_log_probs.data.clone()
        log(f"    2번 결과_표 모양 {tuple(true_distribution.size())}")

        fill_value = self.smoothing / (self.vocab_size - 2)
        log(f"[3] 모든 칸을 smoothing / (단어장 크기 - 2) 로 채워요. 정답 칸과 패딩 칸 2개를 빼고 나머지 칸에 smoothing을 고르게 나누는 거예요.")
        true_distribution.fill_(fill_value)
        log(f"    {self.smoothing} / ({self.vocab_size} - 2) = {fill_value:.4f}")
        log("    3번 결과_정답분포:")
        log(true_distribution)

        log(f"[4] 자리마다 정답 단어 칸에 confidence = 1 - {self.smoothing} = {self.confidence:.2f} 를 넣어요. (scatter_)")
        log("    정답에 가장 큰 확률을 주되 100%는 주지 않아요.")
        true_distribution.scatter_(1, target_ids.data.unsqueeze(1), self.confidence)
        log(f"    {follow_row}번 줄은 정답이 {target_list[follow_row]}번 단어라서 {target_list[follow_row]}번째 칸 = {self.confidence:.2f}")
        log("    4번 결과_정답분포:")
        log(true_distribution)

        log(f"[5] 패딩 단어({self.pad_id}번) 칸은 모든 줄에서 0으로 만들어요. 패딩은 진짜 단어가 아니라서 정답 후보가 될 수 없어요.")
        true_distribution[:, self.pad_id] = 0
        log("    5번 결과_정답분포:")
        log(true_distribution)

        log(f"[6] 정답 자체가 패딩({self.pad_id})인 줄을 찾아요. 그 자리는 문장 길이를 맞추려고 채운 빈칸이라 손실에서 빼야 해요.")
        pad_row_positions = torch.nonzero(target_ids.data == self.pad_id)  # 원본 이름: mask
        pad_row_list = [int(position) for position in pad_row_positions.numpy().reshape(-1).tolist()]
        log(f"    6번 결과_패딩줄 = {pad_row_list}  (정답이 {self.pad_id}인 줄 번호)")

        log("[7] 6번 결과_패딩줄에 있는 줄을 통째로 0으로 채워요. (index_fill_) 정답분포가 전부 0이면 그 줄의 손실도 0이 돼요.")
        if pad_row_positions.dim() > 0:
            true_distribution.index_fill_(0, pad_row_positions.squeeze(), 0.0)
        log("    7번 결과_정답분포 (최종):")
        log(true_distribution)
        self.true_dist = true_distribution

        log("[8] KL divergence 손실: 칸마다 정답확률 * (log 정답확률 - 예측 로그확률) 을 구해서 모두 더해요.")
        log("    정답분포와 예측이 똑같으면 log 두 개가 같아서 0이 되고, 다를수록 커져요. 정답확률이 0인 칸은 0으로 쳐요.")
        loss = self.criterion(predicted_log_probs, true_distribution.clone().detach())
        true_rows = true_distribution.numpy().tolist()
        log_prob_rows = predicted_log_probs.numpy().tolist()
        row_sums = []
        for row_index, (true_row, log_prob_row) in enumerate(zip(true_rows, log_prob_rows)):
            row_sum = 0.0
            for column, (true_prob, log_prob) in enumerate(zip(true_row, log_prob_row)):
                if true_prob > 0:
                    row_sum += true_prob * (math.log(true_prob) - log_prob)
            row_sums.append(row_sum)
        follow_true_row = true_rows[follow_row]
        follow_log_prob_row = log_prob_rows[follow_row]
        log(f"    {follow_row}번 줄을 칸마다 따라가 봐요. (정답 {target_list[follow_row]})")
        for column, (true_prob, log_prob) in enumerate(zip(follow_true_row, follow_log_prob_row)):
            if true_prob > 0:
                term = true_prob * (math.log(true_prob) - log_prob)
                log(f"      {column}번째 칸: {true_prob:.4f} * (log {true_prob:.4f} - ({log_prob:.4f})) = {true_prob:.4f} * ({math.log(true_prob):.4f} - ({log_prob:.4f})) = {term:.4f}")
            else:
                log(f"      {column}번째 칸: 정답확률이 0이라서 0")
        log(f"    {follow_row}번 줄 합 = {row_sums[follow_row]:.4f}")
        log("    줄마다 합: " + ",  ".join(f"{row_index}번 줄 {row_sum:.4f}" for row_index, row_sum in enumerate(row_sums)))
        log(f"    8번 결과_손실 = 줄마다 합을 다 더한 값 = {float(loss):.4f}")
        if math.isinf(float(loss)):
            log("    손실이 무한대(inf)예요. 예측 확률이 0인 칸(로그가 -inf)에 정답분포가 0보다 큰 값을 줬기 때문이에요.")
        return loss


def example_label_smoothing():
    """원본 예제예요. smoothing 0.4로 정답분포가 어떻게 생기는지 봐요.
    원본은 정답분포를 색깔 그림(altair)으로 그리는데, 여기서는 표 로그로 보여줘요.
    """
    log("[예제] LabelSmoothing(단어장 5, 패딩 0, smoothing 0.4) 를 만들어요.")
    criterion = LabelSmoothing(vocab_size=5, pad_id=0, smoothing=0.4)  # 원본 이름: crit

    log("[예제] 자리 5개 모두 같은 예측 확률 [0, 0.2, 0.7, 0.1, 0] 을 써요. 로그를 취해서 넣어요.")
    predicted_probs = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )  # 원본 이름: predict
    target_ids = torch.LongTensor([2, 1, 0, 3, 3])
    log(f"[예제] 자리마다 정답 단어는 {[int(t) for t in target_ids.numpy().tolist()]} 예요. 2번 줄은 정답이 0(패딩)이에요.")
    criterion(predicted_probs.log(), target_ids)
    log("[예제] 손실이 inf로 나온 건 예측에 0이 있어서 그래요. 원본 예제도 손실 값은 쓰지 않고, 아래 정답분포만 그림으로 그려요.")

    log("")
    log("[예제 결과] 원본이 그림으로 그리는 criterion.true_dist 표예요. 줄은 자리, 칸은 단어 번호예요.")
    log("    정답 칸은 0.60, 나머지는 0.13, 패딩 칸(0번)과 정답이 패딩인 2번 줄은 0이에요.")
    log(criterion.true_dist)
    return criterion.true_dist


def penalization_visualization():
    """원본 예제예요. 모델이 정답을 너무 확신하면 label smoothing이 오히려 손실을 키워서 벌을 준다는 걸 봐요.
    원본은 pandas와 altair로 선 그래프를 그리는데, 여기서는 Steps와 Loss를 표 로그로 보여줘요.
    """
    from train_func.loss_compute import loss_for_correct_score

    log("")
    log("[벌점 예제 1] LabelSmoothing(단어장 5, 패딩 0, smoothing 0.1) 을 만들어요. 정답은 1번 단어예요.")
    criterion = LabelSmoothing(vocab_size=5, pad_id=0, smoothing=0.1)  # 원본 이름: crit
    log("    그러면 정답분포는 [0, 0.90, 0.0333, 0.0333, 0.0333] 이 돼요.")

    log("[벌점 예제 2] 정답 점수 correct_score가 3일 때 하나만 처음부터 끝까지 자세히 계산해 봐요.")
    detailed_loss = loss_for_correct_score(3, criterion)
    log(f"    2번 결과_손실(correct_score 3) = {float(detailed_loss):.4f}")

    log("")
    log("[벌점 예제 3] correct_score를 1부터 99까지 바꾸며 손실을 구해요. 99번 다 찍으면 너무 길어서 로그는 꺼 두고 계산만 해요.")
    with trace.quiet():
        loss_values = [float(loss_for_correct_score(correct_score, criterion)) for correct_score in range(1, 100)]
    steps = list(range(99))
    log(f"    3번 결과_손실 {len(loss_values)}개를 구했어요. Steps 0이 correct_score 1이에요.")

    log("[벌점 예제 4] 원본 그래프 대신 표로 봐요. 막대(#)가 길수록 손실이 커요. 작은 값도 보이게 막대 길이는 손실의 제곱근에 맞췄어요.")
    lowest_step = min(steps, key=lambda step: loss_values[step])
    shown_steps = sorted(set([0, 1, 2, 4, 9, 14, 19, 24, lowest_step, 29, 39, 49, 69, 98]))
    largest_loss = max(loss_values)
    log("    Steps | correct_score | 정답 예측확률 | Loss")
    for step in shown_steps:
        correct_score = step + 1
        correct_prob = correct_score / (correct_score + 3)
        bar = "#" * round(math.sqrt(loss_values[step] / largest_loss) * 30)
        mark = "   (가장 작음)" if step == lowest_step else ""
        log(f"    {step:5d} | {correct_score:13d} | {correct_prob:12.4f} | {loss_values[step]:.4f} {bar}{mark}")

    log("[벌점 예제 5] 왜 손실이 다시 커질까요?")
    log(f"    정답분포는 정답 칸에 0.90만 줘요. correct_score가 {lowest_step + 1}일 때 예측확률도 {(lowest_step + 1) / (lowest_step + 4):.2f}이라 정답분포와 똑같아져서 손실이 가장 작아요.")
    log("    그보다 더 확신하면 다른 칸 예측이 0.0333보다 작아져서 손실이 다시 커져요. 너무 확신하는 모델에 벌을 주는 거예요.")
    return steps, loss_values


def main():
    # 원본처럼 show_example로 예제를 돌려요. 이 파일을 직접 실행할 때만 돌아가요.
    show_example(example_label_smoothing)
    show_example(penalization_visualization)


if __name__ == "__main__":
    main()
