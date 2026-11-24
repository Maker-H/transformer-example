
import numpy as np

from arr_fun.tensor import Tensor, _tensor_to_array, _convert_negative_axis_to_positive, softmax


def log_softmax(scores, dim=-1):
    """
    softmax에 로그를 취한 값
    softmax는 숫자 목록을 다 더하면 1이 되는 확률로 바꿔주는 함수 
    """
    score_array = _tensor_to_array(scores)
    softmax_axis = _convert_negative_axis_to_positive(dim, score_array.ndim)

    scores_minus_max = score_array - score_array.max(axis=softmax_axis, keepdims=True)
    exp_scores = np.exp(scores_minus_max)
    return Tensor(
        # log(exp_scores)= scores_minus_max임
        # log(exp_scores / exp_scores.sum) 이라고 이해하면 됨
        _array=scores_minus_max - np.log(exp_scores.sum(axis=softmax_axis, keepdims=True))
    )


def relu(input_tensor):
    """음수는 0으로 바꾸고 양수는 그대로 두는 함수"""
    return Tensor(_array=np.maximum(_tensor_to_array(input_tensor), 0))


def kl_div(predicted_log_probs, target_probs, reduction="sum"):
    """KL divergence 손실이에요. torch.nn.KLDivLoss와 같은 계산이에요.
    두 확률 분포가 얼마나 다른지 재는 값이고, 같으면 0이에요.
    칸마다 정답확률 * (log 정답확률 - 예측 로그확률) 을 구하고, reduction에 따라 더하거나 평균 내요.
    predicted_log_probs는 예측 확률에 이미 로그를 취한 값이어야 해요.
    정답확률이 0인 칸은 0 * log 0 = 0 으로 정해서 0으로 둬요. (예측이 0이라 로그가 -inf여도 마찬가지예요.)
    """
    log_prob_array = _tensor_to_array(predicted_log_probs)
    target_array = _tensor_to_array(target_probs)
    # 0인 칸은 계산 중에 log 0, 0 * inf 경고가 나지만 어차피 0으로 바꿀 칸이라 경고를 꺼요.
    with np.errstate(divide="ignore", invalid="ignore"):
        cell_terms = np.where(target_array > 0, target_array * (np.log(target_array) - log_prob_array), 0.0)
    if reduction == "none":
        return Tensor(_array=cell_terms)
    if reduction == "sum":
        return Tensor(_array=np.array(cell_terms.sum()))
    if reduction == "mean":
        return Tensor(_array=np.array(cell_terms.mean()))
    if reduction == "batchmean":
        return Tensor(_array=np.array(cell_terms.sum() / cell_terms.shape[0]))
    raise ValueError("reduction은 none, sum, mean, batchmean 중 하나여야 해요.")
