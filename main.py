"""
MultiHeadedAttention까지 구성한 코드를 한 번 실행해 보는 진입점입니다.
VS Code에서 이 파일을 열고 실행하거나, 디버그 구성 "Python: main"을 쓰면 됩니다.
로그를 끄고 싶으면 help_func/trace.py의 LOG를 False로 바꾸면 됩니다.
"""

import arr_fun as torch

from transformer.attention.multi_headed_attention import MultiHeadedAttention
from help_func import trace
from help_func.trace import fmt, log, to_array


def main():
    # 적어 두신 예: 문장 1개, 단어 3개, dim_model=4, heads=2, dim_key=2
    torch.manual_seed(0)
    heads = 2
    dim_model = 4

    # 원래 값은 [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] 였어요.
    # 숫자가 크면 softmax가 한 단어에 100%를 몰아줘서 비율이 안 보여서, 10으로 나눈 값을 써요.
    query = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],  # 단어A
                [0.5, 0.6, 0.7, 0.8],  # 단어B
                [0.9, 1.0, 1.1, 1.2],  # 단어C
            ]
        ]
    )
    key = query.clone()
    value = query.clone()

    log("[입력] 단어 3개, 단어마다 숫자 4칸. 아래 로그는 단어A 하나를 끝까지 따라가요.")
    for name, row in zip("ABC", to_array(query)[0]):
        log(f"    x_{name} = {fmt(row)}")

    model = MultiHeadedAttention(heads, dim_model, dropout_rate=0.0)
    model.eval()

    log("    (Q, K, V 표는 clones 때문에 셋 다 같아서 Q, K, V 값도 같게 나와요.)")
    with torch.no_grad(), trace.place("멀티헤드 어텐션 (main.py)", trace.ATTENTION_ONLY_STEPS):
        output = model(query, key, value)

    log("")
    log("[끝] 단어A의 여정")
    log(f"    x_A = {fmt(to_array(query)[0][0])}  처음 입력")
    final_name = trace._short(trace.ref("멀티헤드 어텐션 출력", "A"))
    log(f"    {final_name} (멀티헤드 어텐션 출력) = {fmt(to_array(output)[0][0])}  B, C를 참고해서 새로 만든 값")


if __name__ == "__main__":
    main()
