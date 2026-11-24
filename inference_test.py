import arr_fun as torch

from help_func.helper import show_example
from help_func import trace
from help_func.trace import fmt, log, to_array
from transformer.model.make_model import make_model
from transformer.attention.subsequent_mask import subsequent_mask

def inference_test():
    # 원래 줄: test_model = make_model(11, 11, 2)
    # 기본 크기(model_dim=512, hidden_dim=2048, num_heads=8)는 숫자가 너무 많아서 로그로 볼 수가 없어요.
    # 그래서 단어 벡터 4칸, FF 은닉 8칸, head 2개, 레이어 2개로 아주 작게 만들어요. make_model의 기본값은 그대로예요.
    torch.manual_seed(0)  # 실행할 때마다 같은 숫자가 나오게 고정
    test_model = make_model(11, 11, num_layers=2, model_dim=4, hidden_dim=8, num_heads=2)
    test_model.eval()

    # 원래 줄: src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 원래 줄: src_mask = torch.ones(1, 1, 10)
    # 단어가 10개면 로그가 너무 길어서 3개(토큰 1, 2, 3번)만 써요. 로그에서는 단어A, B, C라고 불러요.
    src = torch.LongTensor([[1, 2, 3]])
    src_mask = torch.ones(1, 1, 3)

    trace.COMPACT = True
    trace.WORD_NAMES = ["A", "B", "C"]
    trace.FOLLOW = 0
    log("[입력] 입력 문장 토큰 [1, 2, 3] 을 단어A, B, C라고 불러요. 아래 로그는 단어A 하나를 따라가요.")
    log("    모델 크기: 단어 벡터 4칸, head 2개 (head마다 2칸), FF 은닉 8칸, 인코더 레이어 2개, 디코더 레이어 2개, 단어장 11개")
    log("    src_mask는 전부 1이라 인코더에서는 가리는 단어가 없어요.")

    memory = test_model.encode(src, src_mask)
    generated_tokens = torch.zeros(1, 1).type_as(src)

    log("")
    log("[디코딩 시작] 디코더는 토큰 0번(시작)에서 출발해요. 지금 디코더 단어는 '시작' 1개예요.")
    log("    첫 번째 다음 단어를 고르는 과정만 자세히 찍고, 그 뒤 단계는 한 줄씩만 찍어요.")

    for step in range(9):
        decoder_output = test_model.decode(
            memory, src_mask, generated_tokens, subsequent_mask(generated_tokens.size(1)).type_as(src.data)
        )

        next_word_log_probs = test_model.generator(decoder_output[:, -1])
        if trace.LOG:
            trace.section("다음 단어 고르기: 로그확률이 가장 큰 토큰", "inference_test.py")
            log(f"  한 줄 요약: {trace.ref('로그확률', trace.word_name(0))}에서 값이 가장 큰 토큰을 다음 단어로 골라요.")
            log("  왜: 모델이 가장 그럴듯하다고 본 토큰 하나를 고르는 가장 간단한 방법이에요. 고른 토큰을 문장 뒤에 붙이고 디코더를 다시 돌려요.")
        _, next_word = torch.max(next_word_log_probs, dim=1)
        next_word = next_word.data[0]
        if trace.LOG:
            best = to_array(next_word_log_probs)[0]
            log(f"  torch.max로 가장 큰 값을 찾음: 토큰{int(next_word)} ({best[int(next_word)]:.2f})")
            log(f"  그래서 다음 단어 = 토큰 {int(next_word)}번", level=trace.SIMPLE)
        generated_tokens = torch.cat(
            [generated_tokens, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if step == 0:
            # 자세한 로그는 첫 단계만 찍고 여기서 꺼요. 계산에는 영향이 없어요.
            trace.LOG = False
            # 아래 세 줄은 trace.LOG를 끈 뒤에도 보여야 해서 log 대신 print로 찍어요.
            print("")
            print("[나머지 디코딩 단계] 같은 계산을 반복해요. 단계마다 고른 토큰만 한 줄씩 찍어요.")
        note = "   (위에서 자세히 본 단계)" if step == 0 else ""
        print(f"    {step + 1}단계: 토큰 {int(next_word)}번을 골라서 지금까지 문장 = {to_array(generated_tokens)[0].tolist()}{note}")

    trace.LOG = True
    print("example untrained model prediction:", generated_tokens)

def run_tests():
    # 원래는 range(10)으로 10번 돌렸어요. 로그를 한 번만 보려고 1번만 돌려요.
    for _ in range(1):
        inference_test()

if __name__ == "__main__":
    show_example(run_tests)
