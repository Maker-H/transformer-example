import os
import random
import sys
import time
from dataclasses import dataclass

# 이 파일을 실행 버튼으로 바로 돌려도 transformer 폴더를 찾을 수 있게, 프로젝트 폴더를 경로에 넣어요.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from help_func import trace
from help_func.helper import show_example
from help_func.trace import log


@dataclass
class TrainState:
    """지금까지 학습한 양을 세어 두는 기록장이에요. 이름은 Annotated Transformer 원본 그대로예요."""

    step: int = 0  # 학습한 묶음(batch) 수
    accum_step: int = 0  # 가중치를 실제로 고친 횟수 (optimizer.step을 부른 횟수)
    samples: int = 0  # 학습한 문장 수
    tokens: int = 0  # 학습한 정답 토큰 수 (패딩 제외)


def run_epoch(
    batches,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    gradient_accumulation_steps=1,
    train_state=None,
):
    """가진 묶음(batch)을 한 바퀴 다 돌면서 학습하거나 평가해요. 한 바퀴를 epoch이라고 불러요.
    원본 이름: batches는 data_iter, gradient_accumulation_steps는 accum_iter,
    optimizer_step_count는 n_accum, decoder_output은 out, loss_for_backward는 loss_node예요.
    """
    # 원본은 train_state=TrainState() 를 기본값으로 뒀는데, 그러면 모든 호출이 기록장 하나를 같이 써요.
    # 그래서 기본값은 None으로 두고, 없을 때만 새로 만들어요.
    if train_state is None:
        train_state = TrainState()
    is_training = mode == "train" or mode == "train+log"

    log("[준비] 시간을 재기 시작해요. 나중에 1초에 정답 토큰을 몇 개 처리했는지 계산할 때 써요.")
    start_time = time.time()
    total_loss = 0
    total_target_tokens = 0
    tokens_since_last_log = 0
    optimizer_step_count = 0
    log(f"    모드는 {mode}이고, {gradient_accumulation_steps}묶음마다 한 번씩 가중치를 고쳐요.")

    for batch_index, batch in enumerate(batches):
        log("")
        log(f"[묶음 {batch_index}] 1. 순전파: 원문과 디코더 입력을 모델에 넣어서, 자리마다 다음 단어를 고를 벡터를 만들어요.")
        log("    모델 안쪽 로그는 inference_test.py에서 자세히 봤으니, 여기서는 잠깐 꺼 둬요.")
        with trace.quiet():
            decoder_output = model.forward(
                batch.source_tokens, batch.decoder_input_tokens, batch.source_padding_mask, batch.decoder_mask
            )
        sentence_count, position_count, model_dim = decoder_output.size()
        log(f"    decoder_output 모양 {(sentence_count, position_count, model_dim)}: 문장 {sentence_count}개, 자리 {position_count}개, 자리마다 숫자 {model_dim}칸이에요.")

        log(f"[묶음 {batch_index}] 2. 손실 계산: 모델이 정답을 얼마나 못 맞혔는지 숫자 하나로 만들어요.")
        log("    먼저 이번 묶음에서 패딩을 뺀 정답 토큰 수를 세요. 손실을 평균 낼 때 나누는 값이에요.")
        num_target_tokens = int(batch.num_target_tokens)
        log(f"    정답 토큰 수 = {num_target_tokens}개")
        loss, loss_for_backward = loss_compute(decoder_output, batch.decoder_target_tokens, num_target_tokens)
        log(f"    loss(합계) = {loss:.2f}, loss_for_backward(토큰 하나당 평균) = {float(loss_for_backward):.4f}")
        # loss_for_backward = loss_for_backward / gradient_accumulation_steps  (원본에서도 주석 처리된 줄이에요)

        if is_training:
            log(f"[묶음 {batch_index}] 3. 역전파: 손실을 줄이려면 가중치를 어느 쪽으로 얼마나 바꿔야 하는지(기울기, grad) 구해요.")
            if hasattr(loss_for_backward, "backward"):
                loss_for_backward.backward()
                log("    기울기를 구했어요.")
            else:
                log("    arr_fun Tensor에는 아직 backward가 없어서 이 단계는 건너뛰어요. 그래서 기울기가 없고, 가중치도 안 바뀌어요.")

            log(f"[묶음 {batch_index}] 4. TrainState 기록: 지금까지 학습한 묶음 수, 문장 수, 정답 토큰 수를 세요.")
            train_state.step += 1
            train_state.samples += batch.source_tokens.shape[0]
            train_state.tokens += num_target_tokens
            log(f"    step = {train_state.step}, samples = {train_state.samples}, tokens = {train_state.tokens}")

            log(f"[묶음 {batch_index}] 5. 가중치 고치기: batch_index % {gradient_accumulation_steps} == 0 일 때만 optimizer.step을 불러요.")
            log("    여러 묶음의 기울기를 모았다가 한 번에 반영하는 방식이라 '기울기 누적'이라고 해요. 메모리가 작아도 큰 묶음처럼 학습할 수 있어요.")
            should_update = batch_index % gradient_accumulation_steps == 0
            log(f"    지금은 {batch_index} % {gradient_accumulation_steps} = {batch_index % gradient_accumulation_steps} 이라서 {'가중치를 고쳐요' if should_update else '고치지 않고 기울기를 계속 모아요'}.")
            if should_update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1
                train_state.accum_step += 1
                log(f"    가중치를 고친 횟수 optimizer_step_count = {optimizer_step_count}, train_state.accum_step = {train_state.accum_step}")

            log(f"[묶음 {batch_index}] 6. 학습률 바꾸기: 처음에는 학습률을 조금씩 올리고(warmup), 나중에는 천천히 내려요.")
            scheduler.step()

        log(f"[묶음 {batch_index}] 7. 이번 묶음의 손실과 정답 토큰 수를 epoch 전체 합계에 더해요.")
        total_loss += loss
        total_target_tokens += num_target_tokens
        tokens_since_last_log += num_target_tokens
        log(f"    total_loss = {total_loss:.2f}, total_target_tokens = {total_target_tokens}")

        if batch_index % 40 == 1 and is_training:
            log(f"[묶음 {batch_index}] 8. 진행 상황 출력: 1, 41, 81번째처럼 40묶음마다 손실, 속도, 학습률을 한 줄로 찍어요.")
            learning_rate = optimizer.param_groups[0]["lr"]
            elapsed_seconds = time.time() - start_time
            log(f"    지난 출력 뒤로 {elapsed_seconds:.3f}초 동안 정답 토큰 {tokens_since_last_log}개를 처리했어요.")
            log(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (
                    batch_index,
                    optimizer_step_count,
                    loss / num_target_tokens,
                    tokens_since_last_log / max(elapsed_seconds, 1e-9),
                    learning_rate,
                )
            )
            start_time = time.time()
            tokens_since_last_log = 0
        del loss
        del loss_for_backward

    log("")
    log("[epoch 끝] epoch 전체 손실 합계를 전체 정답 토큰 수로 나눠서, 토큰 하나당 평균 손실을 구해요.")
    average_loss = total_loss / total_target_tokens
    log(f"    {total_loss:.2f} / {total_target_tokens} = {average_loss:.4f}")
    return average_loss, train_state


def rate(step, model_size, factor, warmup):
    """Annotated Transformer의 학습률 공식이에요.
    warmup 단계까지는 학습률이 점점 커지고, 그 뒤로는 step의 제곱근에 반비례해서 천천히 작아져요.
    step이 0이면 0의 음수 제곱이 돼서 계산이 안 되니까 1로 바꿔요.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def example_learning_schedule():
    """원본 예제예요. 모델 크기와 warmup이 다를 때 학습률이 단계마다 어떻게 바뀌는지 봐요.
    원본은 pandas와 altair로 선 그래프 3개를 그리는데, 여기서는 표 로그로 보여줘요.
    원본은 torch.optim.Adam을 쓰는데, 학습률 값은 optimizer 종류와 상관없이 rate 공식으로만 정해져서 arr_fun Optimizer를 써요.
    원본 이름: schedule_options는 opts, example_index는 idx, learning_rates_of_example은 tmp예요.
    """
    import arr_fun as torch
    from arr_fun.optim import LambdaLR, Optimizer

    log("")
    log("[학습률 예제 1] 세 가지 설정을 비교해요. [모델 크기, factor, warmup] 순서예요.")
    schedule_options = [
        [512, 1, 4000],  # 예제 1
        [512, 1, 8000],  # 예제 2
        [256, 1, 4000],  # 예제 3
    ]
    for example_index, example in enumerate(schedule_options):
        log(f"    예제 {example_index + 1}: 모델 크기 {example[0]}, factor {example[1]}, warmup {example[2]}")

    log("[학습률 예제 2] 공식을 한 번 직접 계산해 봐요. 예제 1의 4000단계(warmup이 끝나는 단계)예요.")
    log("    학습률 = factor * 모델크기^(-0.5) * min(단계^(-0.5), 단계 * warmup^(-1.5))")
    sample_rate = rate(4000, 512, 1, 4000)
    log(f"    = 1 * 512^(-0.5) * min(4000^(-0.5), 4000 * 4000^(-1.5)) = 1 * {512 ** -0.5:.5f} * min({4000 ** -0.5:.5f}, {4000 * 4000 ** -1.5:.5f}) = {sample_rate:.6f}")
    log("    warmup 단계에서는 min 안의 두 값이 같아져요. 그 전에는 오른쪽 값이 작아서 학습률이 커지고, 그 뒤에는 왼쪽 값이 작아서 학습률이 줄어요.")

    log("[학습률 예제 3] 예제마다 2만 단계를 돌며 학습률을 기록해요. 6만 번을 다 찍으면 너무 길어서 로그는 꺼 두고 계산만 해요.")
    dummy_model = torch.Linear(1, 1)
    learning_rates = []
    with trace.quiet():
        for example_index, example in enumerate(schedule_options):
            optimizer = Optimizer(dummy_model.parameters(), lr=1)
            lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda step: rate(step, *example))
            learning_rates_of_example = []
            for step in range(20000):
                learning_rates_of_example.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                lr_scheduler.step()
            learning_rates.append(learning_rates_of_example)
    log(f"    3번 결과_학습률: 예제 {len(learning_rates)}개, 예제마다 {len(learning_rates[0])}단계")

    log("[학습률 예제 4] 원본 그래프 대신 표로 봐요. 칸 제목은 '모델 크기:warmup' 이에요.")
    shown_steps = [0, 1, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 15000, 19999]
    log("     step |   512:4000 |   512:8000 |   256:4000")
    for step in shown_steps:
        row_text = " | ".join(f"{learning_rates[example_index][step]:10.7f}" for example_index in range(3))
        log(f"    {step:5d} | {row_text}")
    for example_index, label in enumerate(["512:4000", "512:8000", "256:4000"]):
        peak_step = max(range(20000), key=lambda step: learning_rates[example_index][step])
        log(f"    {label}: 가장 큰 학습률은 {peak_step}단계의 {learning_rates[example_index][peak_step]:.6f}이에요.")
    log("[학습률 예제 5] warmup이 길수록 늦게, 더 낮게 꼭대기에 닿아요. 모델 크기가 작을수록 학습률이 전체적으로 커요.")
    return learning_rates


def data_gen(vocab_size, batch_size, num_batches, sentence_length, pad_id):
    """Annotated Transformer의 복사 문제 데이터예요.
    무작위 단어로 문장을 만들고, 정답도 입력과 똑같은 문장이에요. 모델이 입력을 그대로 따라 쓰도록 배우는 문제예요.
    """
    from train_func.batch import Batch

    for batch_number in range(num_batches):
        log("")
        log(f"[데이터] {batch_number}번 묶음을 만들어요. 1부터 {vocab_size - 1} 사이 무작위 토큰으로 문장 {batch_size}개를 만들고, 첫 토큰은 시작 표시 1로 바꿔요.")
        sentences = [[random.randint(1, vocab_size - 1) for _ in range(sentence_length)] for _ in range(batch_size)]
        for sentence in sentences:
            sentence[0] = 1
        for sentence_index, sentence in enumerate(sentences):
            log(f"    문장 {sentence_index + 1}: {sentence}")

        log("    복사 문제라서 원문과 정답이 같아요. Batch가 디코더 입력(마지막 토큰 뺌)과 정답(첫 토큰 뺌)으로 나눠요.")
        log("    Batch 안쪽 로그는 batch.py에서 자세히 봤으니 꺼 둬요.")
        import arr_fun as torch

        source_tokens = torch.LongTensor(sentences)
        target_tokens = torch.LongTensor(sentences)
        with trace.quiet():
            batch = Batch(source_tokens, target_tokens, pad_id)
        log(f"    디코더 입력: {batch.decoder_input_tokens.numpy().tolist()}")
        log(f"    맞혀야 할 정답: {batch.decoder_target_tokens.numpy().tolist()}")
        yield batch


def main():
    import arr_fun as torch
    from arr_fun.optim import LambdaLR, Optimizer
    from train_func.loss_compute import SimpleLossCompute
    from transformer.model.make_model import make_model

    # 실행할 때마다 같은 숫자가 나오게 고정해요.
    torch.manual_seed(0)
    random.seed(0)

    vocab_size = 11
    pad_id = 0
    model_dim = 4

    log("[준비] 아주 작은 모델을 만들어요. 단어 벡터 4칸, FF 은닉 8칸, head 2개, 레이어 2개, 단어장 11개예요.")
    with trace.quiet():
        model = make_model(vocab_size, vocab_size, num_layers=2, model_dim=model_dim, hidden_dim=8, num_heads=2)
    model.train()
    log(f"    가중치 묶음(Parameter) 수 = {len(list(model.parameters()))}개")

    log("[준비] 손실 계산기를 만들어요. 패딩 토큰은 0번이에요.")
    loss_compute = SimpleLossCompute(model.generator, pad_id)

    log("[준비] optimizer와 scheduler를 만들어요. 처음 학습률 0.5에 rate 공식을 곱해서 실제 학습률을 정해요.")
    optimizer = Optimizer(model.parameters(), lr=0.5)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: rate(step, model_size=model_dim, factor=1.0, warmup=400),
    )
    log(f"    0단계 학습률 = {optimizer.param_groups[0]['lr']:.6f}")

    batches = data_gen(vocab_size, batch_size=2, num_batches=3, sentence_length=5, pad_id=pad_id)
    average_loss, train_state = run_epoch(
        batches, model, loss_compute, optimizer, scheduler, mode="train", gradient_accumulation_steps=2
    )

    log("")
    log("[끝] epoch 하나를 다 돌았어요.")
    log(f"    평균 손실 = {average_loss:.4f}")
    log(f"    TrainState = {train_state}")
    log("    arr_fun에 backward가 아직 없어서 가중치는 그대로예요. 그래서 지금 손실은 학습 전 모델의 손실이에요.")

    # 원본의 example_learning_schedule() 예제예요. 이 파일을 직접 실행할 때만 돌아가요.
    show_example(example_learning_schedule)


if __name__ == "__main__":
    main()
