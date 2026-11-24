import os
import sys

# 이 파일을 실행 버튼으로 바로 돌려도 transformer 폴더를 찾을 수 있게, 프로젝트 폴더를 경로에 넣어요.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from help_func.trace import log
from transformer.attention.subsequent_mask import subsequent_mask


class Batch:

    def __init__(self, source_tokens, target_tokens=None, pad_id=2):
        self.source_tokens = source_tokens

        log("")
        log("[1] source_tokens != pad_id 를 먼저 계산해요. 패딩이 아니면 True예요.")
        source_not_pad = source_tokens != pad_id
        log(source_not_pad)

        log("[1] 그 결과에 unsqueeze(-2)로 괄호를 한 겹 씌운 것이 source_padding_mask예요.")
        self.source_padding_mask = source_not_pad.unsqueeze(-2)
        log(f"    모양이 {source_not_pad.size()} 에서 {self.source_padding_mask.size()} 로 바뀌어요.")
        log(self.source_padding_mask)

        if target_tokens is not None:
            log("")
            log("[2] target_tokens[:, :-1] 은 모든 문장에서 마지막 단어를 뺀 것이에요. 디코더에 넣는 입력이에요.")
            self.decoder_input_tokens = target_tokens[:, :-1]
            log(self.decoder_input_tokens)

            log("[2] target_tokens[:, 1:] 은 모든 문장에서 첫 단어를 뺀 것이에요. 모델이 맞혀야 하는 정답이에요.")
            self.decoder_target_tokens = target_tokens[:, 1:]
            log(self.decoder_target_tokens)
            log("    위아래를 같은 칸끼리 보면, 입력 단어 하나를 보고 바로 다음 정답 단어를 맞히는 짝이 돼요.")

            self.decoder_mask = self.make_decoder_mask(
                self.decoder_input_tokens, pad_id
            )

            log("")
            log("[4] 패딩을 뺀 정답 단어 수 num_target_tokens 는 손실을 평균 낼 때 나누는 값이에요.")
            self.num_target_tokens = (self.decoder_target_tokens != pad_id).sum()
            log(self.num_target_tokens)

    @staticmethod
    def make_decoder_mask(decoder_input_tokens, pad_id):
        log("")
        log(f"[3] 패딩 마스크예요. 단어마다 진짜 단어면 O, 패딩({pad_id})이면 X 표시를 붙여요.")
        log("    X가 붙은 단어는 나중에 attention에서 점수를 아주 작은 값(-1e9)으로 바꿔요.")
        log("    그러면 softmax 결과가 0이 돼서, 다른 단어가 그 단어를 전혀 참고하지 않아요. 이게 '가린다'는 뜻이에요.")
        padding_mask = (decoder_input_tokens != pad_id).unsqueeze(-2)
        for sentence_index in range(decoder_input_tokens.size(0)):
            token_list = [int(token) for token in decoder_input_tokens[sentence_index].numpy().tolist()]
            marks = padding_mask[sentence_index].numpy().reshape(-1).tolist()
            pairs = "  ".join(f"{token}({'O' if mark else 'X'})" for token, mark in zip(token_list, marks))
            log(f"    문장 {sentence_index + 1}:  {pairs}")

        log("[3] 디코더 입력의 단어 개수예요. 미래 마스크의 크기가 돼요.")
        sequence_length = decoder_input_tokens.size(-1)
        log(sequence_length)

        log("[3] 미래 마스크예요. 몇 번째 자리든 자기 자리와 그 앞 자리만 보고, 뒤 자리는 못 보게 가려요.")
        future_mask = subsequent_mask(sequence_length).type_as(padding_mask)
        for position, row in enumerate(future_mask[0].numpy().tolist()):
            visible = [f"{i + 1}" for i, can_see in enumerate(row) if can_see]
            log(f"    {position + 1}번째 자리가 볼 수 있는 자리: {', '.join(visible)}번째")

        log("[3] 최종 decoder_mask예요. 미래 마스크와 패딩 마스크를 둘 다 적용해서, 실제 단어로 보면 이렇게 돼요.")
        decoder_mask = padding_mask & future_mask
        for sentence_index in range(decoder_mask.size(0)):
            Batch._log_mask_by_word(
                decoder_mask[sentence_index],
                decoder_input_tokens[sentence_index],
                f"문장 {sentence_index + 1}",
            )
        return decoder_mask

    @staticmethod
    def _log_mask_by_word(mask, tokens, title):
        """True/False 표를 "몇 번째 단어는 어떤 단어까지 볼 수 있다"는 문장으로 바꿔 찍어요."""
        token_list = [int(token) for token in tokens.numpy().tolist()]
        mask_rows = mask.numpy().tolist()
        log(f"    {title}, 디코더 입력 단어 {token_list}")
        for word_index, row in enumerate(mask_rows):
            visible = [str(token_list[i]) for i, can_see in enumerate(row) if can_see]
            log(f"      {word_index + 1}번째 단어 {token_list[word_index]} 이(가) 볼 수 있는 단어: {', '.join(visible)}")


def main():
    import arr_fun as torch

    # 패딩 번호는 2예요. 문장 길이를 맞추려고 빈 칸에 채워 넣는 숫자예요.
    pad_id = 2

    log("[입력] 원문 문장 묶음 source_tokens 예요. 두 번째 문장은 짧아서 끝에 2(패딩)를 채웠어요.")
    source_tokens = torch.Tensor([[5, 6, 7, 8], [5, 9, 2, 2]])
    log(source_tokens)

    log("[입력] 정답 문장 묶음 target_tokens 예요. 0은 시작, 1은 끝, 2는 패딩이에요.")
    target_tokens = torch.Tensor([[0, 3, 4, 5, 1], [0, 3, 1, 2, 2]])
    log(target_tokens)

    Batch(source_tokens, target_tokens, pad_id)


if __name__ == "__main__":
    main()