import arr_fun as nn
from help_func import trace


class EncoderDecoder(nn.Module):
    """인코더, 디코더, 임베딩, Generator를 하나로 묶은 트랜스포머 전체 모델
    인코더는 입력 문장을 읽어서 결과(encoder_output)를 만들고,
    디코더는 그 결과를 참고해서 출력 문장의 다음 단어를 만들 벡터를 계산함
    """

    def __init__(self, encoder, decoder, source_embedding, target_embedding, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 입력 문장의 단어 번호를 벡터로 바꾸고 위치 정보를 더하는 부분 (원본 코드의 src_embed)
        self.source_embedding = source_embedding
        # 출력 문장의 단어 번호를 벡터로 바꾸고 위치 정보를 더하는 부분 (원본 코드의 tgt_embed)
        self.target_embedding = target_embedding
        # 디코더 결과를 단어 확률로 바꾸는 Generator
        self.generator = generator

    def forward(self, source_tokens, target_tokens, source_mask, target_mask):
        # 입력 문장을 인코딩한 뒤, 그 결과를 참고해서 출력 문장을 디코딩함
        encoder_output = self.encode(source_tokens, source_mask)
        return self.decode(encoder_output, source_mask, target_tokens, target_mask)

    def encode(self, source_tokens, source_mask):
        # 입력 문장 단어 번호를 벡터로 바꾼 뒤 인코더에 통과시킴
        # 로그에서 입력 문장 단어 이름(A, B, C)을 쓰도록 정해요. 계산에는 영향이 없어요.
        trace.Q_NAMES = None
        trace.KV_NAMES = None
        _set_embedding_labels(self.source_embedding, "임베딩", "인코더 레이어 1 입력")
        with trace.place("인코더 입력 (encoder_decoder.py)", trace.INPUT_STEPS):
            embedded = self.source_embedding(source_tokens)
        return self.encoder(embedded, source_mask)

    def decode(self, encoder_output, source_mask, target_tokens, target_mask):
        # 출력 문장 단어 번호를 벡터로 바꾼 뒤, 인코더 결과와 함께 디코더에 통과시킴
        # 로그에서 디코더 쪽 단어 이름을 "시작", "출력1", ... 로 정해요. 계산에는 영향이 없어요.
        trace.Q_NAMES = ["시작"] + [f"출력{i}" for i in range(1, target_tokens.size(1))]
        trace.KV_NAMES = None
        _set_embedding_labels(self.target_embedding, "디코더 임베딩", "디코더 레이어 1 입력")
        with trace.place("디코더 입력 (encoder_decoder.py)", trace.INPUT_STEPS):
            embedded = self.target_embedding(target_tokens)
        return self.decoder(embedded, encoder_output, source_mask, target_mask)


def _set_embedding_labels(embedding_block, embedding_label, output_label):
    """임베딩과 위치 정보 부품에 로그 이름을 붙여요. 계산에는 영향이 없어요."""
    if not hasattr(embedding_block, "__iter__"):
        return
    for part in embedding_block:
        if hasattr(part, "lut"):
            part.output_label = embedding_label
        elif hasattr(part, "pe"):
            part.input_label = embedding_label
            part.output_label = output_label
