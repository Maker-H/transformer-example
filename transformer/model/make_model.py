import copy

import arr_fun as nn

from transformer.decoder.decoder import Decoder
from transformer.decoder.decoder_layer import DecoderLayer
from transformer.embedding.embeddings import Embeddings
from transformer.encoder.encoder import Encoder
from transformer.model.encoder_decoder import EncoderDecoder
from transformer.encoder.encoder_layer import EncoderLayer
from transformer.model.generator import Generator
from transformer.attention.multi_headed_attention import MultiHeadedAttention
from transformer.embedding.positional_encoding import PositionalEncoding
from transformer.blocks.positionwise_feed_forward import PositionwiseFeedForward


def make_model(
    source_vocab_size, target_vocab_size, num_layers=6, model_dim=512, hidden_dim=2048, num_heads=8, dropout_rate=0.1
):
    """설정값을 받아서 트랜스포머 전체 모델(EncoderDecoder)을 조립해 리턴함
    원본 코드 이름: num_layers는 N, model_dim은 d_model, hidden_dim은 d_ff, num_heads는 h
    """
    deep_copy = copy.deepcopy
    # 부품 원본을 하나씩 만들어 두고, 쓰는 곳마다 deep_copy로 복사해서 서로 다른 weight를 갖게 함
    attention_layer = MultiHeadedAttention(num_heads, model_dim)
    feed_forward = PositionwiseFeedForward(model_dim, hidden_dim, dropout_rate)
    positional_encoding = PositionalEncoding(model_dim, dropout_rate)
    model = EncoderDecoder(
        # 인코더: self attention과 feed forward를 가진 층을 num_layers개 쌓음
        Encoder(EncoderLayer(model_dim, deep_copy(attention_layer), deep_copy(feed_forward), dropout_rate), num_layers),
        # 디코더: self attention, source attention, feed forward를 가진 층을 num_layers개 쌓음
        Decoder(DecoderLayer(model_dim, deep_copy(attention_layer), deep_copy(attention_layer), deep_copy(feed_forward), dropout_rate), num_layers),
        # 입력 문장용 임베딩과 위치 정보
        nn.Sequential(Embeddings(model_dim, source_vocab_size), deep_copy(positional_encoding)),
        # 출력 문장용 임베딩과 위치 정보
        nn.Sequential(Embeddings(model_dim, target_vocab_size), deep_copy(positional_encoding)),
        # 디코더 결과를 출력 단어장 확률로 바꾸는 Generator
        Generator(model_dim, target_vocab_size),
    )

    # 축이 2개 이상인 parameter(weight 행렬)를 xavier 방식으로 다시 초기화함, 원본 코드에서 중요하다고 적어 둔 부분
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model
