"""정수 인덱스를 벡터로 바꿉니다. 단어 번호가 들어오면 그 행의 벡터를 꺼냅니다."""

import numpy as np

from arr_fun.modules import Module, Parameter
from arr_fun.tensor import Tensor, _tensor_to_array


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 주어진 모양의 배열을 만들고 
        # 모든 칸을 평균 0, 표준편차 1인 정규분포에서 뽑은 무작위 값으로 채움 
        self.weight = Parameter(
            _array=np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        )

    def forward(self, token_ids):
        token_id_array = _tensor_to_array(token_ids).astype(np.int64, copy=False)
        return Tensor(_array=self.weight._array[token_id_array])
