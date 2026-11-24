"""가중치 초기화입니다."""

import math

import numpy as np

from arr_fun.tensor import _tensor_to_array


def xavier_uniform_(tensor, gain=1.0):
    """Glorot 초기화입니다. 입력과 출력 개수의 평균을 써서 균등분포의 폭을 정합니다."""
    # fan_in은 입력 개수, fan_out은 출력 개수를 뜻하는 관례적인 이름입니다.
    weight_array = _tensor_to_array(tensor)
    
    if weight_array.ndim < 2:
        return tensor
    
    fan_out = weight_array.shape[0] # 표의 행 수 
    fan_in = weight_array.shape[1] # 표의 열 수 
    
    receptive_field_size = 1 # 한번에 보는 창 개수, 9면 3x3만큼 돌아가면서 읽음 
    
    if weight_array.ndim > 2:
        receptive_field_size = int(np.prod(weight_array.shape[2:]))

    """ 
    합성곱 가중치 shape (16, 3, 3, 3)의 자리는 이런 뜻으로 정해져 있어요.
    -3×3 도장이 색깔 판 3장마다 하나씩 있는 세트가 16개

    1. 첫째 자리 16은 출력 채널 수예요.
    2. 둘째 자리 3은 입력 채널 수예요.
    3. 셋째 자리 3은 창의 높이예요.
    4. 넷째 자리 3은 창의 너비예요.
    """

    fan_in *= receptive_field_size
    fan_out *= receptive_field_size

    # 2. 값이 퍼지는 정도(std)를 계산해요.
    # gain은 기본값 1.0이라서 곱해도 그대로예요.
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    
    # 3. 무작위 값을 고를 범위의 끝을 계산해요.
    uniform_bound = math.sqrt(3.0) * std

    # 4. 표의 모든 칸을 -1.1과 1.1 사이의 무작위 값으로 채워요.
    weight_array[:] = np.random.uniform(-uniform_bound, uniform_bound, size=weight_array.shape).astype(weight_array.dtype, copy=False)
    
    return tensor
