from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = ((r_max - r_min) / (q_max - q_min))
    zero_point = (np.round((r_max * q_min - r_min * q_max) / (r_max - r_min))).astype(np.int32)
    return QuantizationParameters(scale, zero_point, q_min, q_max)
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.clip(np.round(r / qp.scale + qp.zero_point), qp.q_min, qp.q_max).astype(np.int8)
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return (qp.scale * (q.astype(np.int32) - qp.zero_point))
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        self.min = min(self.min, torch.min(x).detach().numpy().astype(np.float64))
        self.max = max(self.max, torch.max(x).detach().numpy().astype(np.float64))
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    min_max = MinMaxObserver()
    min_max(torch.tensor(weights))
    min_w, max_w = (-np.max(np.abs([min_max.min, min_max.max])), np.max(np.abs([min_max.min, min_max.max])))
    q_parameters = compute_quantization_params(min_w, max_w, np.int32(-127), np.int32(127))
    return quantize(weights, q_parameters), q_parameters
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    q_weights_ch = []
    q_parameters = []
    for i, ch in enumerate(weights):
        cur_channel_q = quantize_weights_per_tensor(ch)
        q_parameters.append(cur_channel_q[1])
        q_weights_ch.append(cur_channel_q[0])
    return np.stack(q_weights_ch, axis=0), q_parameters
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    scale_b = scale_w * scale_x
    return np.round(bias / scale_b + 0).astype(np.int32)
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    min_m0 = 0.5
    pow_approx = m / min_m0
    n = -np.floor(np.log2(pow_approx)).astype(np.int32)
    if n >= 0:   
        m_0 = m * (2 ** (n))
    else:
        m_0 = m / (2 ** (-n))
    m_0 = np.round(m_0 * (2 ** 31)).astype(np.int32)
    return np.int32(n), m_0
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    mult = np.multiply(m0, accum, dtype=np.int64)
    mult_bin = np.binary_repr(mult, 64)
    # print(mult_bin)
    # print(n)

    pos_number = 33 - n
    first_bit_after_point = np.int32(mult_bin[pos_number])
    # print(mult_bin[:pos_number])
    mult_shifted = np.int32(mult >> (64 - pos_number)) + first_bit_after_point
    return mult_shifted
    # your code goes here /\
