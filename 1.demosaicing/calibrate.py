#!/usr/bin/env python3

import json
import timeit

import numpy as np


def create_arrays(size, scale=1.0):
    assert len(size) == 2
    a = np.random.normal(scale=scale, size=size)
    b = np.random.normal(scale=scale, size=size)
    return a, b


def do_calculations(a, b, dtype):
    old_dtype = a.dtype

    a = a.astype(dtype)
    b = b.astype(dtype)

    a.astype(old_dtype)
    b.astype(old_dtype)

    a.clip(-10.0, 10.0)
    np.where(a > 0, a, b)
    np.roll(a, 10)

    a + b
    a * b

    b[b == 0] = 1.0
    a / b

    a.sum()
    a.sum(axis=0)
    a.sum(axis=1)


np.random.seed(702)
size = (512, 512)
dtypes = (np.float32, np.float64, np.uint8, np.int32)
scale = 10.0
n_iter = 3000

print("Measuring local performance. This might take a minute...")

t_local = 0.0
a, b = create_arrays(size, scale)
for dtype in dtypes:
    t_local += timeit.timeit(
        lambda: do_calculations(a, b, dtype),
        number=n_iter,
    )

print(f"Local time: {t_local:.2f} seconds.")

t_system = 63.79
coeff = t_system / t_local

print(f"Writing coefficient {coeff:.2f}.")
with open("calibration.json", "w") as f:
    json.dump({"coefficient": coeff}, f)
