import os

import numpy as np

import image_compression as ic


def test_pca_decompression_1():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    block_1 = np.array(
        [
            # fmt: off
            [216, 202, 198, 183, 188, 205, 201, 188, 201, 229],
            [225, 210, 204, 199, 191, 193, 200, 204, 217, 239],
            [229, 203, 194, 202, 218, 233, 229, 215, 205, 204],
            [231, 221, 187, 213, 227, 218, 213, 225, 230, 222],
            [238, 237, 206, 192, 197, 223, 237, 233, 231, 239],
            [231, 230, 214, 210, 204, 208, 224, 240, 233, 218],
            [234, 235, 241, 245, 240, 228, 223, 227, 232, 230],
            [243, 239, 221, 212, 216, 230, 225, 204, 201, 213],
            [232, 219, 210, 226, 230, 225, 231, 241, 229, 206],
            [228, 212, 226, 223, 221, 220, 213, 207, 216, 229]
            # fmt: on
        ]
    ).astype(np.float64)
    answer = ic.pca_decompression(
        [
            ic.pca_compression(block_1, 3),
            ic.pca_compression(block_1, 4),
            ic.pca_compression(block_1, 5),
        ]
    )
    true_answer = np.array(
        [
            # fmt: off
            [[217, 219, 215], [208, 205, 202], [200, 196, 197], [184, 185, 184], [184, 188, 187], [201, 201, 204], [201, 199, 201], [188, 187, 188], [198, 199, 200], [224, 227, 229]],
            [[217, 219, 222], [215, 210, 213], [208, 203, 202], [193, 195, 196], [187, 192, 192], [196, 197, 195], [203, 200, 198], [204, 203, 202], [217, 218, 218], [237, 240, 239]],
            [[228, 230, 225], [215, 211, 206], [195, 191, 192], [201, 203, 200], [216, 220, 219], [229, 230, 234], [227, 224, 228], [214, 213, 213], [203, 205, 205], [198, 201, 204]],
            [[226, 230, 235], [220, 210, 215], [201, 190, 189], [207, 211, 214], [213, 223, 224], [218, 220, 216], [225, 218, 214], [231, 227, 226], [225, 229, 229], [215, 224, 220]],
            [[245, 243, 240], [232, 237, 234], [201, 206, 207], [195, 193, 192], [201, 196, 196], [220, 219, 222], [231, 234, 237], [231, 233, 233], [232, 230, 230], [240, 236, 238]],
            [[225, 223, 227], [223, 229, 233], [206, 212, 211], [209, 207, 209], [210, 204, 205], [212, 211, 208], [223, 227, 224], [237, 239, 238], [235, 232, 232], [226, 222, 219]],
            [[226, 227, 233], [231, 229, 235], [243, 241, 240], [241, 242, 245], [237, 239, 240], [232, 232, 227], [229, 228, 223], [228, 228, 227], [231, 231, 231], [233, 234, 230]],
            [[237, 235, 243], [224, 229, 237], [217, 223, 221], [209, 207, 212], [219, 214, 215], [237, 236, 229], [228, 231, 224], [203, 205, 204], [203, 201, 201], [221, 217, 212]],
            [[224, 225, 227], [222, 221, 223], [209, 208, 207], [223, 223, 224], [230, 231, 231], [228, 228, 226], [232, 231, 229], [240, 240, 239], [229, 229, 229], [207, 208, 207]],
            [[221, 223, 223], [220, 216, 216], [228, 223, 223], [220, 222, 222], [218, 222, 222], [220, 221, 220], [216, 213, 212], [207, 206, 205], [213, 215, 215], [227, 231, 230]]
            # fmt: on
        ]
    )
    assert np.max(np.abs(answer - true_answer)) <= 2


def test_pca_decompression_2():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    block_2 = np.array(
        [
            # fmt: off
            [101, 110, 132, 185, 181, 119, 228, 205, 192, 180, 186, 180, 160, 126, 109, 132, 171, 191, 153, 207],
            [107, 142, 109, 136, 136, 149, 169, 123, 150, 108, 164, 129, 128, 60, 123, 127,  99, 118,  77, 166],
            [127, 117, 140, 112, 149, 155, 122, 127, 127, 132, 138, 111, 115, 116, 111, 116, 116,  91, 104,  84],
            [114, 119, 149, 136,  92, 102, 120, 137, 138, 119, 113, 155, 138, 156, 139, 122, 122, 139, 106, 118],
            [125, 134, 124, 160, 148, 123, 125, 119, 130, 144, 133, 133, 153, 125, 141, 126, 112, 135, 138, 128],
            [149, 127, 139, 115, 152, 122, 126, 133, 111, 121, 140, 120, 117, 123, 111, 119, 126, 107, 132, 118],
            [135, 139, 133, 135, 138, 141, 126, 111, 140, 131, 117, 168,  97, 126, 123, 124, 118, 122, 118, 110],
            [142, 127, 137, 137, 150, 125, 122, 142, 121, 138, 139, 141, 140, 130, 124, 131, 128, 116, 110, 105],
            [114, 127, 133, 145, 135, 110, 139, 139, 132, 123, 142, 147, 131, 139, 103, 102, 114, 108,  94, 104],
            [118, 145, 123, 133, 135, 120, 143, 140, 132, 122, 140, 137, 123, 135, 104,  92,  99,  95,  98,  99],
            [106, 134, 143, 141, 140, 132, 139, 135, 137, 130, 123, 118, 105, 102,  64,  39,  39,  41,  54,  68],
            [ 96, 110, 136, 127, 130, 131, 127, 121, 123, 120, 107, 105,  86, 57,  13,   0,   0,   0,   0,   8],
            [101, 108, 110, 108, 122, 140, 135, 124, 115, 112, 105, 101,  72, 21,   0,   0,   0,   0,   8,   6],
            [103, 111, 108, 111, 114, 132, 128, 124, 113, 118, 111,  96,  60, 0,   0,   8,   8,   0,  11,   0],
            [109, 106, 100, 103,  87,  95,  98, 105,  98, 115, 114, 101,  71, 2,   0,   2,   0,   0,  12,   0],
            [127, 105,  82,  94,  75,  83,  95, 107,  96, 118, 118, 116, 102, 30,  12,   0,   0,  17,   6,  21],
            [134, 123, 105,  69,  41,  80,  95, 127, 117, 137, 145, 135, 113, 63,  43,  14,  10,   9,  27,  34],
            [120, 123,  92,  58,  25,  51,  77, 120, 118, 124, 127, 145, 141, 76,  53,  33,  53,  54,  34,  71],
            [115, 114,  68,  47,  20,  30,  60, 102, 114, 111, 143, 147, 143, 93,  79,  52,  45,  22, 110,  82],
            [111,  94,  48,  40,  26,  31,  56,  80, 106, 108, 123, 130, 130, 105,  60,  49,  81, 120, 202, 166]
            # fmt: on
        ]
    ).astype(np.float64)
    answer = ic.pca_decompression(
        [
            ic.pca_compression(block_2, 3),
            ic.pca_compression(block_2, 4),
            ic.pca_compression(block_2, 5),
        ]
    )
    true_answer = np.load("1.npy")
    assert np.max(np.abs(answer - true_answer)) <= 2
