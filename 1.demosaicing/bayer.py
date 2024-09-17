import numpy as np
import math

def get_bayer_masks(n_rows, n_cols):
    """
        :param n_rows: `int`, number of rows
        :param n_cols: `int`, number of columns

        :return:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
            containing red, green and blue Bayer masks
    """
    masks = np.zeros((n_rows, n_cols, 3), dtype="bool")

    green_mask = np.tile(np.array([[1, 0], [0, 1]], dtype="bool"), (n_rows // 2 + 1, n_cols // 2 + 1))
    green_mask = green_mask[:n_rows, :n_cols]

    red_mask = np.tile(np.array([[0, 1], [0, 0]], dtype="bool"), (n_rows // 2 + 1, n_cols // 2 + 1))
    red_mask = red_mask[:n_rows, :n_cols]

    blue_mask = np.tile(np.array([[0, 0], [1, 0]], dtype="bool"), (n_rows // 2 + 1, n_cols // 2 + 1))
    blue_mask = blue_mask[:n_rows, :n_cols]

    masks[:, :, 0] = red_mask
    masks[:, :, 1] = green_mask
    masks[:, :, 2] = blue_mask

    return masks


def get_colored_img(raw_img):
    """
        :param raw_img:
            `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
            raw image

        :return:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
            each channel contains known color values or zeros
            depending on Bayer masks
    """
    n_rows, n_cols = raw_img.shape[0], raw_img.shape[1]
    bayer_masks = get_bayer_masks(n_rows, n_cols)

    result_img = np.zeros((n_rows, n_cols, 3), dtype="uint8")
    result_img[:, :, 0] = raw_img * bayer_masks[:, :, 0]
    result_img[:, :, 1] = raw_img * bayer_masks[:, :, 1]
    result_img[:, :, 2] = raw_img * bayer_masks[:, :, 2]

    return result_img


def get_raw_img(colored_img):
    """
        :param colored_img:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
            colored image

        :return:
            `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
            raw image as captured by camera
    """
    n_rows, n_cols = colored_img.shape[0], colored_img.shape[1]
    bayer_masks = get_bayer_masks(n_rows, n_cols)

    result_img = np.zeros((n_rows, n_cols), dtype="uint8")
    result_img += colored_img[:, :, 0] * bayer_masks[:, :, 0]
    result_img += colored_img[:, :, 1] * bayer_masks[:, :, 1]
    result_img += colored_img[:, :, 2] * bayer_masks[:, :, 2]

    return result_img

# from https://habr.com/ru/articles/489734/
def roll(a, b, dx=1, dy=1):
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b.shape[-2]) // dy + 1,) + \
            ((a.shape[-1] - b.shape[-1]) // dx + 1,) + \
            b.shape
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def bilinear_interpolation(raw_img):
    """
        :param raw_img:
            `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
            raw image

        :return:
            `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
            result of bilinear interpolation
    """
    n_rows, n_cols = raw_img.shape[0], raw_img.shape[1]
    bayer_masks = get_bayer_masks(n_rows, n_cols)
    colored_mask = get_colored_img(raw_img)
    keras = np.zeros((3, 3))

    for channel in range(3):
        channel_img_view = roll(colored_mask[:, :, channel], keras).reshape((-1, 3, 3))
        channel_mask_view = roll(bayer_masks[:, :, channel], keras).reshape((-1, 3, 3))

        ids = np.where(channel_mask_view[:, 1, 1] != 1)
        result_values = np.array(channel_img_view[ids].sum(axis=(1,2)) / channel_mask_view[ids].sum(axis=(1,2)), dtype="uint8")
        cropped_mask = bayer_masks[1:-1, 1:-1, channel]
        ids_to_replace = np.where(cropped_mask != 1)
        cropped_colored_mask = colored_mask[1:-1, 1:-1, channel]
        cropped_colored_mask[ids_to_replace] = result_values

    return colored_mask


G_at_B = np.array([[0.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [-1.0, 2.0, 4.0, 2.0, -1.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, 0.0]])

G_at_R = G_at_B

R_at_G_Rrow_Bcol = np.array([[0.0, 0.0, 0.5, 0.0, 0.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [-1.0, 4.0, 5.0, 4.0, -1.0],
                            [0.0, -1.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.5, 0.0, 0.0]])
                            
R_at_G_Brow_Rcol = R_at_G_Rrow_Bcol.T

R_at_B_Brow_Bcol = np.array([[0.0, 0.0, -1.5, 0.0, 0.0],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [-1.5, 0.0, 6.0, 0.0, -1.5],
                            [0.0, 2.0, 0.0, 2.0, 0.0],
                            [0.0, 0.0, -1.5, 0.0, 0.0]])
                            
B_at_G_Brow_Rcol = R_at_G_Rrow_Bcol

B_at_G_Rrow_Bcol = R_at_G_Brow_Rcol

B_at_R_Rrom_Rcol = R_at_B_Brow_Bcol


def improved_interpolation(raw_img):
    """
        :param raw_img:
            `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

        :return:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
            result of improved interpolation
    """
    n_rows, n_cols = raw_img.shape[0], raw_img.shape[1]
    bayer_masks = get_bayer_masks(n_rows, n_cols)
    colored_mask = get_colored_img(raw_img).astype(np.float64)
    result_colored_mask = np.copy(colored_mask).astype(np.float64)
    img = raw_img.astype(np.float64)
    
    keras = np.zeros((5, 5), dtype=np.float64)
    img_view = roll(img, keras).reshape((-1, 5, 5))
    r_mask_view = roll(bayer_masks[:, :, 0], keras).reshape((-1, 5, 5))
    g_mask_view = roll(bayer_masks[:, :, 1], keras).reshape((-1, 5, 5))
    b_mask_view = roll(bayer_masks[:, :, 2], keras).reshape((-1, 5, 5))
    
    cropped_mask = bayer_masks[2:-2, 2:-2, :]
    cropped_colored_mask = colored_mask[2:-2, 2:-2, :]
    
    Rrow = (np.expand_dims(np.any(cropped_mask[:, :, 0] == 1, axis=1), axis=0).T * np.ones(cropped_mask[:, :, 0].shape)).astype(int)
    Rcol = (np.expand_dims(np.any(cropped_mask[:, :, 0] == 1, axis=0), axis=0) * np.ones(cropped_mask[:, :, 0].shape)).astype(int)
    Brow = (np.expand_dims(np.any(cropped_mask[:, :, 2] == 1, axis=1), axis=0).T * np.ones(cropped_mask[:, :, 2].shape)).astype(int)
    Bcol = (np.expand_dims(np.any(cropped_mask[:, :, 2] == 1, axis=0), axis=0) * np.ones(cropped_mask[:, :, 2].shape)).astype(int)

    # G
    # G_at_B
    ids = np.where((g_mask_view[:, 2, 2] != 1) & (b_mask_view[:, 2, 2] == 1))
    result_values = (img_view[ids] * G_at_B).sum(axis=(1,2)) / 8.
    ids_to_replace = np.where((cropped_mask[:, :, 1] != 1) & (cropped_mask[:, :, 2] == 1))
    cropped_colored_mask[:, :, 1][ids_to_replace] = result_values
    # G_at_R
    ids = np.where((g_mask_view[:, 2, 2] != 1) & (r_mask_view[:, 2, 2] == 1))
    result_values = (img_view[ids] * G_at_R).sum(axis=(1,2)) / 8.
    ids_to_replace = np.where((cropped_mask[:, :, 1] != 1) & (cropped_mask[:, :, 0] == 1))
    cropped_colored_mask[:, :, 1][ids_to_replace] = result_values
    
    # R
    # R_at_G_Rrow_Bcol
    if raw_img.shape == keras.shape:
        # only R_at_G_Rrow_Bcol
        ids = np.where((r_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (r_mask_view[:, 2, 1] == 1))
        result_values = (img_view[ids] * R_at_G_Rrow_Bcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 0] != 1) & (cropped_mask[:, :, 1] == 1))
        cropped_colored_mask[:, :, 0][ids_to_replace] = result_values
    else:
        # R_at_G_Rrow_Bcol
        ids = np.where((r_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (r_mask_view[:, 2, 1] == 1))
        result_values = (img_view[ids] * R_at_G_Rrow_Bcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 0] != 1) & (cropped_mask[:, :, 1] == 1) & (Rrow == 1) & (Bcol == 1))
        cropped_colored_mask[:, :, 0][ids_to_replace] = result_values
        # R_at_G_Brow_Rcol
        ids = np.where((r_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (r_mask_view[:, 1, 2] == 1))
        result_values = (img_view[ids] * R_at_G_Brow_Rcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 0] != 1) & (cropped_mask[:, :, 1] == 1) & (Brow == 1) & (Rcol == 1))
        cropped_colored_mask[:, :, 0][ids_to_replace] = result_values
        # R_at_B_Brow_Bcol
        ids = np.where((r_mask_view[:, 2, 2] != 1) & (b_mask_view[:, 2, 2] == 1) & (r_mask_view[:, 1, 1] == 1))
        result_values = (img_view[ids] * R_at_B_Brow_Bcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 0] != 1) & (cropped_mask[:, :, 2] == 1) & (Brow == 1) & (Bcol == 1))
        cropped_colored_mask[:, :, 0][ids_to_replace] = result_values
    
    # B
    if raw_img.shape == keras.shape:
        # only B_at_G_Rrow_Bcol
        ids = np.where((b_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (b_mask_view[:, 1, 2] == 1))
        result_values = (img_view[ids] * B_at_G_Rrow_Bcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 2] != 1) & (cropped_mask[:, :, 1] == 1))
        cropped_colored_mask[:, :, 2][ids_to_replace] = result_values
    else:
        # B_at_G_Brow_Rcol
        ids = np.where((b_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (b_mask_view[:, 2, 1] == 1))
        result_values = (img_view[ids] * B_at_G_Brow_Rcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 2] != 1) & (cropped_mask[:, :, 1] == 1) & (Brow == 1) & (Rcol == 1))
        cropped_colored_mask[:, :, 2][ids_to_replace] = result_values
        # B_at_G_Rrow_Bcol
        ids = np.where((b_mask_view[:, 2, 2] != 1) & (g_mask_view[:, 2, 2] == 1) & (b_mask_view[:, 1, 2] == 1))
        result_values = (img_view[ids] * B_at_G_Rrow_Bcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 2] != 1) & (cropped_mask[:, :, 1] == 1) & (Rrow == 1) & (Bcol == 1))
        cropped_colored_mask[:, :, 2][ids_to_replace] = result_values
        # B_at_R_Rrom_Rcol
        ids = np.where((b_mask_view[:, 2, 2] != 1) & (r_mask_view[:, 2, 2] == 1) & (b_mask_view[:, 1, 1] == 1))
        result_values = (img_view[ids] * B_at_R_Rrom_Rcol).sum(axis=(1,2)) / 8.
        ids_to_replace = np.where((cropped_mask[:, :, 2] != 1) & (cropped_mask[:, :, 0] == 1) & (Rrow == 1) & (Rcol == 1))
        cropped_colored_mask[:, :, 2][ids_to_replace] = result_values
    
    return np.clip(colored_mask, 0, 255).astype("uint8")


def compute_psnr(img_pred, img_gt):
    """
        :param img_pred:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
            predicted image
        :param img_gt:
            `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
            ground truth image

        :return:
            `float`, PSNR metric
    """
    img_p, img_g = img_pred.astype(np.float64), img_gt.astype(np.float64)
    mse = ((img_p - img_g) ** 2).mean()
    if mse == 0:
        raise ValueError
    
    return 10 * math.log10(img_g.max() ** 2 / mse)



if __name__ == "__main__":
    from PIL import Image

    raw_img_path = 'tests/04_unittest_bilinear_img_input/02.png'
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save('bilinear.png')

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save('improved.png')
