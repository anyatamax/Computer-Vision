import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    h, w = raw_img.shape
    new_h = h // 3
    coords = (np.array([new_h * 2, 0]), np.array([new_h, 0]), np.array([0, 0])) # b,g,r -> r,g,b
    unaligned_rgb = (np.array(raw_img[new_h * 2:new_h * 3, :]), np.array(raw_img[new_h:new_h * 2, :]), np.array(raw_img[:new_h, :]))

    if crop == True:
        h_skip = int(new_h * 0.1)
        w_skip = int(w * 0.1)
        delta = [h_skip, w_skip]
        
        coords = (coords[0] + delta, coords[1] + delta, coords[2] + delta)
        unaligned_rgb = (
            unaligned_rgb[0][h_skip : new_h - h_skip, w_skip : w - w_skip],
            unaligned_rgb[1][h_skip : new_h - h_skip, w_skip : w - w_skip],
            unaligned_rgb[2][h_skip : new_h - h_skip, w_skip : w - w_skip],
        )

    return unaligned_rgb, coords


def mse(I_1, I_2):
    if I_1.shape == I_2.shape:
        return 1. / (I_1.shape[0] * I_1.shape[1]) * np.sum((I_1 - I_2) ** 2)
    return 0.0


def cross_valid(I_1, I_2):
    if I_1.shape == I_2.shape:
        return np.sum(I_1 * I_2, dtype=np.float64) / np.sqrt(np.sum(I_1 ** 2, dtype=np.float64) * np.sum(I_2 ** 2, dtype=np.float64), dtype=np.float64)
    return 0.0


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


def find_best_shift_in_range(img_a, img_b, r_h_min=-15, r_h_max=15, r_w_min=-15, r_w_max=15):
    result_shift = np.array([0, 0])
    if img_a.shape != img_b.shape:
        return result_shift
    
    r_h_max += 1
    r_w_max += 1
    min_mse = 1000000000.0
    # print(min_mse)
    for h_shift in range(r_h_min, r_h_max):
        for w_shift in range(r_w_min, r_w_max):
            cropped_img_a = img_a[max(0, -h_shift) : img_a.shape[0] - max(0, h_shift), max(0, -w_shift) : img_a.shape[1] - max(0, w_shift)]
            cropped_img_b = img_b[max(0, h_shift) : img_b.shape[0] - max(0, -h_shift), max(0, w_shift) : img_b.shape[1] - max(0, -w_shift)]
            # print(cropped_img_a.shape, cropped_img_b.shape)
            cur_mse = mse(cropped_img_a, cropped_img_b)
            if cur_mse < min_mse:
                result_shift = np.array([h_shift, w_shift])
                min_mse = cur_mse

    return result_shift
     

def find_relative_shift_pyramid(img_a, img_b):
    a_to_b = np.array([0, 0])

    if max(img_a.shape) < 500 and max(img_b.shape) < 500:
        result_shift = find_best_shift_in_range(img_a, img_b)
        a_to_b[0] = result_shift[0]
        a_to_b[1] = result_shift[1]
        return a_to_b

    conv_img_a = roll(img_a, np.zeros((2,2)), 2, 2).mean(axis=(2,3))
    conv_img_b = roll(img_b, np.zeros((2,2)), 2, 2).mean(axis=(2,3))
    
    cropped_shift = find_relative_shift_pyramid(conv_img_a, conv_img_b)
    shift_expanded = np.array([cropped_shift[0] * 2, cropped_shift[1] * 2])
    result_shift = find_best_shift_in_range(img_a, img_b, shift_expanded[0] - 2, shift_expanded[0] + 2, shift_expanded[1] - 2, shift_expanded[1] + 2)
    a_to_b[0] = result_shift[0]
    a_to_b[1] = result_shift[1]
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    r_img, g_img, b_img = crops[0], crops[1], crops[2]
    rel_shift_r_g = find_relative_shift_fn(r_img, g_img)
    rel_shift_b_g = find_relative_shift_fn(b_img, g_img)
    
    dist_r_g_h = crop_coords[1][0] - crop_coords[0][0]
    dist_r_g_h += rel_shift_r_g[0]
    dist_r_g_w = crop_coords[1][1] - crop_coords[0][1]
    dist_r_g_w += rel_shift_r_g[1]
    
    dist_b_g_h = crop_coords[1][0] - crop_coords[2][0]
    dist_b_g_h += rel_shift_b_g[0]
    dist_b_g_w = crop_coords[1][1] - crop_coords[2][1]
    dist_b_g_w += rel_shift_b_g[1]
        
    r_to_g = np.array([dist_r_g_h, dist_r_g_w])
    b_to_g = np.array([dist_b_g_h, dist_b_g_w])
    return r_to_g, b_to_g


def get_cut_shifts(img_a_shift, img_b_shift, img_shape):
    return [max(0, -img_a_shift) + max(max(0, img_b_shift) - max(0, img_a_shift), 0), img_shape - max(0, img_a_shift) - max(max(0, -img_b_shift) - max(0, -img_a_shift), 0)]

def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    r_img, g_img, b_img = channels[0], channels[1], channels[2]
    # print(r_img.shape)
    
    r_g_h_shift = r_to_g[0] - (channel_coords[1][0] - channel_coords[0][0])
    r_g_w_shift = r_to_g[1] - (channel_coords[1][1] - channel_coords[0][1])
    
    b_g_h_shift = b_to_g[0] - (channel_coords[1][0] - channel_coords[2][0])
    b_g_w_shift = b_to_g[1] - (channel_coords[1][1] - channel_coords[2][1])
    
    r_cut_h = get_cut_shifts(r_g_h_shift, b_g_h_shift, r_img.shape[0])
    r_cut_w = get_cut_shifts(r_g_w_shift, b_g_w_shift, r_img.shape[1])
    r_img = r_img[r_cut_h[0] : r_cut_h[1], r_cut_w[0] : r_cut_w[1]]
    
    b_cut_h = get_cut_shifts(b_g_h_shift, r_g_h_shift, b_img.shape[0])
    b_cut_w = get_cut_shifts(b_g_w_shift, r_g_w_shift, b_img.shape[1])
    b_img = b_img[b_cut_h[0] : b_cut_h[1], b_cut_w[0] : b_cut_w[1]]
    
    g_cut_h = [max(0, max(r_g_h_shift, b_g_h_shift)), g_img.shape[0] - max(0, max(-r_g_h_shift, -b_g_h_shift))]
    g_cut_w = [max(0, max(r_g_w_shift, b_g_w_shift)), g_img.shape[1] - max(0, max(-r_g_w_shift, -b_g_w_shift))]
    g_img = g_img[g_cut_h[0] : g_cut_h[1], g_cut_w[0] : g_cut_w[1]]
    
    aligned_img = np.array([r_img, g_img, b_img]).transpose(1, 2, 0)
    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    fft_img_a = np.fft.fft2(img_a)
    fft_img_b = np.fft.fft2(img_b)
    conj_fft_img_a = np.conjugate(fft_img_a)

    corss_cor = np.fft.ifft2(conj_fft_img_a * fft_img_b)
    result_coord = np.argmax(corss_cor)
    # print(result_coord)
    result_coord = np.unravel_index(result_coord, corss_cor.shape)

    a_to_b = np.array([result_coord[0], result_coord[1]])
    if result_coord[0] > img_a.shape[0] / 2:
        a_to_b[0] = result_coord[0] - img_a.shape[0]
    if result_coord[1] > img_a.shape[1] / 2:
        a_to_b[1] = result_coord[1] - img_a.shape[1]
    return a_to_b


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
