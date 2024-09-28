import numpy as np

import align


def align_image(raw_img, method):
    assert raw_img.ndim == 2
    raw_img.flags.writeable = False

    find_relative_shift_fn = {
        "pyramid": align.find_relative_shift_pyramid,
        "fourier": align.find_relative_shift_fourier,
    }[method]

    # Crop only the inner-most part of each channel
    # and compute their absolute alignment shifts
    crops, crop_coords = align.extract_channel_plates(
        raw_img,
        crop=True,
    )
    r_to_g, b_to_g = align.find_absolute_shifts(
        crops,
        crop_coords,
        find_relative_shift_fn,
    )

    # Create an aligned image from whole channels
    # and previously extracted alignment information
    chans, chan_coords = align.extract_channel_plates(
        raw_img,
        crop=False,
    )
    aligned_image = align.create_aligned_image(
        chans,
        chan_coords,
        r_to_g,
        b_to_g,
    )

    return r_to_g, b_to_g, aligned_image


def visualize_point(raw_img, r_point, g_point, b_point, t=1, l=9):
    # Convert grayscale image to RGB
    assert raw_img.ndim == 2
    vis_img = np.repeat(raw_img[..., None], 3, axis=-1)

    def _draw_rect(y0, x0, y1, x1, color):
        vis_img[
            y0.clip(0, None) : y1.clip(0, None),
            x0.clip(0, None) : x1.clip(0, None),
        ] = color

    hor = np.array([t, l])
    ver = np.array([l, t])

    for point, color in [
        (r_point, [1, 0, 0]),
        (g_point, [0, 1, 0]),
        (b_point, [0, 0, 1]),
    ]:
        # Draw a + sign, centered around the point
        _draw_rect(*(point - hor), *(point + hor + 1), color)
        _draw_rect(*(point - ver), *(point + ver + 1), color)

    return vis_img
