import json
import os
import pathlib

import numpy as np
import PIL.Image


def get_test_images(test):
    test_root = pathlib.Path(test).resolve().parent
    raw_images = sorted(test_root.glob("[0-9][0-9].png"))

    error = (
        "Couldn't the {!r} image required for tests. "
        "Did you completely extract the test archive?"
    )

    for raw_img_path in raw_images:
        gt_img_path = test_root / f"gt_{raw_img_path.name}"

        try:
            raw_img = PIL.Image.open(raw_img_path)
            assert raw_img.mode == "L"
            raw_img = np.asarray(raw_img)
        except (AssertionError, OSError) as err:
            assert False, error.format(str(raw_img_path))

        try:
            gt_img = PIL.Image.open(gt_img_path)
            assert gt_img.mode == "RGB"
            gt_img = np.asarray(gt_img)
        except (AssertionError, OSError) as err:
            assert False, error.format(str(gt_img_path))

        yield str(raw_img_path), raw_img, gt_img


def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    error = f"Value should be an instance of np.ndarray, but it is {type(value)}."
    assert isinstance(value, (np.ndarray, np.generic)), error


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    error = (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )
    assert np.can_cast(actual_dtype, correct_dtype, casting="same_kind"), error
    assert np.can_cast(correct_dtype, actual_dtype, casting="same_kind"), error


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    error = (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )
    assert len(actual_shape) == len(correct_shape), error
    assert actual_shape == correct_shape, error


def assert_ndarray_equal(*, actual, correct, rtol=0, atol=1e-6, err_msg=""):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(
        actual,
        correct,
        atol=atol,
        rtol=rtol,
        verbose=True,
        err_msg=err_msg,
    )


def assert_time_limit(*, actual, limit):
    __tracebackhide__ = True
    filename = "calibration.json"

    if os.environ.get("CHECKER"):
        extra = ""
    else:
        try:
            with open(filename, "r") as f:
                coefficient = json.load(f)["coefficient"]

            extra = f" (based on server time limit of {limit:.2f}s)"
            limit /= coefficient
        except (OSError, ValueError, ZeroDivisionError):
            assert False, (
                "Couldn't read the local calibration coefficient. "
                "Did you forget to run the calibrate.py script?"
            )

    assert actual <= limit, (
        "This test must complete within the specified time limit.\n"
        f"actual_time={actual:.2f}s, time_limit={limit:.2f}s{extra}"
    )
