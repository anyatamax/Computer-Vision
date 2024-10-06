import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Отцентруем каждую строчку матрицы
    mean_matrix = matrix.mean(axis=1)
    centr_matrix = matrix - mean_matrix[:, None]
    # Найдем матрицу ковариации
    cov = np.cov(centr_matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Посчитаем количество найденных собственных векторов
    cnt = eigenvectors.shape[1]
    # Сортируем собственные значения в порядке убывания
    values_idx_sort = np.argsort(eigenvalues)[::-1]
    values_sort = eigenvalues[values_idx_sort]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    vectors_sort = eigenvectors[:, values_idx_sort]
    # Оставляем только p собственных векторов
    vectors_sort = vectors_sort[:, :p]
    # Проекция данных на новое пространство
    result_matrix = np.dot(vectors_sort.T, centr_matrix)
    return vectors_sort, result_matrix, mean_matrix


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        vectors_sort, matrix, mean_matrix = comp
        result_img.append(np.clip(mean_matrix[:, None] + np.dot(vectors_sort, matrix), 0, 255))
        
    return np.dstack([img for img in result_img])


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[..., j], p))
        decompressed = pca_decompression(compressed)
        # print(decompressed.shape)

        axes[i // 3, i % 3].imshow(decompressed.astype(int))
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")
    

bias = np.array([
    [0],
    [128],
    [128],
], dtype=np.float64)

rgb2ycbcr_m = np.array([
    [0.299, 0.587, 0.114],
    [-0.1687, -0.3313, 0.5],
    [0.5, -0.4187, -0.0813],
], dtype=np.float64)

ycbcr2rgb_m = np.array([
    [1, 0, 1.402],
    [1, -0.34414, -0.71414],
    [1, 1.77, 0]
], dtype=np.float64)


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    return np.clip(np.dot(rgb2ycbcr_m, np.expand_dims(img, axis=3)).transpose((1, 2, 0, 3)) + bias, 0, 255)[:, :, :, 0]


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    return np.clip(np.dot((np.expand_dims(img, axis=3) - bias)[:, :, :, 0], ycbcr2rgb_m.T), 0, 255)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 1 : 3] = gaussian_filter(ycbcr_img[:, :, 1 : 3], sigma=10)
    new_img = ycbcr2rgb(ycbcr_img)
    
    plt.imshow(new_img.astype(int))
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], sigma=10)
    new_img = ycbcr2rgb(ycbcr_img)
    
    plt.imshow(new_img.astype(int))

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    return gaussian_filter(component, sigma=10)[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    dct_result = np.copy(block).astype(np.float64)
    for u in range(0, 8):
        for v in range(0, 8):
            dct_result[u][v] = 0.
            for x in range(0, 8):
                for y in range(0, 8):
                    dct_result[u][v] += block[x][y] * np.cos(((2 * x + 1) * u * np.pi) / 16) * np.cos(((2 * y + 1) * v * np.pi) / 16)
            
            if u == 0:
                dct_result[u][v] /= np.sqrt(2)
            if v == 0:
                dct_result[u][v] /= np.sqrt(2)
            
            dct_result[u][v] /= 4.
     
    return dct_result


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    S = 1 if q == 100 else 5000. / q if 1 <= q and q < 50 else 200 - 2 * q
    new_q_matrix = (50 + S * default_quantization_matrix) // 100
    new_q_matrix[new_q_matrix == 0] = 1

    return new_q_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    zigzag_view = []
    i, j = 0, 0
    while i <= block.shape[0] - 1 and j <= block.shape[0] - 1:
        if i == 0 and j == 0:
            zigzag_view.append(block[0][0])
            j += 1
            continue
        if i == 0:
            while j != 0:
                zigzag_view.append(block[i][j])
                i += 1
                j -= 1
            zigzag_view.append(block[i][j])
            i += 1
            continue
        if j == 0:
            while i != 0:
                zigzag_view.append(block[i][j])
                i -= 1
                j += 1
            zigzag_view.append(block[i][j])
            j += 1
            continue
    i = block.shape[0] - 1
    j = 0
    while i != block.shape[0] - 1 or j != block.shape[0] - 1:
        if i == block.shape[0] - 1:
            j += 1
            while j != block.shape[0] - 1:
                zigzag_view.append(block[i][j])
                i -= 1
                j += 1
            zigzag_view.append(block[i][j])
            continue
        if j == block.shape[0] - 1:
            i += 1
            while i != block.shape[0] - 1:
                zigzag_view.append(block[i][j])
                i += 1
                j -= 1
            zigzag_view.append(block[i][j])
            continue

    return zigzag_view


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    cnt_zeros = 0
    new_list = []
    for v in zigzag_list:
        if v == 0:
            cnt_zeros += 1
            continue
        if cnt_zeros != 0:
            new_list += [0, cnt_zeros]
            cnt_zeros = 0
        new_list.append(v)
    
    if cnt_zeros != 0:
        new_list += [0, cnt_zeros]
    
    return new_list


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


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Переходим из RGB в YCbCr
    ycbcr_img = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    Y = ycbcr_img[:, :, 0]
    Cr = ycbcr_img[:, :, 1]
    Cb = ycbcr_img[:, :, 2]
    Cr = downsampling(Cr)
    Cb = downsampling(Cb)
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    Y, Cb, Cr = Y - 128, Cr - 128, Cb - 128
    Y = np.copy(roll(Y, np.zeros((8,8)), 8, 8)).reshape(-1, 8, 8)
    Cb = np.copy(roll(Cb, np.zeros((8,8)), 8, 8)).reshape(-1, 8, 8)
    Cr = np.copy(roll(Cr, np.zeros((8,8)), 8, 8)).reshape(-1, 8, 8)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    result_Y = [compression(zigzag(quantization(dct(block), quantization_matrixes[0]))) for block in Y]
    result_Cb = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in Cb]
    result_Cr = [compression(zigzag(quantization(dct(block), quantization_matrixes[1]))) for block in Cr]
    return [result_Y, result_Cb, result_Cr]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    full_list = []
    idx = 0
    while idx < len(compressed_list):
        if compressed_list[idx] == 0:
            cnt_zeros = compressed_list[idx + 1]
            for i in range(cnt_zeros):
                full_list.append(0)
            idx += 2
            continue
        full_list.append(compressed_list[idx])
        idx += 1

    return full_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    full_matrix = np.zeros((8, 8))
    i, j = 0, 0
    idx = 0
    while i <= full_matrix.shape[0] - 1 and j <= full_matrix.shape[0] - 1:
        if i == 0 and j == 0:
            full_matrix[0][0] = input[idx]
            idx += 1
            j += 1
            continue
        if i == 0:
            while j != 0:
                full_matrix[i][j] = input[idx]
                idx += 1
                i += 1
                j -= 1
            full_matrix[i][j] = input[idx]
            idx += 1
            i += 1
            continue
        if j == 0:
            while i != 0:
                full_matrix[i][j] = input[idx]
                idx += 1
                i -= 1
                j += 1
            full_matrix[i][j] = input[idx]
            idx += 1
            j += 1
            continue
    i = full_matrix.shape[0] - 1
    j = 0
    while i != full_matrix.shape[0] - 1 or j != full_matrix.shape[0] - 1:
        if i == full_matrix.shape[0] - 1:
            j += 1
            while j != full_matrix.shape[0] - 1:
                full_matrix[i][j] = input[idx]
                idx += 1
                i -= 1
                j += 1
            full_matrix[i][j] = input[idx]
            idx += 1
            continue
        if j == full_matrix.shape[0] - 1:
            i += 1
            while i != full_matrix.shape[0] - 1:
                full_matrix[i][j] = input[idx]
                idx += 1
                i += 1
                j -= 1
            full_matrix[i][j] = input[idx]
            idx += 1
            continue
    return full_matrix


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    dct_result = np.copy(block).astype(np.float64)
    for x in range(0, 8):
        for y in range(0, 8):
            dct_result[x][y] = 0.
            for u in range(0, 8):
                for v in range(0, 8):
                    alpha_1 = alpha_2 = 1
                    if u == 0:
                        alpha_1 = 1. / np.sqrt(2)
                    if v == 0:
                        alpha_2 = 1. / np.sqrt(2)
                    dct_result[x][y] += alpha_1 * alpha_2 * block[u][v] * np.cos(((2 * x + 1) * u * np.pi) / 16) * np.cos(((2 * y + 1) * v * np.pi) / 16)
            
            
            dct_result[x][y] /= 4.
     
    return np.round(dct_result)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    return np.repeat(np.repeat(component, 2, axis=1), 2, axis=0)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    Y = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[0])) for block in result[0]]
    Cb = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[1])) for block in result[1]]
    Cr = [inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(block)), quantization_matrixes[1])) for block in result[2]]
    
    cnt_blocks_row = result_shape[0] // 8
    cnt_blocks_col = result_shape[1] // 8

    Y_result = np.zeros((result_shape[0], result_shape[1]))
    for i in range(0, cnt_blocks_row):
        Y_rows = Y[i * cnt_blocks_row]
        for j in range(1, cnt_blocks_col):
            Y_rows = np.concatenate((Y_rows, Y[i * cnt_blocks_row + j]), axis=1)
        Y_result[i * 8 : i * 8 + 8, :] += Y_rows
        
    cnt_blocks_row = result_shape[0] // 16
    cnt_blocks_col = result_shape[1] // 16
    Cb_result = np.zeros((result_shape[0] // 2, result_shape[1] // 2))
    Cr_result = np.zeros((result_shape[0] // 2, result_shape[1] // 2))
    for i in range(0, cnt_blocks_row):
        Cb_rows = Cb[i * cnt_blocks_row]
        Cr_rows = Cr[i * cnt_blocks_row]
        for j in range(1, cnt_blocks_col):
            Cb_rows = np.concatenate((Cb_rows, Cb[i * cnt_blocks_row + j]), axis=1)
            Cr_rows = np.concatenate((Cr_rows, Cr[i * cnt_blocks_row + j]), axis=1)
        Cb_result[i * 8 : i * 8 + 8, :] += Cb_rows
        Cr_result[i * 8 : i * 8 + 8, :] += Cr_rows
        
    Cb_result, Cr_result = upsampling(Cb_result), upsampling(Cr_result)
    result_img = ycbcr2rgb(np.dstack([Y_result + 128, Cb_result + 128, Cr_result + 128]))
    return np.clip(result_img, 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        own_y_matrix = own_quantization_matrix(y_quantization_matrix, p)
        own_color_matrix = own_quantization_matrix(color_quantization_matrix, p)
        img_compress = jpeg_compression(img, [own_y_matrix, own_color_matrix])
        img_full = jpeg_decompression(img_compress, img.shape, [own_y_matrix, own_color_matrix])

        axes[i // 3, i % 3].imshow(img_full)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
