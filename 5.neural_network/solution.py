from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.maximum(inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.grad_inputs = np.multiply(grad_outputs, self.forward_inputs >= 0)
        return self.grad_inputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        normalized_input = np.subtract(inputs, inputs.max(axis=1, keepdims=True))
        return np.exp(normalized_input) / np.expand_dims(np.sum(np.exp(normalized_input), axis=1), axis=1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        sofmax = self.forward_outputs
        self.grad_inputs = (grad_outputs - np.expand_dims(np.sum(grad_outputs * sofmax, axis=1), axis=1)) * sofmax
        return self.grad_inputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.weights_grad = grad_outputs.T @ self.forward_inputs
        self.biases_grad = np.sum(grad_outputs, axis=0)
        self.grad_inputs = grad_outputs @ self.weights
        return self.grad_inputs
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        input_clamp = np.clip(y_pred, eps, 1 - eps)
        
        output = np.sum(np.multiply(y_gt, -np.log(input_clamp))) / y_pred.shape[0]
        return output.reshape((1,))
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        input_clamp = np.clip(y_pred, eps, 1 - eps)
        
        self.grad_inputs = np.multiply(y_gt, -1 / input_clamp) / y_pred.shape[0]
        return self.grad_inputs
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss_function = CategoricalCrossentropy()
    optimizer = SGD(lr=1e-3)
    model = Model(loss_function, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784,), units=1024))
    model.add(ReLU())
    model.add(Dense(units=512))
    model.add(ReLU())
    model.add(Dense(units=64))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        batch_size=64,
        epochs=5,
        shuffle=True,
        verbose=True,
        x_valid=x_valid,
        y_valid=y_valid,
    )

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    batch_size, channel_input, h, w = inputs.shape
    channel_output, channel_input, h_kernel, w_kernel = kernels.shape
    output_h = h - h_kernel + 1 + 2 * padding
    output_w = w - w_kernel + 1 + 2 * padding
    
    inputs_pad = np.zeros([batch_size, channel_input, h + 2 * padding, w + 2 * padding])
    inputs_pad[:, :, padding : h + padding, padding : w + padding] = inputs
    
    output = np.zeros([batch_size, channel_output, output_h, output_w])

    input_view = np.lib.stride_tricks.sliding_window_view(inputs_pad, window_shape=(h_kernel, w_kernel), axis=(-2,-1))
    
    return np.einsum('abcdef,gbef->agcd', input_view, np.flip(kernels, axis = (-2, -1)))
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        same_padding = (self.kernel_size - 1) // 2
        conv = convolve(inputs, self.kernels, same_padding)
        return conv + np.expand_dims(self.biases, axis=(0, 2, 3))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        same_padding = (self.kernel_size - 1) // 2
        transpose_grad_outputs = np.transpose(grad_outputs, (1, 0, 2, 3))
        transpose_forward_inputs = np.transpose(np.flip(self.forward_inputs, axis=(-2, -1)), (1, 0, 2, 3))
        transpose_kernels = np.transpose(np.flip(self.kernels, axis=(-2, -1)), (1, 0, 2, 3))
        
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        
        self.kernels_grad = np.transpose(convolve(transpose_forward_inputs, transpose_grad_outputs, same_padding), (1, 0, 2, 3))
        
        self.grad_inputs = convolve(grad_outputs, transpose_kernels, same_padding)
        return self.grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        inputs_view = np.copy(np.lib.stride_tricks.as_strided(
            inputs,
            shape=(inputs.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2], self.pool_size, self.pool_size),
            strides=(inputs.strides[0], inputs.strides[1], inputs.strides[2] * self.pool_size, inputs.strides[3] * self.pool_size, inputs.strides[2], inputs.strides[3])))
        # print(inputs_view)
        # print(inputs)
        self.saved_ids = inputs_view.reshape(-1, self.pool_size * self.pool_size)
        self.saved_ids = np.argmax(self.saved_ids, axis=-1, keepdims=True)
        
        return inputs_view.max(axis=(-2, -1)) if self.pool_mode == "max" else inputs_view.mean(axis=(-2, -1))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        output_view_shape = (grad_outputs.shape[0], grad_outputs.shape[1], grad_outputs.shape[2], grad_outputs.shape[3], self.pool_size, self.pool_size)
        if self.pool_mode == "max":
            self.grad_inputs = np.zeros(output_view_shape).reshape((-1, self.pool_size * self.pool_size))
            self.grad_inputs[np.arange(self.saved_ids.shape[0]), self.saved_ids[:, 0]] = grad_outputs.reshape(-1)
            self.grad_inputs = self.grad_inputs.reshape(output_view_shape)
        else:
            self.grad_inputs = grad_outputs[:, :, :, :, None, None] * np.ones(output_view_shape) / (self.pool_size * self.pool_size)
        return self.grad_inputs.swapaxes(3, 4).reshape((self.forward_inputs.shape[0], self.forward_inputs.shape[1], self.forward_inputs.shape[2], self.forward_inputs.shape[3]))
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            input_mean = np.mean(inputs, axis=(0, 2, 3))
            input_var = np.var(inputs, axis=(0, 2, 3))
            self.input_mean_diff = (inputs - input_mean[:, None, None])
            self.input_var_diff = np.sqrt(input_var + eps)[:, None, None]
            self.output = self.input_mean_diff / self.input_var_diff
            self.running_mean = self.running_mean * self.momentum + input_mean * (1 - self.momentum)
            self.running_var = self.running_var * self.momentum + input_var * (1 - self.momentum)
        else:
            self.output = (inputs - self.running_mean[:, None, None]) / np.sqrt(self.running_var + eps)[:, None, None]

        return self.gamma[:, None, None] * self.output + self.beta[:, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        x_hat = grad_outputs * self.gamma[:, None, None]
        var_grad = -np.sum(x_hat * self.input_mean_diff, axis=(0, 2, 3))[:, None, None] / (2 * (self.input_var_diff ** 3))
        mean_grad = -np.sum(x_hat, axis=(0, 2, 3))[:, None, None] / self.input_var_diff - var_grad * 2 * np.sum(self.input_mean_diff, axis=(0, 2, 3))[:, None, None] / (self.forward_inputs.shape[0] * self.forward_inputs.shape[2] * self.forward_inputs.shape[3])
        self.grad_inputs = x_hat * 1 / self.input_var_diff + var_grad * 2 * self.input_mean_diff / (self.forward_inputs.shape[0] * self.forward_inputs.shape[2] * self.forward_inputs.shape[3]) + mean_grad / (self.forward_inputs.shape[0] * self.forward_inputs.shape[2] * self.forward_inputs.shape[3])
        
        self.gamma_grad = np.sum(grad_outputs * self.output, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        
        return self.grad_inputs
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        batch_size = inputs.shape[0]
        return inputs.reshape((batch_size, -1))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        self.grad_inputs = grad_outputs.reshape(self.forward_inputs.shape)
        return self.grad_inputs
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.mask = np.random.uniform(0, 1, size=inputs.shape) >= self.p
            dropout_input = self.mask * inputs
            self.output = dropout_input
        else:
            self.output = (1 - self.p) * inputs

        return self.output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        self.grad_inputs = self.mask * grad_outputs
        return self.grad_inputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss_function = CategoricalCrossentropy()
    optimizer = SGDMomentum(lr=1e-3, momentum=0.9)
    model = Model(loss_function, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(32, 3, (3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())
    model.add(Conv2D(64))
    model.add(ReLU())
    model.add(Pooling2D())
    
    model.add(Conv2D(64))
    model.add(ReLU())
    model.add(BatchNorm())
    # model.add(Conv2D(128))
    model.add(Pooling2D())
    # model.add(Dropout(p=0.2))
    
    # model.add(Conv2D(256))
    # model.add(ReLU())
    # model.add(Pooling2D())
    # model.add(BatchNorm())
    
    model.add(Flatten())
    model.add(Dense(units=256))
    model.add(ReLU())
    model.add(Dropout(p=0.1))
    model.add(Dense(units=10))
    model.add(Softmax())
    
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        batch_size=64,
        epochs=7,
        shuffle=True,
        verbose=True,
        x_valid=x_valid,
        y_valid=y_valid,
    )

    # your code here /\
    return model


# ============================================================================
