from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.lib.stride_tricks import as_strided




class Layer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _forward(self):
        pass

    @abstractmethod
    def _backward(self):
        pass


class FC(Layer):
    def __init__(self, d_in, d_out) -> None:
        self.x_tmp = None  # 存储尚未更新的输入x
        self.d_in = d_in
        self.d_out = d_out
        self.w = {
            "val": np.random.normal(0.0, np.sqrt(2 / d_in), (d_in, d_out)),
            # "val" : np.ones((d_out, d_in))*2,
            "grad": 0
        }

        self.b = {'val': np.random.randn(d_out), 'grad': 0}

    def _forward(self, x):
        # x.shape = (batch, inchannels)
        # print(x.shape, self.w["val"].shape)
        y = np.tensordot(x, self.w["val"], ([1], [0])) + self.b['val']
        self.x_tmp = x
        # y.shape = (batch, outchannels)
        return y

    def _backward(self, dy):
        # compute loss
        batch, d_out = dy.shape
        assert (d_out == self.d_out)
        self.w["grad"] = np.tensordot(self.x_tmp, dy, [(0), (0)]) # 取平均
        self.b["grad"] = dy.sum(axis=0)
        assert self.w["grad"].shape == self.w["val"].shape
        assert self.b["grad"].shape == self.b["val"].shape

        # compute delta x
        dx = np.tensordot(dy, self.w["val"], [(1), (1)])
        return dx


class Conv2d(Layer):
    def __init__(self, c_in, c_out, k_size, stride=1, padding=1):
        super().__init__()
        self.x_tmp = None
        self.d_in = c_in
        self.d_out = c_out
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.kernel = {
            'val':
            np.random.normal(0.0, np.sqrt(2 / c_in),
                            [c_in, c_out, k_size, k_size]),
            'grad':
            0
        }
        self.b = {'val': np.random.randn(c_out), 'grad': 0}
        

    def im2col(self, x: np.ndarray) -> np.ndarray:
        """将image转化为用于卷积的矩阵

        Args:
            x (ndarray): size为(batch, c_in, xh, xw)的输入图像
            
        return:
            x_col: size-(batch, c_in, out_h, out_w, k_size, k_size)
        """
        batch, c_in, xh, xw = x.shape
        out_h = (xh - self.k_size) // self.stride + 1
        out_w = (xw - self.k_size) // self.stride + 1
        stride = (*x.strides[:-2], x.strides[-2] * self.stride,
                  x.strides[-1] * self.stride, *x.strides[-2:])
        x_col = as_strided(
            x, (*x.shape[:2], out_h, out_w, self.k_size, self.k_size),
            strides=stride)
        return x_col

    def _forward(self, x):
        batch, c_in, xh, xw = x.shape
        assert (c_in == self.d_in)
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)),
                       'constant',
                       constant_values=0)
        x_col = self.im2col(x_pad)
        # print(x_col.shape, self.kernel["val"].shape)
        y = np.tensordot(x_col, self.kernel["val"],
                         [(1, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)
        # y.shape = (batch, c_out, o_w, o_h)
        self.x_tmp = x_pad
        y =np.add(self.b["val"].reshape((1,self.d_out,1,1)),y)
        return y

    def _backward(self, dy):
        delta_padding = self.k_size - self.padding - 1
        dy_pad = np.pad(dy, ((0, 0), (0, 0), (delta_padding, delta_padding),
                             (delta_padding, delta_padding)),
                        'constant',
                        constant_values=0)
        batch, c_out, out_h, out_w = dy.shape
        assert (c_out == self.d_out)
        # compute grad
        stride = (*self.x_tmp.strides[:-2], *self.x_tmp.strides[-2:],
                  self.x_tmp.strides[-2] * self.stride,
                  self.x_tmp.strides[-1] * self.stride)
        x_col = as_strided(
            self.x_tmp,
            (*self.x_tmp.shape[:2], self.k_size, self.k_size, out_h, out_w),
            strides=stride)
        self.kernel['grad'] = (np.tensordot(x_col, dy,
                                           [(0, 4, 5),
                                            (0, 2, 3)]).transpose(0, 3, 1, 2))
        self.b["grad"] = np.sum(dy, axis=(0, 2, 3))
        assert self.kernel["grad"].shape == self.kernel["val"].shape
        assert self.b["grad"].shape == self.b["val"].shape

        # compute delta_x
        delta_kernel = np.rot90(self.kernel['val'], 2, (2, 3))  # 旋转卷积核180度
        dy_pad_col = self.im2col(dy_pad)  # 没考虑步长
        # print(dy_pad_col.shape, delta_kernel.shape)
        # dy_pad_col.shape = (batch, c_out, in_w, in_h, k_size, k_size)
        delta_x = np.tensordot(dy_pad_col,
                               delta_kernel,
                               axes=[(1, 4, 5),
                                     (1, 2, 3)]).transpose(0, 3, 1, 2)
        return delta_x 


class MaxPooling(Layer):
    def __init__(self, pooling_size, stride=2, padding=0):
        self.pooling_size = pooling_size
        self.stride = stride
        self.padding = padding
        self.max_index = None

    def _forward(self, x):
        x: np.ndarray
        batch, cin, xw, xh = x.shape
        stride = (*x.strides[:-2], x.strides[-2] * self.pooling_size,
                  x.strides[-1] * self.pooling_size, *x.strides[-2:])
        x_col = as_strided(
            x, (batch, cin,int(xw / self.pooling_size), int(xh / self.pooling_size),
                self.pooling_size, self.pooling_size),
            strides=stride)
        ndim = x_col.shape.__len__()
        y = x_col.max(axis=(ndim - 2, ndim - 1))
        # 标记最大位置
        ndim = y.shape.__len__()
        self.max_index = np.equal(
            y.repeat(self.pooling_size,
                     axis=ndim - 2).repeat(self.pooling_size, axis=ndim - 1),
            x)  #扩充后两维，得到mask
        return y

    def _backward(self, dy):
        ndim = dy.shape.__len__()
        dy = dy.repeat(self.pooling_size, axis=ndim - 2).repeat(self.pooling_size,
                                                           axis=ndim - 1)
        dx = dy * self.max_index
        return dx


class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.x_tmp = None

    def _forward(self, x):
        self.x_tmp = x
        return np.maximum(0, x)

    def _backward(self, dy):
        dx = dy.copy()
        dx[self.x_tmp <= 0] = 0
        return dx


class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        # x.shape(batch,10)
        Z = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], 1))  # 防止指数溢出
        y = (Z) / (Z).sum(axis=1).reshape(x.shape[0], 1)
        return y
    
    def _backward(self):
        pass

    # softmax的反向传播和交叉熵结合，不用单独写
