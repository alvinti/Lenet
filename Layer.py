from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.lib.stride_tricks import as_strided


class Layer(ABCMeta):
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
    def __init__(self, in_channels, out_channels) -> None:
        self.tmp = None  # 存储尚未更新的输入x
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = {
            "val":
            np.random.normal(0.0, np.sqrt(2 / in_channels),
                             (in_channels, out_channels)),
            "grad":
            0
        }

        self.b = {'val': np.random.randn((out_channels)), 'grad': 0}

    def _forward(self, x):
        # x.shape = (batch, x)
        y = np.tensordot(x, self.w, ([0], 1)) + self.b
        self.tmp = x
        return y

    def _backward(self, d_out):
        # todo
        return super()._backward()


class Conv2d(Layer):
    def __init__(self, c_in, c_out, x_h, x_w, k_size, stride, padding):
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
                             [c_out, c_in, k_size, k_size]),
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
        out_h = (x.shape[2] - self.k_size) // self.stride + 1
        out_w = (x.shape[3] - self.k_size) // self.stride + 1
        stride = (*x.strides[:-2], x.strides[-2] * self.stride,
                  x.strides[-1] * self.stride, *x.strides[-2:])
        x_col = as_strided(
            x, (x.shape[:2], out_h, out_w, self.k_size, self.k_size),
            strides=stride)
        return x_col

    def _forward(self, x):
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)),
                       'constant',
                       constant_values=0)
        x_col = self.im2col(x_pad)
        y = np.tensordot(x_col, self.kernel,
                         [(1, 4, 5), (1, 2, 3)]).transpose(0, 3, 1, 2)
        # y.shape = (batch, c_out, o_w, o_h)
        self.x_tmp = x_pad
        return y

    def _backward(self, dy):
        delta_padding = self.k_size - self.padding
        dy_pad = np.pad(dy, ((0, 0), (0, 0), (delta_padding, delta_padding),
                             (delta_padding, delta_padding)),
                        'constant',
                        constant_values=0)
        batch, c_out, out_h, out_w = dy.shape
        # compute grad
        stride = (*self.x_tmp.strides[:-2], *self.x_tmp.strides[-2:],
                  self.x_tmp.strides[-2] * self.stride,
                  self.x_tmp.strides[-1] * self.stride)
        x_col = as_strided(
            self.x_tmp,
            (self.x_tmp.shape[:2], self.k_size, self.k_size, out_h, out_w),
            strides=stride)
        self.w['grad'] = np.tensordot(x_col, dy,
                                      [(0, 4, 5),
                                       (0, 2, 3)]).transpose(3, 0, 1, 2)
        self.b["grad"] = np.sum(dy, axis=(0, 2, 3))

        # compute delta_x
        delta_kernel = np.rot90(self.kernel, 2, (2, 3))  # 旋转卷积核180度
        dy_pad_col = self.im2col(dy_pad)  # 没考虑步长
        # dy_pad_col.shape = (batch, c_out, in_w, in_h, k_size, k_size)
        delta_x = np.tensordot(dy_pad_col,
                               delta_kernel,
                               axes=[(1, 4, 5),
                                     (0, 2, 3)]).transpose(3, 0, 1, 2)
        return delta_x

    def _updatePara(self):
        # todo: 乘上lr，减去grad
        pass


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
            x, (batch, cin, xw / self.pooling_size, xh / self.pooling_size,
                self.pooling_size, self.pooling_size),
            strides=stride)
        ndim = x_col.shape.__len__()
        y = x_col.max(axis=(ndim - 2, ndim-1))
        # 标记最大位置
        ndim = y.shape.__len__()
        self.max_index = np.equal(
            y.repeat(2, axis=ndim - 2).repeat(2, axis=ndim-1), x) #扩充后两维，得到mask
        return y
    
    def _backward(self, dy):
        ndim = dy.shape.__len__()
        dy.repeat(2, axis=ndim - 2).repeat(2, axis=ndim-1)
        dx = dy*self.max_index
        return dx

class Relu(Layer):
    def __init__(self):
        super().__init__()
        