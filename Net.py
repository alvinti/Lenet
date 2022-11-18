import Layer
import numpy as np


class Network():
    def __init__(self, ) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Lenet(Network):
    def __init__(self):
        self.conv1 = Layer.Conv2d(1, 6, k_size=5, padding=2)
        self.relu1 = Layer.Relu()
        self.maxPol1 = Layer.MaxPooling(2)
        self.conv2 = Layer.Conv2d(6, 16, 5, padding=0)
        self.relu2 = Layer.Relu()
        self.maxPol2 = Layer.MaxPooling(2)
        self.fc1 = Layer.FC(400, 120)
        self.relu3 = Layer.Relu()
        self.fc2 = Layer.FC(120, 84)
        self.relu4 = Layer.Relu()
        self.fc3 = Layer.FC(84, 10)
        self.softmax = Layer.SoftMax()

        self.x_shape = None

    def forward(self, x):
        x = self.conv1._forward(x)
        x = self.relu1._forward(x)
        x = self.maxPol1._forward(x)
        x = self.conv2._forward(x)
        x = self.relu2._forward(x)
        x = self.maxPol2._forward(x)
        self.x_shape = x.shape
        x = np.reshape(x, (x.shape[0], -1), 'C')
        x = self.fc1._forward(x)
        x = self.relu3._forward(x)
        x = self.fc2._forward(x)
        x = self.relu4._forward(x)
        y = self.fc3._forward(x)
        y = self.softmax._forward(y)

        return y

    def backward(self, dy):
        dy = self.fc3._backward(dy)
        dy = self.relu4._backward(dy)
        dy = self.fc2._backward(dy)
        dy = self.relu3._backward(dy)
        dy = self.fc1._backward(dy)
        dy = dy.reshape(self.x_shape)
        dy = self.maxPol2._backward(dy)
        dy = self.relu2._backward(dy)
        dy = self.conv2._backward(dy)
        dy = self.maxPol1._backward(dy)
        dy = self.relu1._backward(dy)
        dy = self.conv1._backward(dy)

        return super().backward()

    def parameters(self) -> list:
        return [
            self.conv1.kernel, self.conv1.b, self.conv2.kernel, self.conv2.b,
            self.fc1.w, self.fc1.b, self.fc2.w, self.fc2.b
        ]
