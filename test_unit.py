import math
import mnist
import numpy as np
import unittest
from unittest import TestCase
import Layer 
import torch.nn as nn
import torch
import Optimizer
from test_code import layer_test, Net_test, test
from train_net import cross_entropy


class test_layer(TestCase):
    def setUp(self):
        self.conv1 = Layer.Conv2d(1,6,k_size=5,weight=1, padding=2)
        self.relu1 = Layer.Relu()
        self.maxPol1 = Layer.MaxPooling(2)
        self.conv2 = Layer.Conv2d(6,16,5,padding=0)
        self.fc1 = Layer.FC(1176,10)
        self.softmax = Layer.SoftMax()
        
        self.conv1T = layer_test.Conv(1,6,28,28,5,1, 2)
        self.conv2T = nn.Conv2d(6,16,5,padding=0)
        self.reluT = layer_test.ReLU()
        self.maxPT = layer_test.MaxPool(2,2)
        self.fcT = layer_test.FC(1176,10)
        self.softT = layer_test.Softmax()
        
        self.sgd = Optimizer.SGD([self.conv1.kernel, self.conv1.b, self.fc1.w, self.fc1.b],lr = 0.001, momentum=0.99)
        self.sgdT = Net_test.SGDMomentum([self.conv1T.W, self.conv1T.b, self.fcT.W, self.fcT.b],momentum=0.99, reg=0)
        
    # def test_net_forward(self):
    #     conv1_test = np.random.randn(6,1,28,28)
        
    #     b = self.conv1._forward(conv1_test)
    #     a = self.conv1T._forward(conv1_test)
    #     self.assertTrue(np.equal(a,b).all())
    #     a = self.maxPol1._forward(a)
    #     b = self.maxPT._forward(b)
    #     self.assertTrue(np.equal(a,b).all())
    #     a = self.relu1._forward(a)
    #     b = self.reluT._forward(b)
    #     self.assertTrue(np.equal(a,b).all())
        
    #     self.x_shape = a.shape
    #     a = np.reshape(a, (a.shape[0], -1),'C')
    #     b = np.reshape(b, (b.shape[0], -1),'C')
    #     self.assertEqual(a.sum(),b.sum())
        
        
    #     a = self.fc1._forward(a)
    #     b = self.fcT._forward(b)
    #     # print(a.shape, b.shape)
        
    #     a = self.softmax._forward(a)
    #     b = self.softT._forward(b)
    #     # self.assertEqual(a.sum(),b.sum())
        
        
    #     dy_test = np.ones((6,10))
    #     a = self.fc1._backward(dy_test)
    #     b = self.fcT._backward(dy_test)
    #     self.assertEqual(a.sum(),b.sum())
        
    #     a = a.reshape(self.x_shape)
    #     b = b.reshape(self.x_shape)
    #     a = self.relu1._backward(a)
    #     b = self.reluT._backward(b)
        
    #     self.assertEqual(a.sum(),b.sum())
        
    #     a = self.maxPol1._backward(a)
    #     b = self.maxPT._backward(b)
    #     # self.assertTrue(np.equal(a,b).all())
    #     self.assertEqual(a.sum(),b.sum())
    #     a = self.conv1._backward(a)
    #     b = self.conv1T._backward(b)
    #     # self.assertEqual(a.sum(),b.sum())
    #     # self.assertTrue(np.equal(a,b).all())
    #     print(self.fc1.w["grad"].shape, self.fcT.W["grad"].shape)
    #     self.assertEqual(self.conv1.kernel["grad"].sum(), self.conv1T.W["grad"].sum())
        
    #     self.sgd.step()
        
    #     self.sgdT.step()
        
    #     self.assertEqual(self.conv1.kernel["val"].sum(), self.conv1T.W["val"].sum())
    #     self.assertEqual(self.conv1.b["val"].sum(), self.conv1T.b["val"].sum())
    #     self.assertEqual(self.fc1.b["val"].sum(), self.fcT.b["val"].sum())
    #     self.assertEqual(self.fc1.w["val"].sum(), self.fcT.W["val"].sum())

    def test_CE(self):
        a = np.random.normal(5,1, (5,10))
        b = np.zeros((5,10))
        b[np.arange(5), np.random.randint(0,9)] = 1
        ce = test.CrossEntropyLoss()
        self.assertEqual(ce.get(a,b)[0], cross_entropy(a,b))
        
        
    

if __name__ =='__main__':
        unittest.main()
        
        