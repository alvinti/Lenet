import numpy as np
from Net import Lenet
import Optimizer
import mnist
import matplotlib.pyplot as plt

def get_batch(x, y, batch_size, class_num):
    x_index = 0
    dataset_size = x.shape[0]
    while x_index < dataset_size:
        batch_x = x[x_index:min(x_index + batch_size, dataset_size), ]
        batch_label = y[x_index:min(x_index + batch_size, dataset_size)]
        # make one-hot
        y_label = np.zeros((batch_label.shape[0], class_num))
        y_label[np.arange(batch_label.shape[0]), batch_label] = 1
        x_index += batch_size
        yield batch_x, y_label


def cross_entropy(y_pred, y_label):
    """计算交叉熵损失

    Args:
        y_pred (ndarray): y.shape=(batch, y_pred)
        y_label (ndarray): y.shape=(batch, y_label), ylabel是one-hot
    """
    tmp = y_pred * y_label
    tmp[tmp == 0] = 1  # 方便log计算
    return -(np.log(tmp).sum() / y_pred.shape[0])  # loss取平均


if __name__ == "__main__":
    # 读取数据
    train_img = mnist.train_images().reshape((60000, 1, 28, 28))
    train_img = (train_img - np.mean(train_img))/255.0 # 标准化
    train_label = mnist.train_labels()
    test_img = mnist.test_images()
    test_img = test_img.reshape((test_img.shape[0], 1, 28, 28))
    test_img = (test_img - np.mean(test_img))/255.0
    test_label = mnist.test_labels()

    lr = 0.001
    momentum = 0.99
    batch_size = 64
    iter = 10

    module = Lenet()
    SGD = Optimizer.SGD(module.parameters(), lr, momentum)
    losses = []
    x_index = 0
    for i in range(iter):
        for x_batch, y_batch in get_batch(train_img,
                                          train_label,
                                          batch_size,
                                          class_num=10):
            y_pred = module.forward(x_batch)
            loss = cross_entropy(y_pred, y_batch)
            losses.append(cross_entropy(y_pred, y_batch))
            module.backward((y_pred - y_batch) / batch_size)
            SGD.step()

        y_final = module.forward(test_img)
        y_result = np.argmax(y_final, axis=1)
        precision = np.count_nonzero((y_result == test_label)) / test_label.shape[0]
        print("\n完成度：%s%%, 交叉熵损失: %s, 测试集准确率： %s" % (100 * i / iter, loss, precision))

    # 计算准确率
    y_final = module.forward(train_img)
    y_result = np.argmax(y_final, axis=1)
    precision = np.count_nonzero((y_result == train_label)) / train_label.shape[0]
    print(precision)
    
    plt.plot(losses)
    plt.show()
