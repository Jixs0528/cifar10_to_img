import imageio.v3 as imageio
import numpy as np
import pickle
import os


# 解压缩，返回类型为字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


# 标签
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 对应标签的图片数量
counts = [0] * 10
# 创建对应标签的文件夹
for item in labels:
    if not os.path.exists("./data/images/train/{}".format(item)):
        os.makedirs("./data/images/train/{}".format(item))
for j in range(1, 6):  # 遍历训练数据集
    path = "data_batch_" + str(j)
    raw_data = unpickle(path)

    for i in range(0, 10000):  # 遍历数据集里的所有数据
        img = np.reshape(raw_data['data'][i], (3, 32, 32))  # 读取图片
        img = img.transpose([1, 2, 0])
        imageio.imwrite(
            "./data/images/train/{}/{}_{}.png".format(labels[raw_data['labels'][i]], labels[raw_data['labels'][i]],
                                                      counts[raw_data['labels'][i]]), img)  # 保存图片
        counts[raw_data['labels'][i]] += 1  # 对应标签的数量加1
    print(path + " loaded.")
