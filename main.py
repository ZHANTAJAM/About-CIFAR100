import cv2
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar100_to_images():
    tar_dir = '/home/zhantajam/cifar-100-python'  # 原始数据库目录
    train_root_dir = '/home/zhantajam/cifar-100-python/TRAIN/'  # 训练集图片保存目录
    test_root_dir = '/home/zhantajam/cifar-100-python/TEST/'  # 测试集图片保存目录
    if not os.path.exists(train_root_dir):
        os.makedirs(train_root_dir)
    if not os.path.exists(test_root_dir):
        os.makedirs(test_root_dir)

    # 加载 label 对应的类别名称
    meta_Name = os.path.join(tar_dir, "meta")
    Meta_dic = unpickle(meta_Name)
    fine_label_names = Meta_dic[b'fine_label_names']

    # 生成训练集图片
    dataName = os.path.join(tar_dir, "train")
    Xtr = unpickle(dataName)
    print(dataName + " 正在加载...")
    for i in range(Xtr[b'data'].shape[0]):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = os.path.join(train_root_dir, str(Xtr[b'fine_labels'][i]) + '_' + fine_label_names[Xtr[b'fine_labels'][i]].decode('utf-8') + '_' + str(i) + '.jpg')
        cv2.imwrite(picName, img)
    print(dataName + " 加载完成.")

    # 生成测试集图片
    testXtr = unpickle(os.path.join(tar_dir, "test"))
    print("test_batch 正在加载...")
    for i in range(testXtr[b'data'].shape[0]):
        img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = os.path.join(test_root_dir, str(testXtr[b'fine_labels'][i]) + '_' + fine_label_names[testXtr[b'fine_labels'][i]].decode('utf-8') + '_' + str(i) + '.jpg')
        cv2.imwrite(picName, img)
    print("test_batch 加载完成.")

cifar100_to_images()
