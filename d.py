import random
 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend
 
from load_data import load_dataset, resize_image, IMAGE_SIZE	# 这里load_data是引入上面预处理代码
 
 
class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据集加载路径
        self.path_name = path_name
        # 当前库采取的维度顺序
        self.input_shape = None
 
    # 加载数据集并按照交叉验证划分数据集再开始相关的预处理
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        images, labels = load_dataset(self.path_name)
        train_images, valid_images, train_labels, valid_lables = train_test_split(images,
                                                                                  labels,
                                                                                  test_size=0.2,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images,
                                                          labels,
                                                          test_size=0.3,
                                                          random_state=random.randint(0, 100))
        # 当前维度顺序如果是'th'，则输入图片数据时的顺序为：channels, rows, cols; 否则：rows, cols, channels
        # 根据keras库要求的维度重组训练数据集
        if backend.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
 
        # 输出训练集、验证集、测试集数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')
 
        # 使用categorical_crossentropy作为损失，因此需要根据类别数量nb_classes将类别标签进行one-hot编码，分类类别为4类，所以转换后的标签维数为4
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_lables = np_utils.to_categorical(valid_lables, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
 
        # 像素数据浮点化以便进行归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')
 
        # 归一化
        train_images /= 255
        valid_images /= 255
        test_images /= 255
 
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_lables
        self.test_labels = test_labels
 
 
"""CNN构建
"""
 
 
class CNNModel:
    def __init__(self):
        self.model = None
 
    # 模型构建
    def build_model(self, dataset, nb_classes=2):
        # 构建一个空间网络模型（一个线性堆叠模型）
        self.model = Sequential()
        self.model.add(Convolution2D(96, 10, 10, input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
 
        self.model.add(Convolution2D(256, 5, 5, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
 
        self.model.add(Convolution2D(384, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
 
        self.model.add(Convolution2D(384, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
 
        self.model.add(Convolution2D(256, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
 
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
 
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
 
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        # 输出模型概况
        self.model.summary()
 
    def train(self, dataset, batch_size=10, nb_epoch=5, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.7, nesterov=True)  # SGD+momentum的训练器
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # 模型配置工作
        # 跳过数据提升
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一次
            # 其顺序生成一组数据，节省内存，该数据生成器其实就是python所定义的数据生成器
            datagen = ImageDataGenerator(featurewise_center=False,              # 是否使输入数据去中心化（均值为0）
                                         samplewise_center=False,               # 是否使输入数据的每个样本均值为0
                                         featurewise_std_normalization=False,   # 是否数据标准化（输入数据除以数据集的标准差）
                                         samplewise_std_normalization=False,    # 是否将每个样本数据除以自身的标准差
                                         zca_whitening=False,                   # 是否对输入数据施以ZCA白化
                                         rotation_range=20,                     # 数据提升时图片随机转动的角度(范围为0～180)
                                         width_shift_range=0.2,                 # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                                         height_shift_range=0.2,                # 垂直偏移幅度
                                         horizontal_flip=True,                  # 是否进行随机水平翻转
                                         vertical_flip=False                    # 是否进行随机垂直翻转
                                         )
            datagen.fit(dataset.train_images)
            self.model.fit_generator(datagen.flow(dataset.train_images,
                                                  dataset.train_labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images, dataset.valid_labels)
                                     )
 
    MODEL_PATH = './cascadeface.model.h5'
 
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
 
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print('%s: %.2f%%' % (self.model.metrics_names[1], score[1] * 100))
 
    # 识别人脸
    def face_predict(self, image):
        # 根据后端系统确定维度顺序
        if backend.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致，都为：IMAGE_SIZE * IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这里是预测单张图像
        elif backend.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
 
        # 归一化
        image = image.astype('float32')
        image /= 255
        # 给出输入属于各类别的概率
        result = self.model.predict_proba(image)
        print('result:', result)
        result = self.model.predict_classes(image)
        # 返回预测结果
        return result[0]
 
 
if __name__ == '__main__':
    dataset = Dataset('./face/')
    dataset.load()
    model = CNNModel()
    model.build_model(dataset)
    # 先前添加的测试build_model()函数的代码
    model.build_model(dataset)
    # 测试训练函数的代码
    model.train(dataset)
 
if __name__ == '__main__':
    dataset = Dataset('./face/')
    dataset.load()
    model = CNNModel()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='./model/cascadeface.model.h5')
 
if __name__ == '__main__':
    dataset = Dataset('./face/')
    dataset.load()
    # 评估模型
    model = CNNModel()
    model.load_model(file_path='./model/cascadeface.model.h5')
    model.evaluate(dataset)
 