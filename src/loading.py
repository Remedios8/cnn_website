from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.misc
from keras import backend as K
from PIL import Image
import struct
import matplotlib.pyplot as plt
# plot 4 images as gray scale

print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("mnist_model.h5")
print("Загрузка сети завершена")


# Загружаем данные
nb_classes = 72
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128
np.random.seed(42)
ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(nb_classes), 160)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)


# Компилируем загруженную модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Оцениваем качество обучения сети загруженной сети на тестовых данных
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)

path=r'C:\Users\Настя\Downloads\нейронки практика\mnist\2.png'
im = Image.open(path)
im_grey = im.convert('L')
im_array = np.array(im_grey)
im_array=np.reshape(im_array,(1, 784)).astype('float32')

# Инвертируем изображение
x = 255 - im_array
# Нормализуем изображение
x /= 255

# prediction = loaded_model.predict(x)

# prediction=np.argmax(prediction, axis=1)
# print(prediction)

k = np.array(X_train[0]) 
print(k.shape)
y= k.reshape(1,32,32,1)
print(y.shape)
prediction = loaded_model.predict(x)
prediction=np.argmax(prediction, axis=1)
print(prediction)