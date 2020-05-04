import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image

# 可以將所有網路層放到一個 list 當中，並做為 tf.keras.Sequential 的參數
# 而這個 list 同樣有順序性，第一個需要定義輸入大小，最後一個為輸出層
model = tf.keras.Sequential([layers.Dense(64, activation='relu', input_shape=(784,)),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(10, activation='softmax')])
# 產生網路拓撲圖
plot_model(model, to_file='Functional_API_Sequential_Model.png')
# 秀出網路拓撲圖
Image('Functional_API_Sequential_Model.png')                            