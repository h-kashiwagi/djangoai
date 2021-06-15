#モデルの定義とトレーニングのコード
#CNNはConvolutional Neural Network 「画像の深層学習」と言えばCNN

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam
#from tensorflow.keras.utils import np_utils ←バージョン変更で廃止
from keras.utils import np_utils
from tensorflow.python.keras.applications.vgg16 import VGG16 


classes = ["apples","clothes"]
num_classes = len(classes)
image_size = 224 #変更した箇所 

#データの読み込み
X_train,X_test,y_train,y_test = np.load("./imagefile_224.npy",allow_pickle=True) #ファイルをロード
#numpy v1.16.3 より、numpy.load()関数の挙動が変更されたためallow_pickle=Trueがないとエラーになる
y_train = np_utils.to_categorical(y_train,num_classes) #ワンホットベクトル形式に変換
y_test = np_utils.to_categorical(y_test,num_classes)
X_train = X_train.astype("float") /225.0
X_test = X_test.astype("float") /225.0

#モデル定義
#畳み込み

#学習済みモデルで代用
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))
print('Model loaded')
model.summary()




#中間層と全結合は再学習


#opt = SGD(lr=0.01) #rmsprop,adam ※「トレーニングの実行とチューニングでコメントアウト

'''
opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=30)

score = model.evaluate(X_test,y_test,batch_size=32)
'''