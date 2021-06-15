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


classes = ["apples","clothes"]
num_classes = len(classes)
image_size = 150 

X_train,X_test,y_train,y_test = np.load("./imagefile.npy",allow_pickle=True) #ファイルをロード
#numpy v1.16.3 より、numpy.load()関数の挙動が変更されたためallow_pickle=Trueがないとエラーになる
y_train = np_utils.to_categorical(y_train,num_classes) #ワンホットベクトル形式に変換
y_test = np_utils.to_categorical(y_test,num_classes)

#モデル定義
#畳み込み
#第１ブロック？
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_size,image_size,3))) 
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#第2ブロック？
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))



model.add(Flatten())        #Flatten
model.add(Dense(256,activation='relu')) #中間層
model.add(Dropout(0.5))     #全結合(2)?
model.add(Dense(num_classes,activation='softmax'))  #softmax

#opt = SGD(lr=0.01) #rmsprop,adam ※「トレーニングの実行とチューニングでコメントアウト

opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=30)

score = model.evaluate(X_test,y_test,batch_size=32)
