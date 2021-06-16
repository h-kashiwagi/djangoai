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

model = Sequential()
#第１ブロック？
#出力　1ニューロンあたり（image_size,image_size,3)の3次元配列
#ニューロン数32
#カーネルサイズ(3,3)
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_size,image_size,3))) 
model.add(Conv2D(32,(3,3),activation='relu'))
#Pooling層1
#縮小対象領域は2×2
model.add(MaxPooling2D(pool_size=(2,2))) #Pooling
#ドロップアウト層
model.add(Dropout(0.25)) 

#第2ブロック？
#ニューロン数64
#カーネルサイズ(3,3)
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
#Pooling層1
#縮小対象領域は2×2
model.add(MaxPooling2D(pool_size=(2,2))) #Pooling
#ドロップアウト層
model.add(Dropout(0.25)) #Dropout


#Flatten層
model.add(Flatten())        
#中間層
#全結合層
#ニューロン層 256
model.add(Dense(256,activation='relu'))  

#ドロップアプト層
#出力　1ユニットあたり
model.add(Dropout(0.5))     

#出力層
#ニューロン数　num_classses
model.add(Dense(num_classes,activation='softmax'))  #softmax

#opt = SGD(lr=0.01) #rmsprop,adam ※「トレーニングの実行とチューニングでコメントアウト

# Sequentialオブジェクトのコンパイル
# 学習方法としてadam（最適化手法）
opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=32,epochs=30)

score = model.evaluate(X_test,y_test,batch_size=32)
