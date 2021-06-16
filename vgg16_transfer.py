#モデルの定義とトレーニングのコード
#CNNはConvolutional Neural Network 「画像の深層学習」と言えばCNN

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import SGD,Adam
#from tensorflow.keras.utils import np_utils ←バージョン変更で廃止
from keras.utils import np_utils
from tensorflow.python.keras.applications.vgg16 import VGG16 

# 出力結果のクラスを定義
classes = ["apples","clothes"]
#classes = ["car","motorbike"]
num_classes = len(classes)
#VGG16用に画像サイズを変更
image_size = 224 

#データの読み込み
X_train,X_test,y_train,y_test = np.load("./imagefile_224.npy",allow_pickle=True) #ファイルをロード
#numpy v1.16.3 より、numpy.load()関数の挙動が変更されたためallow_pickle=Trueがないとエラーになる
y_train = np_utils.to_categorical(y_train,num_classes) #ワンホットエンコーディングに変換
y_test = np_utils.to_categorical(y_test,num_classes)
X_train = X_train.astype("float") /225.0
X_test = X_test.astype("float") /225.0

#モデル定義
#畳み込み式のニューラルネット

#学習済みモデルで代用
#include_top=Falseによって、VGGモデルから全結合層を削除（＝VGGモデルと学習済みの重みをロードしている（全結合層を除く））
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3)) #畳み込み13層＋全結合層３層＝16層のニューラルネットワーク
#VGG16のロードとサマリー確認
'''
print('Model loaded')
model.summary()
'''

#中間層と全結合は再学習

#全結合層の構築
#追加する全結合層もシーケンシャルで作成しておく
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256,activation='relu')) 
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

#inputs=全結合層を削除したVGG16モデルとoutputs=自前で作成したVGGモデルの全結合層を結合
#VGGと追加層を結合する
model = Model(inputs=model.input, outputs=top_model(model.output))
#サマリーを出力
# model.summary()

#CNNレイヤーのフリーズとトレーニングの実行
#14層目までのモデル重みを固定（VGGのモデル重みを用いる）
for layer in model.layers[:15]:
    layer.trainable = False



#AdamやSGDは最適化の関数
#opt = SGD(lr=0.01) #rmsprop,adam ※「トレーニングの実行とチューニングでコメントアウト
#opt = Adam()
opt = Adam(lr=0.0001) #lr: 0以上の浮動小数点数．学習率．
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
#model.fit(X_train,y_train,batch_size=32,epochs=30)
model.fit(X_train,y_train,batch_size=32,epochs=17)

score = model.evaluate(X_test,y_test,batch_size=32)