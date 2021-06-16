#NumPy形式でデータを生成して保存

#モジュールのインポート
from PIL import Image #Imageクラス
import os
import glob
import numpy as np
from sklearn import model_selection 

#パラメーターの初期化
classes = ["apples","clothes"]
#classes = ["car","motorbike"]  #リストを作成
num_classes = len(classes) 
image_size = 224 #変更した箇所


#画像の読み込みとNumPy配列への変換
X = [] #リスト
Y = [] #リスト

for index,classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i,file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image) /255.0 
        

        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)  
xy = (X_train, X_test, y_train, y_test) 
np.save("./imagefile_224.npy", xy) 
