from django.db import models

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

# Create your models here.

class Photo(models.Model):
    image = models.ImageFiled(upload_to='photos')

    IMAGE_SIZE = 224 #画像サイズ
    MODEL_FILE_PATH = './clothes/ml_models/vgg16_trainsfer.h5' #モデルファイル
    
    classes = ["data","clothes"]
    num_classes = len(classes)
 
'''
models.py
      image = models.ImageField(
        upload_to='files/',
        verbose_name='添付画像',
        height_field='url_height',
        width_field='url_width',
    )

'''