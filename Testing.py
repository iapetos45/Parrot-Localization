import glob
import cv2
import numpy as np
import csv
from localization_train import IMAGE_SIZE
from tensorflow.keras.applications.mobilenet_v2 import preprocess_import
from tensorflow.keras.models import load_model
from tensorflow.keras.models import backend as k
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#동영상 입력 경로와 openCV 함수를 사용하기 위한 작업
file_avi = './test_set.avi'
cap = cv2.VideoCapture(file_avi)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
ret, frame = cap.read()

# Ground truth 입력 경로
input_csv_dir = './image_test' # 입력 영상 폴더 경로
img_csv = glob.glob(input_csv_dir+'/*.csv')

def main()
model = load_model('model-22-0.714.hdf5')
model.sumary()
print('Loaded 완료')

# grad-cam 사용하기 위한 사전 작업
out = model.output[:,]
last_conv_layer = model_get_layer('activation')
grads

# .csv에 입력하여 list에 넣기

while True:
    ret, frame = cap.read()
    if ret:
        image_height, image_width, _ = frame.shape
        img = cv2.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # grad-cam input ######
        cam_image = np.expand_dims(image, axis=0)
        cam_image = preprocess_input(cam_image)
        ###
        feat_scaled = preprocessing_input(np.array(image, dtype=np.float32))
        regin = model.predict(x=np.array([feat_scaled]))[0] # x, y, width, height

        y0_pred = int((region[0]))
