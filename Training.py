import csv
import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger
from tensorflow.keras.layers import Conv2D, Reshape, Activation, BatchNormalization
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon
from keras.regularizers import l2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35
# 96, 128, 160, 192, 224
IMAGE_SIZE = 224
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 10
MULTI_PROCESSING = True
THREADS = 4

# train 과 validation 경로
TRAIN_CSV = "train_shuffle.csv"  # train data .csv
VALIDATION_CSV = "validation_shuffle.csv"  # validation data .csv
TEST_CSV = "test_uniform.csv"  # test data .csv
MODEL_SAVE_FOLDER_PATH = './model/'  # 저장될 모델 디렉토리 설정

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    # row[i+1] = int(r)
                    row[i+1] = float(r)

                path, image_height, image_width, x0, y0, width, height, _, _ = row
                self.coords[index, 0] = float((x0 * IMAGE_SIZE / image_width) / IMAGE_SIZE)  # xmin
                self.coords[index, 1] = float((y0 * IMAGE_SIZE / image_height) / IMAGE_SIZE)  # ymin
                self.coords[index, 2] = float((width * IMAGE_SIZE / image_width) / IMAGE_SIZE)  # width
                self.coords[index, 3] = float((height * IMAGE_SIZE / image_height) / IMAGE_SIZE)  # height

                # int형
                # self.coords[index, 0] = x0 * IMAGE_SIZE / image_width / image_width  # xmin
                # self.coords[index, 1] = y0 * IMAGE_SIZE / image_height  # ymin
                # self.coords[index, 2] = (x1 - x0) * IMAGE_SIZE / image_width  #width
                # self.coords[index, 3] = (y1 - y0) * IMAGE_SIZE / image_height #height
                # self.coords[index, 2] = width * IMAGE_SIZE / image_width  # width
                # self.coords[index, 3] = height * IMAGE_SIZE / image_height  # height

                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.convert('RGB')

            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, batch_coords

class Training(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        train_pos_count = 0
        train_neg_count = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)
            train_gt_len = len(gt)

            pred = np.maximum(pred, 0)
            ########################################################################################################
            # iou 계산
            diff_width = np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) - np.maximum(gt[:, 0], pred[:, 0])
            diff_height = np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) - np.maximum(gt[:, 1], pred[:, 1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:, 2] * gt[:, 3]
            area_pred = pred[:, 2] * pred[:, 3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            train_iou = np.round(intersection / (union + epsilon()), 4)
            ########################################################################################################
            for q in range(train_gt_len):
                if train_iou[q] >= 0.5: # iou threshold 0.5 지정
                    train_pos_count += 1
                else:
                    train_neg_count += 1

        train_data_count = 59521  # train_총 갯수
        train_acc = np.round(train_pos_count / train_data_count, 4)
        logs["train_acc"] = train_acc

        print(" - train_acc: {}".format(train_acc))


class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0
        val_pos_count = 0
        val_neg_count = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)
            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]
            val_gt_len = len(gt)

            pred = np.maximum(pred, 0)
            ########################################################################################################
            # iou 계산
            diff_width = np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) - np.maximum(gt[:, 0], pred[:, 0])
            diff_height = np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) - np.maximum(gt[:, 1], pred[:, 1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:, 2] * gt[:, 3]
            area_pred = pred[:, 2] * pred[:, 3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            val_iou = np.round(intersection / (union + epsilon()), 4)
            ########################################################################################################
            for q in range(val_gt_len):
                if val_iou[q] >= 0.5: # iou threshold 0.5 지정
                    val_pos_count += 1
                else:
                    val_neg_count += 1

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        val_data_count = 14879  # validation 총 갯수
        val_acc = np.round(val_pos_count / val_data_count, 4)
        logs["val_acc"] = val_acc

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {} - val_acc: {}".format(iou, mse, val_acc))


class Test_set(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        test_pos_count = 0
        test_neg_count = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            test_gt_len = len(gt)
            pred = np.maximum(pred, 0)
            ########################################################################################################
            # iou 계산
            diff_width = np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) - np.maximum(gt[:, 0], pred[:, 0])
            diff_height = np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) - np.maximum(gt[:, 1], pred[:, 1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:, 2] * gt[:, 3]
            area_pred = pred[:, 2] * pred[:, 3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            test_iou = np.round(intersection / (union + epsilon()), 4)
            ########################################################################################################
            for q in range(test_gt_len):
                if test_iou[q] >= 0.5: # iou threshold 0.5 지정
                    test_pos_count += 1
                else:
                    test_neg_count += 1

        test_data_count = 1245  # test 총 갯수
        test_acc = np.round(test_pos_count / test_data_count, 4)
        logs["test_acc"] = test_acc

        print(" - test_acc: {}".format(test_acc))

def create_model(trainable=False):
    # pre-trained 된 moblienetv2 아키텍처 이용
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights='imagenet', classes=4)
    model.summary()

    # to freeze layers 레이어 동결(가중치 그대로 사용)
    for layer in model.layers:
        layer.trainable = trainable
    
    # 입력 Task에 맞는 딥러닝 모델 변경 가능
    block = model.get_layer("block_16_project_BN").output
    x = Conv2D(112, padding="same", kernel_size=3, strides=1, activation="relu")(block)
    x = Conv2D(112, padding="same", kernel_size=3, strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(4, kernel_size=7, name="coords")(x) # 사이즈 224 경우
    # x = Conv2D(4, kernel_size=3, name="coords")(x) # 사이즈 96 경우
    x = Reshape((4,))(x)

    return Model(inputs=model.input, outputs=x)

def main():
    model = create_model()
    model.summary()

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))
    test_datagen = Test_set(generator=DataGenerator(TEST_CSV))
    train_acc = Training(generator=DataGenerator(TRAIN_CSV))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[])

    # val_iou 값을 모니터링 하면서 최적 weight일때 저장

    checkpoint = ModelCheckpoint("model-{epoch:02d}-{val_iou:.3f}.hdf5", monitor="val_iou", verbose=1,
                                 save_best_only=True, mode="max", period=1)
    # val_iou 값을 모니터링 하면서
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    # val_iou 값을 모니터링 하면서 learning_rate 조절
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
    # 학습시 loss값 또는 callback 함수에 사용되는 train_acc, val_acc, test_acc 로그기록 남김
    cv_csv_logger = CSVLogger('parrot_localization_mobilenetv2.csv')
    # 모델 학습
    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, test_datagen, train_acc, checkpoint, reduce_lr, stop, cv_csv_logger],
                        workers=THREADS,
                        use_multiprocessing=MULTI_PROCESSING,
                        shuffle=True,
                        verbose=1)

if __name__ == "__main__":
    main()
