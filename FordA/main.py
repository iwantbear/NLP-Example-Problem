import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report

save_dir = "/Users/hwang-gyuhan/Desktop/Collage/4-1/자연어처리/Mid/first"
os.makedirs(save_dir, exist_ok=True)

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

# 클래스별 데이터를 시각화하여 저장
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.savefig(os.path.join(save_dir, "forda_class_plot.png"))
plt.close()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=x_train.shape[1:])

epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = keras.models.load_model("best_model.h5")
test_loss, test_acc = model.evaluate(x_test, y_test)
train_loss, train_acc = model.evaluate(x_train, y_train)  # Train accuracy and loss 추가

print("Train accuracy", train_acc)
print("Train loss", train_loss)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

# 예측값 생성
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# classification report 계산
report = classification_report(y_test, y_pred, output_dict=True)

# f1-score, precision, recall, accuracy, macro avg 추가
filtered_report = {
    "class_{}".format(c): {
        "precision": report[str(c)]["precision"],
        "recall": report[str(c)]["recall"],
        "f1-score": report[str(c)]["f1-score"]
    }
    for c in range(num_classes)
}

# 전체 보고서를 파일로 저장
with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write("Classification Report (Filtered):\n")
    for class_id, metrics in filtered_report.items():
        f.write(f"{class_id}:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value}\n")
        f.write("\n")

# Train과 Test의 accuracy와 loss를 그래프에 함께 그리기
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric], label="train " + metric)
plt.plot(history.history["val_" + metric], label="val " + metric)
plt.title("Model " + metric)
plt.ylabel(metric)
plt.xlabel("Epoch")
plt.legend(loc="best")
plt.savefig(os.path.join(save_dir, "training_and_validation_accuracy.png"))
plt.close()

# Train과 Test의 accuracy 및 loss를 그래프에 그리기
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="best")
plt.savefig(os.path.join(save_dir, "training_and_validation_loss.png"))
plt.close()