import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("dataset.csv", delimiter=";")
dups = dataset.duplicated(subset=["X", "Y", "Z"])
dataset = dataset[~dups]
# dataset = np.loadtxt("C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\yours.csv", delimiter=";", encoding='utf-8', dtype=None)

angles = 6


X_super_test = dataset.iloc[90000:, :3].values
Y_super_test = dataset.iloc[90000:, 3:].values
sss = X_super_test[0:1, :3]

X_super = dataset.iloc[:10000, :3].values
Y_super = dataset.iloc[:10000, 3:].values


scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_super)
X_val = scaler.transform(X_super_test)

# TODO no Y transform
scaler = MinMaxScaler(feature_range=(-1, 1))
Y_train = scaler.fit_transform(Y_super)
Y_val = scaler.transform(Y_super_test)

model = Sequential()

# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X_super, Y_super, test_size=0.15, shuffle=1, random_state=0)
#
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_train = scaler.fit_transform(X_train)
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_test = scaler.fit_transform(X_test)
#
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=0)#0.25

DROPOUT_RATE = 0.01
model.add(Dense(3, activation="linear", input_shape=(3,)))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(100, activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(6, activation="linear"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.05),
    loss=tf.losses.mean_squared_error,
    metrics=["accuracy"],
)

model.summary()
# early_stop = EarlyStopping(monitor='accuracy', patience=15)
train_model = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    verbose=2,
    # validation_data=(X_super_test, Y_super_test),
    callbacks=[tf.keras.callbacks.TensorBoard("logs/1/train")],
)

y_pred = model.predict(X_super_test[0:1, :3])

training_loss = train_model.history["loss"]
test_loss = train_model.history["val_loss"]

# Get training and test accuracy histories
training_acc = train_model.history["accuracy"]
test_acc = train_model.history["val_accuracy"]

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
fig, (p1, p2) = plt.subplots(2)
fig.set_size_inches(5, 10)

p1.set_title("Loss")
p1.plot(epoch_count, training_loss)
p1.plot(epoch_count, test_loss)
p1.legend(["Train", "Test"])
p1.set_xlabel("Epoch")
p1.set_ylabel("Loss value")

# Visualize accuracy history
p2.set_title("Acuracy")
p2.plot(epoch_count, training_acc)
p2.plot(epoch_count, test_acc)
p2.legend(["Train", "Test"])
p2.set_xlabel("Epoch")
p2.set_ylabel("Accuracy value")

plt.show()


# model = load_model(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')
model.save(r"C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5")

aa = "dasdasd"
