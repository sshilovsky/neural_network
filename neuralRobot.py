import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt

dataset = np.loadtxt("C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\yours.csv", delimiter=";", encoding='utf-8', dtype=None)

angles = 6

X_super = dataset[:90000,:3]
Y_super = dataset[:90000,3:]

X_super_test = dataset[90000:,:3]
Y_super_test = dataset[90000:,3:]
sss = X_super_test[0:1, :3]
model = Sequential()

# The Input Layer :
model.add(Dense(128, kernel_initializer='normal', input_dim = X_super.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))

# The Output Layer :
model.add(Dense(angles, kernel_initializer='normal', activation='linear'))

model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

model.summary()
early_stop = EarlyStopping(monitor='acc', patience=15)
train_model=model.fit(X_super, Y_super, epochs=300, validation_split =0.2, callbacks=[early_stop])

y_pred = model.predict(X_super_test[0:1, :3])

training_loss = train_model.history['loss']
test_loss = train_model.history['val_loss']

# Get training and test accuracy histories
training_acc = train_model.history['accuracy']
test_acc = train_model.history['val_accuracy']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.figure()
plt.title('Loss')
plt.plot(epoch_count, training_loss)
plt.plot(epoch_count, test_loss)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()

# Visualize accuracy history
plt.figure()
plt.title('Acuracy')
plt.plot(epoch_count, training_acc)
plt.plot(epoch_count, test_acc)
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy value')
plt.show()


#model = load_model(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')
model.save(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')

aa = 'dasdasd'