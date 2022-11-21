import numpy as np
from keras.layers import Dense, Layer, Lambda, InputLayer
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pygad
import pygad.kerasga

from forwardKinematicsKuka import RV

X_train = np.random.rand(100000, 3) * 1000


X_val = X_train[:30000]
X_train = X_train[30000:31000]

Y_train = X_train[:, :3]
Y_val = X_val[:, :3]


n_features = X_train.shape[1]

model = Sequential()

DROPOUT_RATE = 0.01
model.add(InputLayer((n_features,)))
model.add(Dense(3000, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(90, activation="sigmoid"))
model.add(Dense(6, activation="linear"))
model.add(Lambda(lambda q: tf.reshape(RV.getXYZ(q), (1, 3))))

# model.compile(
#     optimizer=tf.keras.optimizers.SGD(0.01),
#     loss=tf.losses.mean_squared_error,
#     metrics=["accuracy"],
# )

model.summary()
# early_stop = EarlyStopping(monitor='accuracy', patience=15)


def train_model(m):
    train_model = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        epochs=20,
        verbose=2,
        batch_size=X_train.shape[0],
        callbacks=[tf.keras.callbacks.TensorBoard("logs/1/train")],
    )

    training_loss = train_model.history["loss"]
    test_loss = train_model.history["val_loss"]

    # # Get training and test accuracy histories
    # training_acc = train_model.history["accuracy"]
    # test_acc = train_model.history["val_accuracy"]

    # # Create count of the number of epochs
    # epoch_count = range(1, len(training_loss) + 1)

    return training_loss, test_loss


keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=20)


def callback_generation(ga_instance):
    print(
        "Gen {}. AE={}".format(
            ga_instance.generations_completed, 1 / ga_instance.best_solution()[1]
        )
    )
    ga_instance.save("ga.pkl")
    model.save("model.h5")


def predict(model, solution, data):
    data = np.array(data)
    solution_weights = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )
    model.set_weights(solution_weights)
    predictions = model.predict(data, batch_size=data.shape[0], verbose=0)
    return predictions


def fitness_func(solution, sol_idx):
    predictions = predict(model, solution, X_train)
    mae = tf.keras.losses.MeanAbsoluteError()
    abs_error = mae(Y_train, predictions).numpy() + 0.00000001
    solution_fitness = 1.0 / abs_error
    return solution_fitness


print("go")
ga_instance = pygad.GA(
    num_generations=25000,
    num_parents_mating=2,
    initial_population=keras_ga.population_weights,
    fitness_func=fitness_func,
    on_generation=callback_generation,
    mutation_probability=0.1,
)
ga_instance.run()
print("end")


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(
    "Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness
    )
)
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Make prediction based on the best solution.
predictions = predict(model=model, solution=solution, data=X_val)

mae = tf.keras.losses.MeanAbsoluteError()
abs_error = mae(Y_val, predictions).numpy()
print("Absolute Error : ", abs_error)


# q = [[tf.constant(0, dtype=tf.float32)] * 6]
# xyz = tf.reshape(RV.getXYZ(q), (1, 3))
# print(xyz)
# xyz_scaled = xyz  # scaler_X.transform(xyz)
# # print(xyz_scaled)
# p_scaled = predict(model=model, solution=solution, data=[xyz_scaled])
# p = p_scaled  # scaler_Y.inverse_transform(p_scaled)
# print(p)

ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# train_loss_history = []
# test_loss_history = []
# try_m = list(range(50, 1501, 50))
# for m in try_m:
#     train_loss, test_loss = train_model(m)
#     train_loss_history.append(train_loss[-1])
#     test_loss_history.append(test_loss[-1])


# plt.plot(try_m, train_loss_history)
# plt.plot(try_m, test_loss_history)
# plt.legend(["Train", "Test"])
# plt.show()

# # Visualize loss history
# fig, (p1, p2) = plt.subplots(2)
# fig.set_size_inches(5, 10)

# p1.set_title("Loss")
# p1.plot(epoch_count, training_loss)
# p1.plot(epoch_count, test_loss)
# p1.legend(["Train", "Test"])
# p1.set_xlabel("Epoch")
# p1.set_ylabel("Loss value")

# # Visualize accuracy history
# p2.set_title("Acuracy")
# p2.plot(epoch_count, training_acc)
# p2.plot(epoch_count, test_acc)
# p2.legend(["Train", "Test"])
# p2.set_xlabel("Epoch")
# p2.set_ylabel("Accuracy value")

# plt.show()


# model = load_model(r'C:\\Users\\Admin\\source\\repos\\STL-Viewer\\RobotOmgtu\\model.h5')
