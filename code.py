
def split_features(features_data_frame, seq_len):
   amount_of_features = len(features_data_frame.columns)
   data = features_data_frame.as_matrix()
   sequence_length = seq_len + 1
   result = []
   for index in range(len(data) - sequence_length):
       result.append(data[index: index + sequence_length])

   result = np.array(result)
   row = round(0.8 * result.shape[0])
   train = result[:int(row), :]
   x_train = train[:, :-1]
   y_train = train[:, -1][:, -1]
   x_test = result[int(row):, :-1]
   y_test = result[int(row):, -1][:, -1]
   x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
   x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
   return [x_train, y_train, x_test, y_test]

# ...

window = 5
X_train, y_train, X_test, y_test = split_features(features[::-1], window)



def build_single_lstm(layers):
   model = Sequential()
   model.add(LSTM(50, input_shape=(layers[1], layers[0]), return_sequences=False))
   model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
   model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
   return model

# ...

model = builder([len(features.columns), window, 1])

model.fit(
   X_train,
   y_train,
   batch_size=settings.batch_size,
   epochs=settings.epochs,
   validation_split=settings.validation_split,
   callbacks=[json_logging_callback],
   verbose=0)


from sklearn.metrics import mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
   y_true, y_pred = np.array(y_true), np.array(y_pred)
   return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

predicted = model.predict(X_test)
actual = y_test

predicted = (predicted * scaling_parameter) + minima
actual = (actual * scaling_parameter) + minima

mape = sqrt(mean_absolute_percentage_error(predicted, actual))
mse = mean_absolute_error(actual, predicted)



pyplot.plot(actual[:24], label='Actual', color="blue")
pyplot.plot(predicted[:24], label='Model1', color="red")
pyplot.plot(predicted[:24], label='Model2', color="green")
pyplot.ylabel("Load(MW)")
pyplot.title("Actual Vs Predicted Results")
pyplot.legend()
pyplot.show()

