from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.cloud import bigquery

print(tf.__version__)


def get_dataset(client, dataset_id):
    dataset = client.get_dataset(dataset_id)

    full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
    friendly_name = dataset.friendly_name
    print(
        "Got dataset '{}' with friendly_name '{}'.".format(
            full_dataset_id, friendly_name
        )
    )

    # View dataset properties
    print("Description: {}".format(dataset.description))
    print("Labels:")
    labels = dataset.labels
    if labels:
        for label, value in labels.items():
            print("\t{}: {}".format(label, value))
    else:
        print("\tDataset has no labels defined.")

    # View tables in dataset
    print("Tables:")
    tables = list(client.list_tables(dataset))  # API request(s)
    if tables:
        for table in tables:
            print("\t{}".format(table.table_id))
    else:
        print("\tThis dataset does not contain any tables.")

def get_rows_as_dataframe(client, dataset_id, table_id):
    QUERY = (
    f'SELECT * FROM `{dataset_id}.{table_id}` '
    'WHERE waterTemperature IS NOT NULL AND ambientTemperature != 0 and ambientHumidity != 0')
    print("{}".format(QUERY))
    df = client.query(QUERY).to_dataframe()  # API request
    return df

dataset_id = 'selfhydro-197504.selfhydro'
table_id = 'selfhydro_state'
client = bigquery.Client()
get_dataset(client, dataset_id)
df = get_rows_as_dataframe(client, dataset_id, table_id)
sorted_df = df.sort_values(by='time')

def clean_data(dataframe):
    dataframe.pop('waterLevel')
    dataframe.pop('deviceId')
    dataframe = dataframe.dropna()
    return dataframe

cleaned_dataframe = clean_data(sorted_df)
print(cleaned_dataframe)
stats = cleaned_dataframe.describe()
print(stats)
cleaned_dataframe.pop('time')


sns.pairplot(cleaned_dataframe[["ambientTemperature", "ambientHumidity", "waterTemperature", "waterElectricalConductivity"]], diag_kind="kde", kind="reg", palette="husl")

train_dataset = cleaned_dataframe.sample(frac=0.8, random_state=0)
test_dataset = cleaned_dataframe.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("waterTemperature")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('waterTemperature')
test_labels = test_dataset.pop('waterTemperature')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation=tf.nn.relu),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()
model.summary()


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
       label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
       label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
       label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
       label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()


plot_history(history)
test_predictions = model.predict(normed_test_data).flatten()
print("\n")
print(test_dataset)
print(test_labels)
print(test_predictions)

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Water Temperature]')
plt.ylabel('Predictions [Water Temperature]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Water Temperature]")
_ = plt.ylabel("Count")
plt.show()
