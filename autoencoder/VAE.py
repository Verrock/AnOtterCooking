#Pour les données
import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 10

encoder_inputs = keras.Input(shape=1026)
x = layers.Dense(1200, activation="relu")(encoder_inputs)
x = layers.Dense(1200, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(1200, activation="tanh")(latent_inputs)
x = layers.Dense(1200, activation="tanh")(x)
x = layers.Dense(1200, activation="tanh")(x)
decoder_outputs = layers.Dense(1026)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            bc = keras.losses.mean_squared_error(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(bc)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def clean_up(x):
    a = x['Quantity'].split('/')
    if len(a) > 1:
        if a[0] == '':
            return np.float32(0)
        res = np.float32(a[0])/np.float32(a[1])
    else:
        if a[0] == '':
            return np.float32(0)
        res = np.float32(a[0])
    return res


def flatten_bis(df):
    lst = []
    for _, d in df.iterrows():
        lst.append(flatten(d))
    return np.array(lst)


def flatten(row):
    x = np.append(row.Units.numpy().flatten(), row.Instructions.numpy().flatten())
    a = np.append(x, [row.Quantity, row.Number])
    return a


def z_norm(x):
    x_mean = x.mean()
    x_std = x.std()
    return (x - x_mean) / x_std


def del_outlier(x):
    for i in range(6):
        x = np.delete(x, 71545, axis=0)
    print(np.max(x[71545,1025]))
    print(x.shape)
    return x


# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
# df = pd.read_csv('final.csv')
# df['Scale'].fillna('', inplace=True)
# df['Ingredient'].fillna('', inplace=True)
# df['Instruction'] = df['Scale'].astype(str) + ' ' + df['Ingredient'].astype(str)
# df = df.drop(['ID', 'Ingredient', 'Scale', 'Ingredient_cpl'], axis=1)
# df = df.dropna()
# df['Instructions'] = df.apply(lambda x: embed(x['Instruction']), axis=1)
# df = df.drop(['Instruction'], axis=1)
# df['Units'] = df.apply(lambda x: embed(x['Unit']), axis=1)
# df = df.drop(['Unit'], axis=1)
# df['Quantity'] = df.apply(clean_up, axis=1)
# df['Number'] = df['Number'].astype(np.float32)
#
# np_arr = flatten_bis(df)
# print(np_arr.shape)
# np.save('tmp.npy', np_arr)

np_arr = np.load('tmp.npy')
np_arr[:,-2] = z_norm(np_arr[:,-2])
np_arr[:,-1] = z_norm(np_arr[:,-1])
np_arr = del_outlier(np_arr)

print(np_arr.shape)

train, test = train_test_split(np_arr, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

vae = VAE(encoder, decoder)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("save_at_Xception_{epoch}"),
]
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.fit(np_arr, epochs=10, batch_size=64)

tf.saved_model.save(vae, 'vae_ingredient')