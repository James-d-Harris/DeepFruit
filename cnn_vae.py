# cnn_vae.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# Image & latent configuration
img_height = 100
img_width = 100
img_channels = 1
latent_dim = 16

# --- Sampling layer ---
class Sampling(layers.Layer):
    """
    z = z_mean + exp(0.5 * z_log_var) * epsilon
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae() -> tuple[Model, Model, Model]:
    """
    Build a CNN-based Variational Autoencoder (VAE) for 100x100x1 images.

    Returns
    -------
    vae : tf.keras.Model
        The full VAE model that maps input -> reconstruction.
    encoder : tf.keras.Model
        Encoder that maps image -> (z_mean, z_log_var, z).
    decoder : tf.keras.Model
        Decoder that maps latent vector z -> reconstructed image.
    """

    # Encoder
    encoder_inputs = layers.Input(
        shape=(img_height, img_width, img_channels), name="encoder_input"
    )

    # 100x100x1 → 50x50x32
    x = layers.Conv2D(
        32, kernel_size=3, strides=2, padding="same", activation="relu", name="enc_conv1"
    )(encoder_inputs)

    # 50x50x32 → 25x25x64
    x = layers.Conv2D(
        64, kernel_size=3, strides=2, padding="same", activation="relu", name="enc_conv2"
    )(x)

    # 25x25x64 → 25x25x128
    x = layers.Conv2D(
        128, kernel_size=3, strides=1, padding="same", activation="relu", name="enc_conv3"
    )(x)

    x = layers.Flatten(name="enc_flatten")(x)
    x = layers.Dense(128, activation="relu", name="enc_dense")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z_sampling")([z_mean, z_log_var])

    encoder = Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name="encoder",
    )

    # Feature map dimensions after encoder:
    # We downsampled twice by factor 2 with "same" padding:
    # 100 -> 50 -> 25
    feature_map_h = img_height // 4
    feature_map_w = img_width // 4
    feature_channels = 128

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_input")

    x = layers.Dense(
        feature_map_h * feature_map_w * feature_channels,
        activation="relu",
        name="dec_dense",
    )(latent_inputs)

    x = layers.Reshape(
        (feature_map_h, feature_map_w, feature_channels),
        name="dec_reshape",
    )(x)

    # 25x25x128 → 25x25x128
    x = layers.Conv2DTranspose(
        128,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        name="dec_deconv1",
    )(x)

    # 25x25x128 → 50x50x64
    x = layers.Conv2DTranspose(
        64,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu",
        name="dec_deconv2",
    )(x)

    # 50x50x64 → 100x100x32
    x = layers.Conv2DTranspose(
        32,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu",
        name="dec_deconv3",
    )(x)

    # 100x100x32 → 100x100x1
    decoder_outputs = layers.Conv2D(
        img_channels,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        name="decoder_output",
    )(x)

    decoder = Model(
        latent_inputs,
        decoder_outputs,
        name="decoder",
    )

    # VAE: connect encoder + decoder
    z_mean_out, z_log_var_out, z_out = encoder(encoder_inputs)
    reconstructed = decoder(z_out)

    vae = Model(
        encoder_inputs,
        reconstructed,
        name="vae",
    )

    # Losses: reconstruction + KL
    recon_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, reconstructed)
    recon_loss = tf.reduce_sum(recon_loss, axis=(1, 2))
    recon_loss = tf.reduce_mean(recon_loss)


    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + z_log_var_out - tf.square(z_mean_out) - tf.exp(z_log_var_out),
            axis=1,
        )
    )

    vae.add_loss(recon_loss + kl_loss)
    vae.add_metric(recon_loss, name="recon_loss", aggregation="mean")
    vae.add_metric(kl_loss, name="kl_loss", aggregation="mean")
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    return vae, encoder, decoder
