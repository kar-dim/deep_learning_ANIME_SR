# Dimitris Karatzas aivc25007

import keras
from keras import layers, models, ops

@keras.saving.register_keras_serializable()
def pixel_shuffle_output_shape(input_shape, upscale_factor=2):
    """Required by the Lambda layer in build_edsr: tells Keras the output shape after depth_to_space
    (H, W, C * r^2) -> (H*r, W*r, C), where r = upscale_factor"""
    in_channels = input_shape[-1]
    out_channels = in_channels // (upscale_factor**2)
    if input_shape[1] is None or input_shape[2] is None:
        return (input_shape[0], None, None, out_channels) 
    return (input_shape[0], input_shape[1] * upscale_factor, input_shape[2] * upscale_factor, out_channels)

def res_block(x_in, filters, scaling=0.1):
    """Deep ResBlock with scaling for stability"""
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x_in)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    if scaling:
        x = x * scaling
    return layers.Add()([x_in, x])

def build_srcnn():
    """Baseline SRCNN"""
    inputs = layers.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, (9, 9), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    outputs = layers.Conv2D(1, (5, 5), activation='linear', padding='same', kernel_initializer='he_normal')(x)
    return models.Model(inputs, outputs, name="SRCNN_Baseline")

def build_edsr(num_blocks=16, num_filters=64, upscale_factor=2):
    """EDSR model. Input: float32 [0, 1]. Default values is baseline (16 blocks, 64 filters), Full-scale: (32, 256)"""
    inputs = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(scale=1.0, offset=-0.5)(inputs)
    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x_head = x
    for _ in range(num_blocks):
        x = res_block(x, num_filters, scaling=0.1)
    x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.Add()([x_head, x])
    x = layers.Conv2D(3 * (upscale_factor**2), (3, 3), padding='same',
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01))(x)
    outputs = layers.Lambda(lambda t: ops.depth_to_space(t, upscale_factor),
                            output_shape=lambda s: pixel_shuffle_output_shape(s, upscale_factor))(x)
    return models.Model(inputs, outputs + 0.5, name=f"EDSR_b{num_blocks}_f{num_filters}")
