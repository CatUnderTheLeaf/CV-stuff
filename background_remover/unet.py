
import keras
from tensorflow_examples.models.pix2pix import pix2pix

def createModel(IMAGE_WIDTH=256, IMAGE_HEIGHT=256, num_classes = 2):
  
  base_model = keras.applications.MobileNetV2(
      include_top=False,
      weights="imagenet",
      input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
      classes=num_classes, # only two classes
    #   name="base_encoder_model"
  )
  # Decoder Layers
  base_model_layer_names = [
          'block_1_expand_relu',
          'block_3_expand_relu',
          'block_6_expand_relu',
          'block_13_expand_relu',
          'block_16_project',
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in base_model_layer_names]

  encoder_model = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  encoder_model.trainable = False

  input_layer = keras.layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
  enc_layers = encoder_model(input_layer)



  # Decoder
  x = pix2pix.upsample(512, 3, apply_dropout=True)(enc_layers[-1])
  x = keras.layers.Concatenate()([x, enc_layers[-2]])
  x = keras.layers.Dropout(0.5)(x)

  x = pix2pix.upsample(256, 3, apply_dropout=True)(x)
  x = keras.layers.Concatenate()([x, enc_layers[-3]])
  x = keras.layers.Dropout(0.5)(x)

  x = pix2pix.upsample(128, 3)(x)
  x = keras.layers.Concatenate()([x, enc_layers[-4]])
  x = keras.layers.Dropout(0.5)(x)

  x = pix2pix.upsample(64, 3)(x)
  x = keras.layers.Concatenate()([x, enc_layers[-5]])

  x = keras.layers.Dropout(0.5)(x)


  output_layer = keras.layers.Conv2DTranspose(num_classes, (3,3), strides=2, padding="same", activation="softmax")(x)
  # output_layer = keras.layers.Conv2DTranspose(num_classes, (3,3), strides=2, padding="same")(x)
  # output_layer = tf.keras.layers.Conv2DTranspose(1, (3,3), strides=2, padding="same", activation='sigmoid')(x)

  return keras.Model(inputs=input_layer, outputs=output_layer)

new_model = createModel()
new_model.summary()