import tensorflow as tf

# Assume 'model' is your Keras model (U2NETP implemented in Keras)
# For example, load your model from an .h5 file:
model = tf.keras.models.load_model('u2netp_keras.h5')

# Convert the Keras model to TensorFlow Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file.
with open("u2netp.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite!")
