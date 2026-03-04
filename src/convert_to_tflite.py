import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('trained_models/vww_96.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Force FP16 weight compression
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the model
with open('trained_models/vww_96_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
