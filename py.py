import tensorflow as tf

# Check oneDNN custom operations status
import os
onednn_status = os.getenv('TF_ENABLE_ONEDNN_OPTS')
print(f"oneDNN Custom Operations Enabled: {onednn_status}")

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("No GPUs Available")

# Simple operation to ensure TensorFlow is working
a = tf.constant(2)
b = tf.constant(3)
c = a + b
print(f"TensorFlow operation result: {c}")
