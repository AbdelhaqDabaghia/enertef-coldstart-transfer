import tensorflow as tf, time
tf.debugging.set_log_device_placement(False)
gpus = tf.config.list_physical_devices('GPU')
print("GPU devices:", gpus)
# force an op and show WHERE it runs
with tf.device('/GPU:0'):
    a = tf.random.normal([4096, 4096]); b = tf.random.normal([4096, 4096])
    t0 = time.time()
    for _ in range(20):
        c = tf.matmul(a, b)
    _ = c.numpy()
    print("matmul device:", c.device, "| 20x4096^2 matmul:", round(time.time()-t0, 3), "s")
# a real forward+backward of OUR model on GPU
from coldstart_transfer.model import build_ev_model
import numpy as np
m = build_ev_model()
xs = tf.constant(np.random.randn(64,672,21).astype('float32'))
xn = tf.constant(np.random.randn(64,20).astype('float32'))
with tf.GradientTape() as tp:
    y = m([xs,xn]); loss = tf.reduce_mean(y**2)
g = tp.gradient(loss, m.trainable_variables)
print("model output device:", y.device)
print("gradient[0] device :", g[0].device)
