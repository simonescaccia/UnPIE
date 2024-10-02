import subprocess as sp
import tensorflow as tf

num_dashes = 100

def print_separator(message=None, space=True, top_new_line=True, bottom_new_line=True):
    if space and top_new_line:
        print()
    print('-'*num_dashes)
    if message:
        print(message)
    if space and bottom_new_line:
        print()
    return

def print_model_size(model, model_name):
    # Calculate the number of parameters
    num_params = model.count_params()
    # Assuming each parameter is 4 bytes (32 bits)
    memory_bytes = num_params * 4
    # Convert to megabytes
    memory_mb = memory_bytes / (1024 ** 2)
    print('')
    print(model_name, f" memory: {memory_mb:.2f} MB\n")

def print_memory_info(info):
    with tf.Session() as sess:
        bytesInUse = tf.contrib.memory_stats.BytesInUse()
        bytesLimit = tf.contrib.memory_stats.BytesLimit()
        maxBytesInUse = tf.contrib.memory_stats.MaxBytesInUse()
        byteInUse_mb = sess.run(bytesInUse) / (1024 ** 2)
        byteLimit_mb = sess.run(bytesLimit) / (1024 ** 2)
        maxByteInUse_mb = sess.run(maxBytesInUse) / (1024 ** 2)
        print(f"[ {info} ]")
        print(f"Bytes in use: {byteInUse_mb:.2f} MB")
        print(f"Bytes limit: {byteLimit_mb:.2f} MB")
        print(f"Max bytes in use: {maxByteInUse_mb:.2f} MB")
    print('')
