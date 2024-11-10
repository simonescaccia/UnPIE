import tensorflow as tf

from utils.pie_utils import update_progress

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

@tf.py_function(Tout=tf.string)
def serialize_pie(x, a, i, y):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
      'a': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(a).numpy()])),
      'i': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(i).numpy()])),
      'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
  }

  # Create a Features message using tf.train.Example.
  pie_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return pie_proto.SerializeToString()

def write_pie_tfrecord(filename, x, a, i, y):
    # Write the `tf.train.Example` observations to the file.
    print('Writing {}...'.format(filename))
    with tf.io.TFRecordWriter(filename) as writer:
        for j in range(x.shape[0]):
            update_progress(j / x.shape[0])
            pie = serialize_pie(x[j], a[j], i[j], y[j])
            writer.write(pie.numpy())

def read_pie_tfrecord(filename):
    # Create a description of the features.
    feature_description = {
        'x': tf.io.parse_tensor([], out_type=tf.float32),
        'a': tf.io.parse_tensor([], out_type=tf.float32),
        'i': tf.io.parse_tensor([], out_type=tf.int32),
        'y': tf.io.parse_tensor([], out_type=tf.int32),
    }

    def _parse_pie_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    raw_dataset = tf.data.TFRecordDataset([filename])
    parsed_dataset = raw_dataset.map(_parse_pie_function)

    return parsed_dataset

