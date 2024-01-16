"Reader for MNIST data."
import struct
import gzip
import numpy as np

def get_data(image_fn, label_fn, force_1d=False):
  print(f"reading data {image_fn} {label_fn}")
  # The image files have the following format:
  # This is a gzipped big-endian file with the following format:
  #   [offset] [type]          [value]          [description]
  #   0000     32 bit integer  0x00000803(2051) magic number
  #   0004     32 bit integer  60000            number of images
  #   0008     32 bit integer  28               number of rows
  #   0012     32 bit integer  28               number of columns
  #   0016     unsigned byte   ??               pixel
  #   0017     unsigned byte   ??               pixel
  #   ...

  images_file = gzip.open(image_fn, "rb")
  _, num_images, num_rows, num_cols = struct.unpack(">IIII",
                                                    images_file.read(16))

  # The label files have the following format:
  #   [offset] [type]          [value]          [description]
  #   0000     32 bit integer  0x00000801(2049) magic number (MSB first)
  #   0004     32 bit integer  60000            number of items
  #   0008     unsigned byte   ??               label
  #   0009     unsigned byte   ??               label
  #   ...

  labels_file = gzip.open(label_fn)
  _, num_labels = struct.unpack(">II", labels_file.read(8))

  assert num_labels == num_images

  images = np.ndarray((num_images, num_rows, num_cols), dtype=np.float32)
  labels = np.zeros((num_images, 10), dtype=np.ubyte)

  for i in range(0, num_images):
    label = labels_file.read(1)[0]
    labels[i][label] = 1
    for r in range(0, num_rows):
      images[i][r] = np.frombuffer(images_file.read(num_cols), dtype=np.ubyte)
  images /= 255.0

  if force_1d:
    images = np.reshape(images, ( images.shape[0], -1)) # (Images, Rows*Cols)
  print(f"{images.shape[0]} images loaded.")
  return images, labels
