# coding: utf-8
def process_image(image_path, img_size =IMG_SIZE):
  """
  Turns the image into a Tensor.

  :param str image_path: image filepath
  """
  # Read in an image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 color channels
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the color channel values from 0-255 to 0-1 values
  image= tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired value (224,224)
  image = tf.image.resize(image,size=[IMG_SIZE, IMG_SIZE ])        

  return image      
