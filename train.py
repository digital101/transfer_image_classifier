##
#
# This is based on : https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning_with_hub.ipynb#scrollTo=pcFeNcrehEue
# Latest:   https://www.tensorflow.org/hub/tutorials/image_feature_vector
# and https://www.tensorflow.org/lite/performance/post_training_integer_quant
#
##
#   need to install pillow   pip intsall pillow
#   need to install tensorflow_hub   pip instal tensorflow_hub
#
#



import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import sys

import pathlib

###################
#CONFIGURATION!!!
###################
plot_images = 1
do_model_quant = 1
do_model_train = 1
###################

#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np

# Data for plotting
#t = np.arange(0.0, 2.0, 0.01)
#s = 1 + np.sin(2 * np.pi * t)

#fig, ax = plt.subplots()
#ax.plot(t, s)

#ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#       title='About as simple as it gets, folks')
#ax.grid()

#fig.savefig("test.png")
#plt.show()

#download apretrained model from TensorFlow Hub
classifier_model ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4" #@param {type:"string"}




IMAGE_SHAPE = (224, 224)

# create a layered model from the saved model from hub using the KerasLayer function, see https://www.tensorflow.org/guide/keras/sequential_model 
# and https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

classifier.summary()
#input("Press a key to continue")


grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper

#convert to float numpy array and scale to 0 to 1
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

#Run the model on the numpy image array
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape

#The result is an array of likelhoods, where the index is the likelyhood for that image. You just need to search for the highest likelyhood
print(result.shape)

#Find the target prediction (the largest value)
predicted_class = np.argmax(result[0], axis=-1)
predicted_class

#download the association between entry index and label, BTW the model output a 1001 vaues.
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
print("labels_path", labels_path)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print(imagenet_labels[predicted_class])


if plot_images == 1:
  plt.imshow(grace_hopper)
  plt.axis('off')
  predicted_class_name = imagenet_labels[predicted_class]
  _ = plt.title("Prediction: " + predicted_class_name.title())
  plt.show()

#input("Press a key to continue")

# ===========================================
#
#      Now retrain to capture flow using transfer learning 
#
# ===========================================

#Load up the flower images
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

print("data_root", data_root)
   
batch_size = 32   

# == Original heights ==
img_height = 224
img_width = 224

# 
# Create a tf.data.dataset  object called train_ds from the directory so  that we can use it in the future training and inferencing
# 
# Generate 1 batch of data , each batch is 32 images, so as  there are 5 directories, 20% is one 5th or the 1st batch.   So thats   5x32 images 
# 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
print("train_ds", train_ds)

class_names = np.array(train_ds.class_names)
print(class_names)
input("Press a key to continue, class_names")

#Create a normalising layer
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

#Simply uses the normalisation layer above to transform the image data to normalise it.
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))



#AUTOTUNE = tf.data.experimental.AUTOTUNE
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)


#This will always iterate once as we have a break, its just to get the iamge_batch and labels batch out of train_ds
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
  
input("Press a key to continue, labels_batch.shape")

  
#
#
# Show that even with the model untrained for flowers it will do its best to detect the nearest match in the 1001 things 
# it has been trained on
#
#
 
result_batch = classifier.predict(train_ds)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

if plot_images == 1:
  plt.subplots_adjust(hspace=0.5)
  for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
  
  _ = plt.suptitle("ImageNet predictions")
  plt.show()

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" #@param {type:"string"}

feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
    
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

num_classes = len(class_names)


model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

predictions = model(image_batch)

predictions.shape

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])
  
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

#
#Train the model  with model.fit with the training data
#
if do_model_train == 1:
  history = model.fit(train_ds, epochs=40,
                    callbacks=[batch_stats_callback])
                    

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)

if do_model_train == 1:
  predicted_batch = model.predict(image_batch)
  predicted_id = np.argmax(predicted_batch, axis=-1)
  predicted_label_batch = class_names[predicted_id]

  if plot_images == 1:
    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
      plt.subplot(6,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(predicted_label_batch[n].title())
      plt.axis('off')
    _= plt.suptitle("Model predictions")
    plt.show()

  t = time.time()

  #Export the standard TF saved model
  export_path = "/tmp/saved_models/{}".format(int(t))
  model.save(export_path)

  export_path


if do_model_train == 1:
  #Run the model against the image batch
  result_batch = model.predict(image_batch)

  #create the 32 bit float lite version of the model (ie into a flat buffer)
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

#if do_model_train == 0:
#  #Reload the standard TF saved model..
#  model = tf.keras.models.load_model(previous_export_path)
#  #Run the saved model agains the image batch
#  reloaded_result_batch = model.predict(image_batch)
#  converter = tf.lite.TFLiteConverter.from_keras_model(model)
#  tflite_model = converter.convert()

#  abs(reloaded_result_batch - result_batch).max()


#Convert to tf lite and quantise


def representative_data_gen():
  i=0
  for image_batch, labels_batch in train_ds:
#    print(image_batch.shape)
#    print(labels_batch.shape)
    print("Rep data: ",i)
    i=i+1
 
    for input_value in tf.data.Dataset.from_tensor_slices(image_batch).batch(1).take(100):
      yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_models_dir = pathlib.Path("/tmp/transfer_flowers_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file =               tflite_models_dir/"transfer_flowers_model.tflite"
tflite_model_quant_file =         tflite_models_dir/"transfer_flowers_model_quant.tflite"

if do_model_quant == 1:
  tflite_model_quant = converter.convert()

  # Save the unquantized/float model:
  tflite_model_file.write_bytes(tflite_model)
  # Save the quantized model:
  tflite_model_quant_file.write_bytes(tflite_model_quant)
  
  

#                                       ***********************
# Helper function to run inference on a **** TFLite model  ****
#                                       ***********************
def run_tflite_model(tflite_file, image_indices):
  global image_batch
  global labels_batch

  print("image_indices",image_indices)

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details  = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(image_indices),), dtype=int)
  #Extract the array index and the index in the images


  for i, test_image_index in enumerate(image_indices):
    print("running tflite model iteration ", i)
    print("test_image_index: ",test_image_index)
    test_image = image_batch [test_image_index]
    test_label = labels_batch[test_image_index]

    print("test_label: ", test_label)

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()

    #Return an array of 5 items, the max value is likely the predicted label 
    output = interpreter.get_tensor(output_details["index"])[0]
    print("output: ", output)

    predictions[i] = output.argmax()

  return predictions


import matplotlib.pylab as plt

# Change this to test a different image range from the first image
test_images_indexes = [3,4,5]

num_test_image_index = len(test_images_indexes)
print("num_test_image_index: ",num_test_image_index)

for n in range(num_test_image_index):
  plt.subplot(1,num_test_image_index,n+1)
  plt.imshow(   image_batch [test_images_indexes[n]])
  plt.title(str(labels_batch[test_images_indexes[n]]))
  plt.axis('off')
_= plt.suptitle("Test images")
plt.show()

## Helper function to test the models on one image
def test_model(tflite_file, image_indexes, model_type):
  global image_batch
  global labels_batch

  print("image_indexes",image_indexes)

  predictions = run_tflite_model(tflite_file, image_indexes)
  
  print("predictions", predictions)
#  for i, in range(len(predictions)):
#    print("prediction ", i, )
#    print("label: ", image_label[predictions[i]])
#
#    plt.imshow(image_batch[test_image_index])
#    template = model_type + " Model \n True:{true}, Predicted:{predict}"
#    _ = plt.title(template.format(true= str(image_batch[test_image_index]), predict=str(predictions[0])))
#    plt.grid(False)
#    plt.show()

  plt.figure(figsize=(10,9))
  plt.subplots_adjust(hspace=0.5)
  numpredictions = len(predictions)
  print("numpredictions: ",numpredictions)
  for n in range(numpredictions):
    plt.subplot(1,numpredictions,n+1)
    plt.imshow(image_batch [predictions[n]])
    plt.title (str(labels_batch[predictions[n]]))
    plt.axis('off')
  _= plt.suptitle("Model predictions")
  plt.show()

 
#
# Test the two models:
# 
test_model(tflite_model_file, test_images_indexes, model_type="Float")
 
test_model(tflite_model_quant_file, test_images_indexes, model_type="Quantized")


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global image_batch

  test_image_indices = range(image_batch.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)
  
  print('test_image_indices: ', test_image_indices)

  accuracy = (np.sum(image_batch== predictions) * 100) / len(image_batch)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(image_batch)))
      
#evaluate_model(tflite_model_file, model_type="Float")

#evaluate_model(tflite_model_quant_file, model_type="Quantized")



if __name__=="__main__":
  print(f"Args count: {len(sys.argv)}")
  for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
    
