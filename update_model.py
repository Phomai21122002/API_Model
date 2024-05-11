import tensorflow as tf
import time
from tensorflow.keras.models import load_model, Model
import json
import h5py
from glob import glob
from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping


def load_model_old():
  path_model = './my_model_NetV2.h5'

  # Load the model
  with h5py.File(path_model, 'r+') as f:
    if 'model_config' in f.attrs:
      config = f.attrs['model_config']
      model_config = json.loads(config)
      for layer in model_config['config']['layers']:
        if layer['class_name'] == 'DepthwiseConv2D':
          # Remove the 'groups' parameter if it exists
          if 'groups' in layer['config']:
            del layer['config']['groups']
      # Update the model configuration in the file
      f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

  model = load_model(path_model)

  # Đường dẫn đến file JSON chứa thông tin ánh xạ nhãn
  json_file_path = './labels.json'

  # Đọc nội dung từ file JSON
  with open(json_file_path, 'r') as json_file:
      labels_dict = json.load(json_file)

  return model, labels_dict

def load_data():
  training_dir = './Dataspl/Train/'
  validation_dir = './Dataspl/Validation/'
  test_dir = './Dataspl/Test/'

  image_files = glob(training_dir + '/*/*.jp*g')
  valid_image_files = glob(validation_dir + '/*/*.jp*g')

  # getting the number of classes i.e. type of fruits
  folders = glob(training_dir + '/*')
  num_classes = len(folders)
  print ('Total Classes = ' + str(num_classes))

  return training_dir, validation_dir, test_dir, num_classes

def create_update_model(num_classes, model):
  IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results.

  # Remove the last layer from the model
  model_layers = model.layers[:-1]

  # Add a new output layer
  output = Dense(num_classes, activation='softmax', name='new_output')(model_layers[-1].output)
  model = Model(inputs=model.input, outputs=output)

  print("[INFO] compiling model ...")
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Freeze the layers in the original model
  for layer in model.layers[:-1]:
      layer.trainable = False

  for layer in model.layers:
    print('Layer: {} ; Trainable: {}'.format(layer, layer.name))

  return IMAGE_SIZE, model


def prepare_data(IMAGE_SIZE, training_dir, validation_dir, test_dir):
  training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

  validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)
  test_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

  training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
  validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
  test_generator = test_datagen.flow_from_directory(test_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')

  return training_generator, validation_generator, test_generator

def update_model_train(model, training_generator, validation_generator):
  early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
  start_time = time.time()

  history = model.fit(training_generator,
                    epochs = 30,  # change this for better results
                    validation_data = validation_generator,
                      callbacks = [early_stopping]
                    )

  end_time = time.time()
  time_train_seconds = end_time - start_time
  time_train_minutes = time_train_seconds / 60

  model.save('./my_model_update.h5')

  return history, time_train_minutes

def file_update_model():
  model, labels_dict = load_model_old()
  training_dir, validation_dir, test_dir, num_classes = load_data()
  IMAGE_SIZE, model = create_update_model(num_classes, model)
  training_generator, validation_generator, test_generator = prepare_data(IMAGE_SIZE, training_dir, validation_dir, test_dir)
  history, time_train_minutes = update_model_train(model, training_generator, validation_generator)

  print(time_train_minutes)