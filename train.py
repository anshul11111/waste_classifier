import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os


base_dir = 'Dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')


os.makedirs(os.path.join(validation_dir, 'B'), exist_ok=True)
os.makedirs(os.path.join(validation_dir, 'N'), exist_ok=True)


img_height, img_width = 224, 224
batch_size = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  
)


validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  


model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


epochs = 10

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)   


model.save('waste_classification_model.h5')

print("Model trained and saved successfully!")



for layer in base_model.layers[-20:]:
    layer.trainable = True


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history_fine = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),  
    epochs=5,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // batch_size)
)


model.save('waste_classification_model_finetuned.h5')

print("dOne")