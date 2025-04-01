import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dir = 'C:/Users/Anshu/Desktop/waste classification/dataset/train/images'
val_dir = 'C:/Users/Anshu/Desktop/waste classification/dataset/val/images'
test_dir = 'C:/Users/Anshu/Desktop/waste classification/dataset/test/images'


train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),  
    batch_size=32,  
    label_mode='int', 
    shuffle=True 
)
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',
    shuffle=True
)
test_ds = image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='int',
    shuffle=False 
)


normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
data_augmentation = models.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])


num_classes = 30
model = models.Sequential([
    data_augmentation,  
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  
    layers.Dense(num_classes, activation='softmax') 
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)


test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc * 100:.2f}%')
model.save('waste_classification_model.keras')

# model.save('waste_classification_model.h5') in case of h5
