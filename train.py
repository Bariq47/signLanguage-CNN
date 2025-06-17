import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#
# Paths
DATASET_DIR = "/kaggle/input/a-z-asl-dataset-old"
MODEL_PATH = "/kaggle/working/a-j/sign_language_model_final.h5"
BEST_MODEL_PATH = "/kaggle/working/a-j/best_sign_language_model20.h5"
EPOCH_MODEL_DIR = "/kaggle/working/Epochs1/"
os.makedirs(EPOCH_MODEL_DIR, exist_ok=True)

# Augmentasi dan normalisasi data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generator training dan validasi
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(26, activation='softmax')  # 29 label
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(EPOCH_MODEL_DIR, "model_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.h5"),
        save_best_only=False,
        verbose=1
    )
]

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# Save final model
model.save(MODEL_PATH)
print(f"Model terakhir disimpan ke: {MODEL_PATH}")
print(f"Model terbaik (val_accuracy tertinggi) disimpan ke: {BEST_MODEL_PATH}")
