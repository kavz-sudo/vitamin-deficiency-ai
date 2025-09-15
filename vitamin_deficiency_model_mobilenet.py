from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paths
dataset_path = 'dataset'  # Change if your dataset is in a different folder
image_size = (224, 224)
batch_size = 32
epochs = 10

# 1. Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 2. Load Pretrained MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 3. Freeze Base Layers (so we train only the top layers first)
for layer in base_model.layers:
    layer.trainable = False

# 4. Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Avoid overfitting
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# 7. Save Model
model.save("vitamin_deficiency_model_mobilenet.h5")
print("✅ Model saved as vitamin_deficiency_model_mobilenet.h5")
