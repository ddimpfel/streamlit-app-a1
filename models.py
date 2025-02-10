import keras

def build_model(model_type, input_shape, num_classes) -> keras.Sequential:
    """Build and compile one of three preset CNN models."""
    if model_type == "Basic CNN Model":
        # Two convolutional neural network
        model = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == "Block CNN Model":
        # Convolutional Block CNN Model
        model = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),
            # Block 1
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(),
            # Block 2
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(),
            # Block 3
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # Model using Batch Normalization
        model = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),

            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
    )
    return model
