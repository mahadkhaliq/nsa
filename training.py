from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=128, 
                verbose=0, use_augmentation=True):
    """Train with data augmentation for CIFAR-10"""
    
    # Use augmentation for color images
    if use_augmentation and x_train.shape[-1] == 3:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            validation_data=(x_val, y_val),
            epochs=epochs,
            steps_per_epoch=len(x_train) // batch_size,
            verbose=verbose
        )
    else:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    
    return history

def evaluate_model(model, x_val, y_val, verbose=0):
    """
    Evaluate a Keras model
    
    Args:
        model: Trained Keras model
        x_val, y_val: Validation data
        verbose: Verbosity level
    
    Returns:
        Accuracy (float)
    """
    loss, accuracy = model.evaluate(x_val, y_val, verbose=verbose)
    return accuracy