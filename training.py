def train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=128, verbose=0):
    """
    Train a Keras model
    
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        History object
    """
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