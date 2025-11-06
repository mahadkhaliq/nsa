def train_model(model, x_train, y_train, x_val, y_val, epochs=5):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=128,
        verbose=0
    )
    return history

def evaluate_model(model, x_val, y_val):
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy
