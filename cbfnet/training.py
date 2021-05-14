import tensorflow as tf
    
def train_basic_model(model, datasets, epochs, params, 
                      min_delta=0.0002, lr_decay=False,
                      early_stopping=False):
    ''' 
    Train model.

    model: Compiled TF model
    datasets: Train and validation TF datasets 
    epochs: int giving number of epochs
    params: dict with various params
    '''

    # A simple LR scheduler. Switch to lr=0.0001 at epoch 100.
    def scheduler(epoch):
        if epoch < 100:
            return params['lr']
        else:
            return 0.0001

    # Create the LR scheduler callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Create an early stopping callback
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=min_delta,
                                                     patience=5)

    callbacks = []

    if lr_decay:
        callbacks.append(lr_callback)
    if early_stopping:
        callbacks.append(stop_callback)

    # Train the model
    history = model.fit(datasets['train'], validation_data=datasets['val'],
                        epochs=epochs, callbacks=callbacks)
                        
    return history             