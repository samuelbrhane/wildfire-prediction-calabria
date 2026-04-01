# LSTM model training used by both zone and regional tuning
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '3_utils'))

from constants import EARLY_STOPPING_PATIENCE, MAX_EPOCHS, LOSS_FUNCTION


def build_and_train_model(X_train, y_train, X_val, y_val, params):
    # Build and train a dynamic LSTM model for zone and regional models
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_layers = params['num_layers']

    model = Sequential()
    for i in range(num_layers):
        return_seq = (i != num_layers - 1)
        if i == 0:
            model.add(LSTM(params['lstm_units'],
                          activation=params['activation_function'],
                          return_sequences=return_seq,
                          input_shape=input_shape))
        else:
            model.add(LSTM(params['lstm_units'],
                          activation=params['activation_function'],
                          return_sequences=return_seq))
        model.add(Dropout(params['dropout_rate']))

    model.add(Dense(1))

    if params['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=params['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {params['optimizer']}")

    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history