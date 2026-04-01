# Transformer model training used by both zone and regional tuning
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Add, Concatenate,
    Embedding, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '3_utils'))

from constants import (
    MONTHS_IN_DATA, DAYS_IN_WEEK,
    EARLY_STOPPING_PATIENCE, MAX_EPOCHS, LOSS_FUNCTION
)


def _transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Single transformer encoder block with multi-head attention and feed-forward layers
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_and_train_model(X_train, y_train, X_val, y_val, params):
    # Build and train a Transformer model for zone and regional models
    X_train_num, X_train_month, X_train_dow = X_train
    X_val_num, X_val_month, X_val_dow = X_val

    input_shapes = {
        'numeric': (X_train_num.shape[1], X_train_num.shape[2]),
        'month': (X_train_month.shape[1],),
        'dow': (X_train_dow.shape[1],)
    }

    numeric_input = Input(shape=input_shapes['numeric'], name='numeric_input')
    month_input = Input(shape=input_shapes['month'], dtype='int32', name='month_input')
    dow_input = Input(shape=input_shapes['dow'], dtype='int32', name='dow_input')

    month_embed = Embedding(input_dim=MONTHS_IN_DATA, output_dim=params['month_embed_dim'])(month_input)
    dow_embed = Embedding(input_dim=DAYS_IN_WEEK, output_dim=params['dow_embed_dim'])(dow_input)

    x = Concatenate()([numeric_input, month_embed, dow_embed])
    x = LayerNormalization(epsilon=1e-6)(x)

    for _ in range(params['num_layers']):
        x = _transformer_encoder(
            x,
            head_size=params['d_model'],
            num_heads=params['num_heads'],
            ff_dim=params['ff_dim'],
            dropout=params['dropout_rate']
        )

    x = x[:, -1, :]
    x = Dropout(params['dropout_rate'])(x)
    output = Dense(1)(x)

    model = Model(inputs=[numeric_input, month_input, dow_input], outputs=output)

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
        [X_train_num, X_train_month, X_train_dow], y_train,
        validation_data=([X_val_num, X_val_month, X_val_dow], y_val),
        epochs=MAX_EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history