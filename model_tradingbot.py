import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Conv1D, TimeDistributed, LayerNormalization, Bidirectional,
    MultiHeadAttention, Flatten, Input, Concatenate
)
from tensorflow.keras import Model
from ta import add_all_ta_features
from sklearn.metrics import r2_score, mean_absolute_error
import krakenex

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv1D(self.filters, 1, padding='same', activation='relu')
        self.conv3 = Conv1D(self.filters, 3, padding='same', activation='relu')
        self.conv5 = Conv1D(self.filters, 5, padding='same', activation='relu')
        self.concat = Concatenate()

    def call(self, inputs):
        return self.concat([self.conv1(inputs), self.conv3(inputs), self.conv5(inputs)])

class InceptionTime(Model):
    def __init__(self, input_shape, output_units):
        super().__init__()
        self.inception1 = InceptionModule(32)
        self.inception2 = InceptionModule(32)
        self.inception3 = InceptionModule(32)
        self.inception4 = InceptionModule(32)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = Dense(output_units)

    def call(self, inputs):
        x = self.inception1(inputs)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.pool(x)
        return self.dense(x)

def build_inception_time_model(input_shape):
    model = InceptionTime(input_shape, 1)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class EnsembleModel(tf.keras.Model):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.combine = Dense(1)

    def call(self, inputs):
        x1, x2 = self.model1(inputs), self.model2(inputs)
        return self.combine(Concatenate()([x1, x2]))

def build_ensemble_model(model1, model2):
    model = EnsembleModel(model1, model2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Conv1D, TimeDistributed, LayerNormalization, Bidirectional,
    MultiHeadAttention, Flatten, Input, Concatenate
)
from tensorflow.keras import Model
from ta import add_all_ta_features
from sklearn.metrics import r2_score, mean_absolute_error
import krakenex

class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = Conv1D(self.filters, 1, padding='same', activation='relu')
        self.conv3 = Conv1D(self.filters, 3, padding='same', activation='relu')
        self.conv5 = Conv1D(self.filters, 5, padding='same', activation='relu')
        self.concat = Concatenate()

    def call(self, inputs):
        return self.concat([self.conv1(inputs), self.conv3(inputs), self.conv5(inputs)])

class InceptionTime(Model):
    def __init__(self, input_shape, output_units):
        super().__init__()
        self.inception1 = InceptionModule(32)
        self.inception2 = InceptionModule(32)
        self.inception3 = InceptionModule(32)
        self.inception4 = InceptionModule(32)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = Dense(output_units)

    def call(self, inputs):
        x = self.inception1(inputs)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.pool(x)
        return self.dense(x)

def build_inception_time_model(input_shape):
    model = InceptionTime(input_shape, 1)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class EnsembleModel(tf.keras.Model):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.combine = Dense(1)

    def call(self, inputs):
        x1, x2 = self.model1(inputs), self.model2(inputs)
        return self.combine(Concatenate()([x1, x2]))

def build_ensemble_model(model1, model2):
    model = EnsembleModel(model1, model2)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

class CryptoOHLC:
    def __init__(self, pair='XXBTZEUR', interval=240, lookback=60, split_ratio=0.8, target_col='close'):
        self.pair = pair
        self.interval = interval
        self.lookback = lookback
        self.split_ratio = split_ratio
        self.target_col = target_col

        api = krakenex.API()
        ohlc_data = api.query_public('OHLC', {'pair': pair, 'interval': interval, 'since': 1577836800})
        ohlc_list = ohlc_data['result'][pair]
        ohlc_df = pd.DataFrame(ohlc_list, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], unit='s')
        ohlc_df.set_index('time', inplace=True)
        self.data = ohlc_df
        self.predicted = None
        self.history = None
    
    def add_sma(self, window):
        self.data['sma'] = self.data['close'].rolling(window=window).mean()
    def add_std(self, window):
        self.data['std'] = self.data['close'].rolling(window=window).std()
    def add_bollinger_bands(self, window):
        self.add_sma(window)
        self.add_std(window)
        self.data['bollinger_upper'] = self.data['sma'] + 2 * self.data['std']
        self.data['bollinger_lower'] = self.data['sma'] - 2 * self.data['std']

    def add_momentum(self, window):
        self.data['momentum'] = self.data['close'] - self.data['close'].shift(window)
        
    def add_rsi(self, window):
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def add_roc(self, window):
        self.data['roc'] = ((self.data['close'] - self.data['close'].shift(window)) / self.data['close'].shift(window)) * 100



    def add_features(self):
        self.data = self.data.apply(pd.to_numeric, errors='coerce')

        self.data['oc_diff'] = self.data['open'].subtract(self.data['close'])
        self.data['hl_diff'] = self.data['high'].subtract(self.data['low'])
        self.data['volume_pct_change'] = self.data['volume'].pct_change()
        self.data['vwap_pct_change'] = self.data['vwap'].pct_change()
        self.data = add_all_ta_features(self.data, open="open", high="high", low="low", close="close", volume="volume")
        self.add_sma(window=10)
        self.add_std(window=10)
        self.add_bollinger_bands(window=10)
        self.add_momentum(window=10)
        self.add_rsi(window=14)
        self.add_roc(window=10)
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna(inplace=True)


    def preprocess_data(self):
        self.add_features()
        x_data = self.data.drop(self.target_col, axis=1)
        y_data = self.data[self.target_col]

        data_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        x_scaled = data_scaler.fit_transform(x_data)
        y_scaled = target_scaler.fit_transform(y_data.values.reshape(-1, 1))

        x, y = [], []
        for i in range(self.lookback, len(x_scaled)):
            x.append(x_scaled[i - self.lookback:i])
            y.append(y_scaled[i])

        x, y = np.array(x), np.array(y)

        return x, y, target_scaler

    def train_test_split(self, x, y):
        split_index = int(len(x) * self.split_ratio)
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return x_train, y_train, x_test, y_test

    def build_model(self, input_shape, num_heads=8, num_layers=2, dff=128, dropout_rate=0.2):
        inputs = Input(shape=input_shape)

        # CNN layer
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)

        # Bi-LSTM layer
        x = Bidirectional(LSTM(units=200, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)
        x = Bidirectional(LSTM(units=200, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)

        # Transformer layers
        for _ in range(num_layers):
            x = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1], dropout=dropout_rate)(x, x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization(epsilon=1e-6)(x)

            x_ffn = TimeDistributed(Dense(dff, activation='relu'))(x)
            x_ffn = TimeDistributed(Dense(x.shape[-1], activation='relu'))(x_ffn)

            x_ffn = Dropout(dropout_rate)(x_ffn)
            x = LayerNormalization(epsilon=1e-6)(x + x_ffn)

        x = Flatten()(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def fit_model(self, model, x_train, y_train, epochs=100, batch_size=32):
        checkpoint_filepath = '/kaggle/working/model_checkpoint.h5'
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        self.history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.2, callbacks=[checkpoint_callback, reduce_lr_callback])

    def predict(self, model, x_test, scaler):
        y_pred = model.predict(x_test)
        self.predicted = scaler.inverse_transform(y_pred)
        return self.predicted

    def evaluate_model(self, model, x_test, y_test, scaler):
        y_pred = model.predict(x_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test)
        mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
        mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mase = mean_absolute_error(y_test, y_pred) / np.mean(np.abs(np.diff(y_test, axis=0)))
        print(f"Mean Squared Error (MSE): {mse.mean()}")
        print(f"Mean Absolute Error (MAE): {mae.mean()}")
        print(f"Root Mean Squared Error (RMSE): {rmse.mean()}")
        print("R^2 score:", r2)
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
        print(f"Mean Absolute Scaled Error (MASE): {mase}")

    def plot_history(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    crypto_ohlc = CryptoOHLC(pair='XXBTZEUR', interval=60, lookback=120, split_ratio=0.8, target_col="open")
    x, y, target_scaler = crypto_ohlc.preprocess_data()
    x_train, y_train, x_test, y_test = crypto_ohlc.train_test_split(x, y)

    lstm_transformer_model = crypto_ohlc.build_model(x_train.shape[1:])
    inception_time_model = build_inception_time_model(x_train.shape[1:])
    ensemble_model = build_ensemble_model(lstm_transformer_model, inception_time_model)

    # Train the ensemble model
    ensemble_model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1)

    # Evaluate the ensemble model
    y_train_pred = ensemble_model.predict(x_train)
    y_test_pred = ensemble_model.predict(x_test)

    # Reverse the target scaling
    y_train = target_scaler.inverse_transform(y_train)
    y_test = target_scaler.inverse_transform(y_test)
    y_train_pred = target_scaler.inverse_transform(y_train_pred)
    y_test_pred = target_scaler.inverse_transform(y_test_pred)

    # Calculate and print the R2 score
    train_r2_score = r2_score(y_train, y_train_pred)
    test_r2_score = r2_score(y_test, y_test_pred)

    print("Train R2 Score: {:.4f}".format(train_r2_score))
    print("Test R2 Score: {:.4f}".format(test_r2_score))

    