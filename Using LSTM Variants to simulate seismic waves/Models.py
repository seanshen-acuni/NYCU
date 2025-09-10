# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 05:36:22 2024

@author: jimmy
"""

from keras.utils import plot_model
import visualkeras
from PIL import ImageFont

def MultiLSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.metrics import RootMeanSquaredError

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM input layer and some Dropout regularisation
    model.add(
        LSTM(
            units = layer_units,
            activation=activateType,
            return_sequences = True,
            input_shape = (time_steps, input_sizes)
        )
    )
    model.add(Dropout(0.2))

    for i in range(0,n_layers):
        model.add(
            LSTM(
                units = layer_units,
                activation=activateType,
                return_sequences = True
            )
        )
        model.add(Dropout(0.2))


    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(
        LSTM(
            units = layer_units,
            activation=activateType
        )
    )
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(
        Dense(
            units = output_sizes,
            activation="linear"
        )
    )
    
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics = [RootMeanSquaredError()]
    )
    
    model.summary()
    plot_model(model, show_shapes=True, to_file='MultiLayer LSTM.png')
    return 'MultiLayer LSTM', model


def LSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.metrics import RootMeanSquaredError

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM input layer and some Dropout regularisation
    
    model.add(Input(shape = (time_steps, input_sizes), name = 'Input'))
    
    model.add(LSTM(units = layer_units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            name ='LSTM'))
    
    # Adding the output layer
    model.add(Dense(units = output_sizes,
            activation="linear",
            name ='Output'))
    
    model.compile(loss='mean_squared_error',
        optimizer='adam',
        metrics = [RootMeanSquaredError()])
    
    model.summary()
    model.save('LSTM.keras')
    
    #plot_model(model, show_shapes=True, show_layer_names=True, to_file='LSTM.png')
    #import visualkeras
    #visualkeras.layered_view(model, legend=True,spacing=60) # without custom font  
    #visualkeras.graph_view(model)
    #from PIL import ImageFont
    return 'LSTM', model

def AELSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential, Model
    from keras.metrics import RootMeanSquaredError
    from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Dropout, Flatten, Reshape
    from tensorflow.keras.optimizers import Adam
    from keras.utils import plot_model
    
        # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(time_steps,1)))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
    
    return 'Autoencoder-LSTM', model

def SATLSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential, Model
    from keras.layers import Input, LSTM, Attention, Dense, Dropout, Flatten
    from keras.metrics import RootMeanSquaredError

    # Initialising the RNN
    model = Sequential()
    
    input_shape = (time_steps, input_sizes)
    
    inputs = Input(shape=input_shape)
    
    lstm_out = LSTM(units = layer_units,
                    activation = 'tanh',
                    recurrent_activation = 'hard_sigmoid',
                    return_sequences = True)(inputs)
    
    attention_out = Attention()([lstm_out, lstm_out])
    
    attention_out = Flatten()(attention_out)
    
    dense_out = Dense(64, activation='relu')(attention_out)
    dropout_out = Dropout(0.2)(dense_out)
    
    output = Dense(output_sizes, activation = 'linear')(dropout_out)
    
    model = Model(inputs = inputs, outputs = output)
    
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics = [RootMeanSquaredError()]
    )
    
    model.summary()
    return 'Self-attention LSTM', model

def CNNLSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.metrics import RootMeanSquaredError
    from keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Reshape, TimeDistributed, GlobalAveragePooling1D
    from keras.models import Model
    from tensorflow.keras.optimizers import Adam, AdamW
    
    # define model
    model = Sequential(name = 'CNNLSTMModel')
    
    model.add(Input(shape = (None, time_steps, 1), name = 'Input'))
    
    model.add(TimeDistributed(
        Conv1D(filters=128, 
               kernel_size=1, 
               activation='relu'), 
        name = '1DConvolution'))
    
    model.add(TimeDistributed(
        GlobalMaxPooling1D(),
        name = '1DGlobalMaxPooling'))
    
    model.add(LSTM(60, activation='tanh', name = 'LSTM'))
    model.add(Dense(1, name = 'Output'))
    model.compile(loss='mean_squared_error', 
                  optimizer="adam", 
                  metrics = [RootMeanSquaredError()])
    
    model.summary()
    model.save('CNN-LSTM.keras')
    
    return 'CNN-LSTM', model

def CNN(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    # Import the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Flatten
    from keras.metrics import RootMeanSquaredError
    from keras.layers import Input, Dense, MaxPooling1D, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential()
    
    model.add(Input(shape = (time_steps, input_sizes), name = 'Input'))
    model.add(Conv1D(filters=128, kernel_size = time_steps, strides = 1, activation="relu", name = '1DConvolution'))
    model.add(GlobalMaxPooling1D(name = '1DGlobalMaxPooling'))
    model.add(Dense(1, activation = 'linear', name = 'Output'))
    
    model.compile(loss='mean_squared_error', 
                  optimizer="adam", 
                  metrics = [RootMeanSquaredError()])
    
    model.summary()
    
    import visualkeras
    model.save('CNN.keras')
    
    import visualkeras
    visualkeras.layered_view(model, legend=True,spacing=60)
    
    return 'CNN', model

def BILSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType):
    
    from keras.layers import Dense, LSTM, Bidirectional, Dropout, Input
    from keras.models import Sequential
    from keras.metrics import RootMeanSquaredError

    # building model
    model = Sequential(name = 'BidirectionalLSTM')
    
    model.add(Input(shape = (time_steps, input_sizes), name = 'Input'))

    #input layer
    model.add(Bidirectional(
                LSTM(
                units=layer_units,
                activation='tanh',
                recurrent_activation = 'hard_sigmoid'),
                name = 'BidirectionalLSTM'))

    #output layer
    model.add(Dense(
            units = output_sizes,
            activation="linear",
            name = 'Output'))

    #compile
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=[RootMeanSquaredError()]
    )

    model.summary()
    model.save('BILSTM.keras')
    plot_model(model, show_shapes=True, to_file='Bidirectional-LSTM.png')
    return 'Bidirectional-LSTM', model