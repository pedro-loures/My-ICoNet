
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def MLP_regressor(input_shape, activation='tanh', final_activation='sigmoid', num_classes=1):

    model = Sequential()
    # Camada com 30 neurônios
    model.add(Dense(50, activation=activation, input_shape=(input_shape,)))
    # Dropout de 20%
    model.add(Dropout(0.1))
    # Camada de 20 neurônios
    model.add(Dense(50, activation=activation))
    # Camada de 10 neurônios
    model.add(Dense(50, activation=activation))
    # Camada de 10 neurônios
    model.add(Dense(50, activation=activation))
    # Camada de 10 neurônios
    model.add(Dense(50, activation=activation))

    # Camda de classificação final, com 1 neurônio para cada classe de saída. Softmax divide a probabilidade de cada classe.
    model.add(Dense(num_classes, activation=final_activation))

    return model