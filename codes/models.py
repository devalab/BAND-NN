from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

def get_bonds_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=17))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def get_angles_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=27))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def get_nonbonds_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=17))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

def get_dihedrals_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=38))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model
