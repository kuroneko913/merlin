import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Highway
from keras.layers import Dropout
from keras.regularizers import l1l2
import logging


class DNN(object):
    def __init__(self, n_in, hidden_layer_size,
                 n_out, L1_reg, L2_reg, hidden_layer_type,
                output_type='LINEAR', dropout_rate=0.0):

        logger = logging.getLogger("DNN initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_layers = len(hidden_layer_size)
        self.dropout_rate = dropout_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.optimizer = 'sgd'
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
        # Number of hidden layers and their types should be equal
        assert len(hidden_layer_size) == len(hidden_layer_type)
        ### Create model graph ###
        self.model = Sequential()
        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i - 1]
            self.model.add(Dense(
                        output_dim=hidden_layer_size[i],
                        input_dim=input_size,
                        init='glorot_uniform',
                        activation=hidden_layer_type[i].lower(),
                        W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
            self.model.add(Dropout(self.dropout_rate))
        
        # add output layer
        if output_type.lower() == 'linear':
            self.final_layer = self.model.add(Dense(
                                output_dim=n_out,
                                input_dim=hidden_layer_size[-1],
                                init='glorot_uniform',
                                activation='linear',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        elif output_type.lower() == 'sigmoid':
            self.final_layer = self.model.add(Dense(
                                output_dim=hidden_layer_size[i],
                                input_dim=input_size,
                                init='glorot_uniform',
                                activation='sigmoid',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        else:
            logger.critical("This output activation function: %s is not supported right now!" %(output_type))
            sys.exit(1)
        
        # Compile the model
        self.model.compile(loss='mse', optimizer=self.optimizer)

class RNN(object):
   def __init__(self, n_in, hidden_layer_size,
                 n_out, L1_reg, L2_reg, hidden_layer_type,
                output_type='LINEAR', dropout_rate=0.0,time_step=1):

        logger = logging.getLogger("LSTM initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_layers = len(hidden_layer_size)
        self.dropout_rate = dropout_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
        # Number of hidden layers and their types should be equal
        assert len(hidden_layer_size) == len(hidden_layer_type)
        ### Create model graph ###
        self.model = Sequential()
        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i - 1]
            self.model.add(LSTM(
                        output_dim=hidden_layer_size[i],
                        batch_input_shape=(None,time_step,input_size),
                        W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        
        # add output layer
        if output_type.lower() == 'linear':
            self.final_layer = self.model.add(Dense(
                                output_dim=n_out,
                                input_dim=hidden_layer_size[-1],
                                init='glorot_uniform',
                                activation='linear',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        elif output_type.lower() == 'sigmoid':
            self.final_layer = self.model.add(Dense(
                                output_dim=hidden_layer_size[i],
                                input_dim=input_size,
                                init='glorot_uniform',
                                activation='sigmoid',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        else:
            logger.critical("This output activation function: %s is not supported right now!" %(output_type))
            sys.exit(1)
        
        # Compile the model
        self.model.compile(loss='mse', optimizer='adam')

class Hybrid(object):
    def __init__(self, n_in, hidden_layer_size,
                 n_out, L1_reg, L2_reg, hidden_layer_type,
                output_type='LINEAR', dropout_rate=0.0,time_step=1):

        logger = logging.getLogger("LSTM initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_layers = len(hidden_layer_size)
        self.dropout_rate = dropout_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
        # Number of hidden layers and their types should be equal
        assert len(hidden_layer_size) == len(hidden_layer_type)
        ### Create model graph ###
        self.model = Sequential()
        # add hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_size[i - 1]
            
            if hidden_layer_type[i].lower() == 'tanh' or hidden_layer_type[i].lower() == 'sigmoid':
                self.model.add(TimeDistributed(
                        output_dim=hidden_layer_size[i],
                        input_shape = (time_step,input_size),
                        init='glorot_uniform',
                        activation=hidden_layer_type[i].lower(),
                        W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
                self.model.add(Dropout(self.dropout_rate))
    
            elif hidden_layer_type[i].lower() == 'lstm':
                self.model.add(LSTM(
                            output_dim=hidden_layer_size[i],
                            batch_input_shape=(None,time_step,input_size),
                            W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
            else:
                logger.critical("This layer: %s is not supported right now!" %(hidden_layer_type[i]))
        # add output layer
        if output_type.lower() == 'linear':
            self.final_layer = self.model.add(Dense(
                                output_dim=n_out,
                                input_dim=hidden_layer_size[-1],
                                init='glorot_uniform',
                                activation='linear',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        elif output_type.lower() == 'sigmoid':
            self.final_layer = self.model.add(Dense(
                                output_dim=hidden_layer_size[i],
                                input_dim=input_size,
                                init='glorot_uniform',
                                activation='sigmoid',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        else:
            logger.critical("This output activation function: %s is not supported right now!" %(output_type))
            sys.exit(1)
        
        # Compile the model
        self.model.compile(loss='mse', optimizer='adam')

class HWAY(object):
    def __init__(self, n_in, hidden_layer_size,
                 n_out, L1_reg, L2_reg, hidden_layer_type,
                output_type='LINEAR', dropout_rate=0.0):

        logger = logging.getLogger("DNN initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_layers = len(hidden_layer_size)
        self.dropout_rate = dropout_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.optimizer = 'adam'
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
        # Number of hidden layers and their types should be equal
        assert len(hidden_layer_size) == len(hidden_layer_type)
        ### Create model graph ###
        self.model = Sequential()
        self.model.add(Dense(
                        output_dim=128,
                        input_dim=n_in,
                        init='glorot_uniform',
                        activation='relu',
                        W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        self.model.add(Dropout(self.dropout_rate))
        num_layers = 15
        for i in xrange(num_layers):
            self.model.add(Highway(activation = 'relu'))
            self.model.add(Dropout(self.dropout_rate))
        #self.model.add(Dropout(dropout))

        # add output layer
        if output_type.lower() == 'linear':
            self.final_layer = self.model.add(Dense(
                                output_dim=n_out,
                                input_dim=hidden_layer_size[-1],
                                init='glorot_uniform',
                                activation='linear',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        elif output_type.lower() == 'sigmoid':
            self.final_layer = self.model.add(Dense(
                                output_dim=n_out,
                                input_dim=hidden_layer_size[-1],
                                init='glorot_uniform',
                                activation='sigmoid',
                                W_regularizer=l1l2(l1=self.L1_reg,l2=self.L2_reg)))
        else:
            logger.critical("This output activation function: %s is not supported right now!" %(output_type))
            sys.exit(1)
        
        # Compile the model
        self.model.compile(loss='mse', optimizer=self.optimizer)