import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
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
        #self.is_train = T.iscalar('is_train')
        assert len(hidden_layer_size) == len(hidden_layer_type)
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
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
        self.model.compile(loss='mse', optimizer='adam')

class RNN(object):
   def __init__(self, n_in, hidden_layer_size,
                 n_out, L1_reg, L2_reg, hidden_layer_type,
                output_type='LINEAR', dropout_rate=0.0,time_step=1):

        logger = logging.getLogger("LSTM initialization")

        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_layers = len(hidden_layer_size)
        self.dropout_rate = dropout_rate
        #self.is_train = T.iscalar('is_train')
        assert len(hidden_layer_size) == len(hidden_layer_type)
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        # fix random seed for reproducibility
        seed = 123
        np.random.seed(seed=seed)
        # Model must have atleast one hidden layer
        assert self.n_layers > 0, 'Model must have at least one hidden layer'
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
