# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 10:26:19 2020

@author: Hanss401
"""
import numpy as np;
import re;
from keras.layers import Input, Dense,Conv2D, MaxPooling2D,Flatten;
from keras.layers.merge import concatenate;
from keras.models import Model, load_model;
from keras.models import Sequential;
from keras.layers import Embedding;
from keras.layers import LSTM;
from keras.layers import Bidirectional;
from keras.layers import Dense,Reshape;
from keras.layers import TimeDistributed;
from keras.layers import Dropout;
from keras.layers import Add,Dot;
from keras import backend as K;
from keras.layers.recurrent import GRU;
from keras import optimizers;
from keras import losses;
import matplotlib.pyplot as plt;
import keras;
import os;
import sys;
# /usr/local/lib/python2.7/dist-packages/keras/layers/merge.py

# ========= define evaluation ==========
def mean_pred(y_true, y_pred):
    return K.square(y_pred-y_true);    

# ========= define task func ===========
def task_finish(y_true, y_pred):
    return K.mean(K.greater(K.dot(K.softmax(y_pred), K.transpose(y_true)),.3), axis=-1)

# ========= load dataset ================
# ----- for MODEL_RWD --------
DATA_STATE  = np.load('DATA_STATE.npy');
DATA_RWD    = np.load('DATA_RWD.npy');
DATA_PRED   = np.load('DATA_PRED.npy');
DATA_PRED_NEXT = np.load('DATA_PRED_NEXT.npy');
# ----- for MODEL_RSN --------
#DATA_PRED   = np.load('DATA_PRED.npy');
DATA_ACTION = np.load('DATA_ACTION.npy');
#DATA_TF     = np.load('DATA_TF.npy').astype('float32');

# ========= define constant data==========
DIM_STATE  = 9;
DIM_PRED   = 4;
DIM_ACTION = 4;
DIM_REPR   = DIM_PRED**2;

# ========= define f_P ============
f_P_in  = Input(shape=(DIM_STATE,), dtype='float32', name='f_P_in'); # STATE inputed;
f_P_md  = Dense(16, activation='relu')(f_P_in);
f_P_md  = Dense(32, activation='relu')(f_P_md);
f_P_md  = Dense(16, activation='relu')(f_P_md);
f_P_md  = Dense(8, activation='relu')(f_P_md);
f_P_out = Dense(DIM_PRED, activation='tanh',name='f_P_out')(f_P_md); # PREDICATE output;
# ========= define f_V ============
f_V_in  = Dense(16, activation='relu')(f_P_out); # PREDICATE inputed;
f_V_md  = Dense(32, activation='relu')(f_V_in);
f_V_md  = Dense(16, activation='relu')(f_V_md);
f_V_md  = Dense(4, activation='relu')(f_V_md);
f_V_out = Dense(1, activation='linear')(f_V_md); # REWARD output;
# ========= define f_R ============
f_R_in  = Input(shape=(DIM_ACTION,), dtype='float32', name='f_R_in'); # ACTION inputed;
f_R_md  = Dense(16, activation='relu')(f_R_in);
f_R_md  = Dense(32, activation='relu')(f_R_md);
f_R_md  = Dense(16, activation='relu')(f_R_md);
f_R_md  = Dense(8, activation='relu')(f_R_md);
f_R_md  = Dense(DIM_REPR, activation='tanh')(f_R_md);
f_R_out = Reshape((DIM_PRED, DIM_PRED), input_shape=(DIM_REPR,))(f_R_md); # REPRESENTATION output;
# ========= define f_I ============
f_I_out = Dot(1)([f_R_out,f_P_out]); # INFERENCE output;

# ========= define MODEL_RWD ============
MODEL_RWD = Model(inputs=f_P_in, outputs=f_V_out);
sgd       = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.4, nesterov=True);
MODEL_RWD.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy']);
# ========= define MODEL_RSN ============
MODEL_RSN = Model(inputs=[f_R_in,f_P_in], outputs=f_I_out);
sgd       = optimizers.SGD(lr=0.0001, decay=0.0, momentum=0.4, nesterov=True);
MODEL_RSN.compile(optimizer=sgd, loss=losses.mean_squared_error, metrics=['accuracy']);

# ========= test model ================
# MODEL_RWD = load_model('MODEL_RWD.h5');
# RES = MODEL_RWD.predict(DATA_STATE[0:10]);
# print(RES);
# print(DATA_RWD[0:10]);

MODEL_RSN = load_model('MODEL_RSN.h5');
RES = MODEL_RSN.predict([DATA_ACTION[0:10],DATA_PRED[0:10]]);
print(RES);
# print(DATA_PRED_NEXT[0:10]);

#sys.exit(1); # ===========================================>!!!!!!!!!!!!!!!!!!!!!!!!!!

# ========= test DATA_ACTION_REPR ================
DATA_ACTION_REPR  = np.load('DATA_ACTION_REPR.npy');
for i in range(10):
    print(np.dot(DATA_PRED[i],DATA_ACTION_REPR[i]));
sys.exit(1); # ===========================================>!!!!!!!!!!!!!!!!!!!!!!!!!!

# ========= make DATA_ACTION_REPR =============
MODEL_RSN = load_model('MODEL_RSN.h5');
#print(MODEL_RSN.layers[6]);
MODEL_fR = Model(inputs=MODEL_RSN.input, outputs=MODEL_RSN.layers[6].output);
DATA_ACTION_REPR = MODEL_fR.predict([DATA_ACTION,DATA_PRED]);
#print(DATA_ACTION_REPR);
#print(DATA_ACTION_REPR.shape);
np.save('DATA_ACTION_REPR',np.array(DATA_ACTION_REPR));
#sys.exit(1); # ===========================================>!!!!!!!!!!!!!!!!!!!!!!!!!!

# ========= make DATA_PRED and DATA_PRED_NEXT =============
MODEL_RWD      = load_model('MODEL_RWD.h5');
DATA_STATE  = np.load('DATA_STATE.npy');
DATA_STATE_NEXT  = np.load('DATA_STATE_NEXT.npy');
#print(MODEL_RWD.layers[5].activation);
#sys.exit(1); # ===========================================>!!!!!!!!!!!!!!!!!!!!!!!!!!
MODEL_fP = Model(inputs=MODEL_RWD.input, outputs=MODEL_RWD.layers[5].output)
DATA_PRED      = MODEL_fP.predict(DATA_STATE);
DATA_PRED_NEXT = MODEL_fP.predict(DATA_STATE_NEXT);
#print(DATA_PRED);
#print(DATA_PRED_NEXT);
np.save('DATA_PRED',np.array(DATA_PRED));
np.save('DATA_PRED_NEXT',np.array(DATA_PRED_NEXT));