import numpy as np
np.random.seed(42)
from keras import backend as K
from keras.regularizers import *
import sys
import pandas as pd
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,LSTM,Input,Embedding, Bidirectional, TimeDistributed
from keras.layers import Dropout
import keras
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
import pandas as pd
import pickle
from keras.optimizers import SGD,Adam
import tensorflow as tf
import ast
from collections import Counter
from keras.callbacks import *
from sklearn.metrics import roc_auc_score,recall_score,precision_score, f1_score
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
################################################################################################################
#                  Functions
################################################################################################################

def build_masked_loss(loss_function, mask_value):
    """Builds a loss function that masks based on targets
    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """
    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)
    return masked_loss_function

def auc(pred,targets):
    """Calculates the auc for each task masking missing values
    Args:
        pred: predictions
        targets: labels
    Returns:
        numpy array with AUC for each task
        """
    auc =[]
    for i in range(len(pred)):
        mask = targets[:,i]!=-1
        p = pred[i][mask]
        #type(p)
        t = targets[mask,i]
        auc.append(roc_auc_score(t,p))
    auc = np.array(auc)
    return auc

# def prec(pred,targets):
#     pre = []
#     for i in range(len(pred)):
#         mask = targets[:,i]!=-1
#         p = pred[i][mask]
#         #type(p)
#         t = targets[mask,i]
#         pre.append(precision_score(t,p))
#     pre= np.array(pre)
#     return pre

# def sklearn_measure_multioutput(pred,targets,function,threshold):
#     container=[]
#     for i in range(len(pred)):
#         pred[i][pred[i]>threshold] = 1
#         pred[i][pred[i] <= threshold] = 0
#         mask = targets[:,i]!=-1
#         p = pred[i][mask]
#         #type(p)
#         t = targets[mask,i]
#         container.append(function(t,p))
#     return np.array(container)

def build_masked_metric(metrics, mask_value):
    """Builds a matric that masks based on targets
    Args:
        matrics: The metric to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a metric that acts like metric with masked inputs
    """
    def masked_metric_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return metrics(y_true * mask, y_pred * mask)
    return masked_metric_function

def calculate_weights(targets):
    dic = {}
    for i,k in zip(range(targets.values.shape[1]),[str(x) for x in range(12)]):
        dic_c = Counter(targets.values[:,i])
        dic[k] = {0:dic_c[1]/dic_c[0],-1:0,1:(dic_c[0])/dic_c[0]}
    return dic

################################################################################################################

def main():
    training = sys.argv[sys.argv.index('--training') + 1]
    target = sys.argv[sys.argv.index('--targets') + 1]
    lr = float(sys.argv[sys.argv.index('--lr') + 1])
    hidden = int(sys.argv[sys.argv.index('--h') + 1])
    nb_epochs = int(sys.argv[sys.argv.index('--n_ep') + 1])
    training = pd.read_csv(training,index_col=0)
    targets = pd.read_csv(target,index_col=0)
    save_model = int(sys.argv[sys.argv.index('--save_model') + 1])
    #training = pd.read_csv("trainings_tox21_net.csv", index_col=0)
    #targets = pd.read_csv("targets_tox21_net.csv", index_col=0)
    #lr=0.1
    #nb_epochs = 100
    
    #build model
    print("#" * 50)
    print(" " * 15 + "BUILDING THE MODEL ...")
    print("#" * 50)
    print(" " * 50)
    #lr=0.1
    inputs = Input(shape=(2048,))
    d=Dense(1024, activation="selu",kernel_initializer="lecun_normal")(inputs)
    d=Dropout(0.7)(d)
    if hidden > 0:
        for i in range(hidden-1):
            d=Dense(1024, activation="selu",kernel_initializer="lecun_normal")(d)
            d=Dropout(0.7)(d)
    # d=Dense(1024, activation="selu",kernel_initializer="lecun_normal")(d)
    # d=Dropout(0.7)(d)
    dic = {}
    for i in [str(x) for x in range(12)]:
        dic[i] = Dense(1, activation="sigmoid", name=i)(d)

    net = Model(inputs, [dic[x] for x in [str(x) for x in range(12)]])
    sgd = SGD(lr=lr, decay=1e-10, momentum=0.9, nesterov=True)
    net.compile(loss=build_masked_loss(K.binary_crossentropy, -1), optimizer=sgd, metrics=[build_masked_metric(keras.metrics.binary_accuracy, -1)])
    net.summary()
    #loss=build_masked_loss(K.binary_crossentropy, -1)
    #checkpoint = ModelCheckpoint("model_tox21.best1.hdf5", monitor='loss', verbose=1, save_best_only=True,save_weights_only=False, mode='auto')
    #early_stopping = EarlyStopping(monitor='loss', patience=1000, mode="auto")
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    #callback_list = [checkpoint, early_stopping, reduce_lr]
    #model.compile(loss=build_masked_loss(K.binary_crossentropy,-1), optimizer=sgd,metrics=["binary_accuracy"])
    #model.fit(training,targets,validation_split=0.2,callbacks=callback_list,epochs=50,batch_size=1000)
    print(" " * 50)
    print("#" * 50)
    print(" " * 22 + "TRAINING ...")
    print("#" * 50)
    print(" " * 50)
    #train model and calculate auc avarage between tasks
    class_weight = calculate_weights(targets)
    #class_weight = {0:0.1,1:100,-1:0}
    net.fit(training.values[0:11353,:],[targets.values[0:11353,i] for i in range(12)],
            validation_split=0.2,epochs=nb_epochs,batch_size=1000, class_weight = class_weight)
    net.evaluate(training.values[11353:,:] ,[targets.values[11353:,i] for i in range(12)])
    pred = net.predict(training.values[12000:,:])
    print(" " * 50)
    print("#"*50)
    print(" "*20 + "PREDICTING ...")
    print("#" * 50)
    print(" " * 50)
    print("AUC per task => "+ str(auc(pred, targets.values[12000:,:])))
    print(" " * 50)
    print("AUC          => "+str(auc(pred, targets.values[12000:,:]).mean()))
    #print(" " * 50)
    #print("precision => " + str(sklearn_measure_multioutput(pred,targets.values[12000:,:],
    # precision_score,0.5).mean()))
    #print(" " * 50)
    #print("recall => "+ str(sklearn_measure_multioutput(pred, targets.values[12000:,:],
    #  recall_score, 0.5).mean()))
    #print(" " * 50)
    #print("f1_score => "+ str(sklearn_measure_multioutput(pred,targets.values[12000:,:],
    # f1_score,0.5).mean()))
    #print(" " * 50)
    print("#" * 50)
    print(" " * 22 + "THE END!")
    print("#" * 50)
    if save_model:
        print("Model saved as: tox21_FID_NET.h5")
        net.save("tox21_FID_NET.h5")


if __name__ == "__main__":
    main()