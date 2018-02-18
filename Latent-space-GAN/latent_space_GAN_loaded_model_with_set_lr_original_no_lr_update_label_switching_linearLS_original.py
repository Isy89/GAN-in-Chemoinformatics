import numpy as np
import sys
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM, Dropout, Bidirectional, RepeatVector, TimeDistributed, \
    Embedding
import random
from keras.models import Model
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy as np
np.random.seed(42)
from keras import optimizers
#from keras.layers.core import Highway
from keras.layers import Input,merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.noise import GaussianNoise
#import pydot
#import graphviz
import pandas as pd
import os
#import rdkit
#from rdkit.Chem import AllChem
import pandas as pd
import random
import tqdm
import sys
import pickle
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.optimizers import SGD,Adam
import keras
from keras.layers.core import Reshape
####

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

####CUDA_VISIBLE_DEVICES=0 python3.5 Test_Final_Program_test_new_GAN1.py X_Chemble_40_15.csv --dic_file DICTIONARY_l2n.csv --lr0 0.001 --lr1 0.001 --g_hidden 1 --d_hidden 1 --nb_epochs 10000 --title test --batchsize 1000 --pret_ep 400
##############################################################################################################
#                                  Functions
##############################################################################################################
def convert_classes_to_numbers(df):
    """
    :param df: pandas dataframe of numpy array with ndim == 3 
    :return: numpy array with ndim == 2 where the argmax along dim 2 has been taken
    """
    if df.__class__.__name__ == "DataFrame":
        df = df.values
    elif df.__class__.__name__ == "ndarray":
        df = df
    else:
        print("Not implemented for this class")
    if df.ndim != 3:
        raise ValueError("dimensions should be equal to 3 are instead: " + str(arr.ndim ) )
    return np.apply_along_axis(np.argmax, axis=2, arr=df)

def load_dic_form_csv(file,key_col=0):
    """
    :param file: file where the dic.csv is saved in two columns key and val
    :param key_col: usees the first column or the second to set the keys
    :return: dictionary with keys corresponding to the indicated key_col and 
             as values the second one.         
    """
    with open(file,"r") as f:
        line = f.readline()
        sep = ","
        for i in [";",":",",","\t","\s"]:
            if i in line:
                sep = i
    dictionary=pd.read_csv(file,index_col=0,sep=sep)
    if key_col == 0:
        return  {k:v for k,v in zip(dictionary.values[:,0],dictionary.values[:,1])}
    else:
        return  {v:k for k,v in zip(dictionary.values[:,0],dictionary.values[:,1])}

def translate_from_to_smiles(df,key = "num",file = "file",dic = "dic"):
    """
    :param df:   pandas dataframe or numpy array with dim = 2
    :param key:  if set to num it translates an array of numbers to letters.
                 loading the dic through the load_dic_from_csv(df,key_col = 1 ) function
                 in this case the parameter file has to be provided
                 if set to let it translates an array of letters to numbers 
                 loading a dic through the load_dic_from_csv(df,key_col=0) function
                 in this case the parameter file has to be provided
                 if set to dic the dic has to be provided in the dic parameter 
                 and the file has not to be set
    :param file: path to the .csv file where the file is stored
    :param dic:  dictionary 
    :return:     numpy array of the same dimension with the symbols translated either
                 num to let or viceversa
    """
    if df.__class__.__name__ == "DataFrame":
        df = df.values
    elif df.__class__.__name__ == "ndarray":
        df = df
    else:
        print("Not implemented for this class")
    if key not in ["num","let","dic"]:
        raise ValueError("key has to be either 'num' or 'let'")
    def dic_get(ls,dic1):
        return [dic1[x] for x in ls]
    if key == "num":
        dic = load_dic_form_csv(file,key_col=1)
    if key == "let":
        dic = load_dic_form_csv(file,key_col=0)
    if key == "dic":
        pass
    return np.apply_along_axis(dic_get,axis=1,arr = df,dic1=dic)

def generate_one_hot(df, vocabulary_size=50):
    flat_x = df.values.flatten()
    on_hot_data = to_categorical([x for x in range(vocabulary_size)], nb_classes=vocabulary_size)
    one_hot_dictionary = {k: v for k, v in zip([x for x in range(vocabulary_size)], on_hot_data)}
    flat_x_one_hot = np.array([one_hot_dictionary[x] for x in flat_x])
    data_one_hot = np.reshape(flat_x_one_hot, newshape=(df.shape[0], df.shape[1], vocabulary_size))
    return data_one_hot

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def load_GAN_from_G_and_D(Gi,Di,lr0,lr1):
    print("enter load models Gi and Di and build GAN")
    print(Gi)
    print(Di)
    print("learning rates = "+str(lr0)+" "+ str(lr1))
    generator = keras.models.load_model(Gi)
    generator.summary()
    generator.name="generator"
    print("set lr Gi")
    K.set_value(generator.optimizer.lr,lr0)
    print("lr Gi => "+ str(K.get_value(generator.optimizer.lr)))
    print("load Di")
    discriminator = keras.models.load_model(Di)
    discriminator.summary()
    discriminator.name="discriminator"
    print("set Di lr")
    K.set_value(discriminator.optimizer.lr,lr1)
    print("Di lr => "+str(str(K.get_value(discriminator.optimizer.lr))))
    # Build GAN
    gan_input = keras.layers.Input(shape=(1000, ))
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = keras.models.Model(gan_input, gan_V)
    opt1 = keras.optimizers.SGD(lr=lr0)
    GAN.compile(loss='binary_crossentropy', optimizer=opt1)
    GAN.summary()
    return generator, discriminator, GAN

def pre_train_discriminator(dataframe, discriminator, generator, ntrain=10000,E="E",pre_num_epochs=3):
    training = dataframe
    print(training)
    mu, sigma = 0, 1  # mean and standard deviation
    for i in range(pre_num_epochs):
        trainidx = random.sample(range(0, training.shape[0]), ntrain)
        print("start encoding")
        XT = training[trainidx, :]
        # Pre-train the discriminator network
        # generate random samples coming from a standard uniform distribution
        print("start noise generation")
        noise_gen = np.random.normal(mu, sigma, size=[XT.shape[0], 1000])
        # forward pass through the generator to generate the sample sampled from the generator
        print("start generator")
        generated_vector = generator.predict(noise_gen)
        # first samples coming from the generator and those one coming from the trainingset
        # encoded through the encoder part of the autoencoder and data concatenated
        print("concatenation")
        X = np.concatenate((XT, generated_vector))
        n = XT.shape[0]
        y = np.zeros([2 * n, 1])
        y[:n] = 1
        make_trainable(discriminator, True)
        print("train d")
        d_loss = discriminator.train_on_batch(X, y)
        print("epochs: "+str(i)+" :: d_loss: "+str(d_loss))
    return d_loss[0]


# Set up our main training loop
def train_for_n(dataframe, generator, discriminator, GAN, nb_epoch=5000, BATCH_SIZE=32, title="figure", name=7,E = "E",dic = "dic",D="D" ):
    losses = {"d": [], "g": []}
    training = dataframe
    mu, sigma = 0, 1  # mean and standard deviation
    # generate vectors
    make_trainable(discriminator, True)
    print("Generating the fake data and sampling the real one")
    vec_batch = training[random.sample(range(0,training.shape[0]),BATCH_SIZE), :]
    noise_gen = np.random.normal(mu, sigma, size=[BATCH_SIZE, 1000])
    generated_vector = generator.predict(noise_gen)
    # Train discriminator on samples vector
    print("Training discriminator on real samples")
    X = vec_batch
    y = np.zeros([BATCH_SIZE, 1])
    # y[0:BATCH_SIZE] = np.random.uniform(0.9, 1.1) # smoothing the lables
    y[0:BATCH_SIZE] = 1
    d_loss_r = discriminator.train_on_batch(X, y)

    # Train discriminator on generated vector
    print("Training discriminator on fake samples")
    X = generated_vector
    y = np.zeros([BATCH_SIZE, 1])
    y[0:BATCH_SIZE] = 0
    d_loss_f = discriminator.train_on_batch(X, y)

    # train Generator-Discriminator stack on input noise
    print("Training generator")
    make_trainable(discriminator, False)
    noise_tr = np.random.normal(mu, sigma, size=[2*BATCH_SIZE,1000])
    y2 = np.zeros([2*BATCH_SIZE, 1])
    y2[:] = 1
    g_loss = GAN.train_on_batch(noise_tr, y2)

    for e in tqdm.tqdm(range(nb_epoch)):
        if (e % 25) == 0:
            print("#"*50)
            print(" "*20 + "learning rates update" )
            print("#"*50)
            K.set_value(generator.optimizer.lr,K.get_value(generator.optimizer.lr)*0.99)
            print("gnerator lr => " + str(K.get_value(generator.optimizer.lr)))
            K.set_value(discriminator.optimizer.lr,K.get_value(discriminator.optimizer.lr)*0.99)
            print("discriminator lr => "+ str(K.get_value(discriminator.optimizer.lr)))
            K.set_value(GAN.optimizer.lr,K.get_value(GAN.optimizer.lr)*0.99)
            print("GAN lr => "+ str(K.get_value(GAN.optimizer.lr)))
            print("#"*50)
        print("GAN lr => "+ str(K.get_value(GAN.optimizer.lr)))
        print("discriminator lr => "+ str(K.get_value(discriminator.optimizer.lr)))
        make_trainable(discriminator, True)
        caunt = 0
        print("enter_for_loop :: " + "g_loss: " + str(g_loss) + "d_loss: " + str(0.5 * np.add(d_loss_r[0],d_loss_f[0])))
        if  0.5 * np.add(d_loss_r[0],d_loss_f[0]) > 0.5:
            n=20
        else:
            n = 10
        print("n=>"+str(n))
        while caunt < n:

            # generate vectors
            vec_batch = training[random.sample(range(0,training.shape[0]),BATCH_SIZE), :]
            noise_gen = np.random.normal(mu, sigma, size=[BATCH_SIZE, 1000])
            generated_vector = generator.predict(noise_gen)

            # Train discriminator on samples vector
            X = vec_batch
            y = np.zeros([BATCH_SIZE, 1])
            #y[0:BATCH_SIZE] = np.random.uniform(0.9, 1.1) # smoothing the lables
            if (e % 25) == 0 :
                y[0:BATCH_SIZE] = np.random.uniform(-0.2, 0.2)
            else:
                y[0:BATCH_SIZE] = 1
            d_loss_r = discriminator.train_on_batch(X, y)

            # Train discriminator on generated vector
            X = generated_vector
            y = np.zeros([BATCH_SIZE, 1])
            y[0:BATCH_SIZE] = 0
            d_loss_f = discriminator.train_on_batch(X, y)
            # print("loss: " + str(d_loss))
            caunt += 1
        losses["d"].append([0.5 * np.add(d_loss_r[0],d_loss_f[0]),0.5 * np.add(d_loss_r[1],d_loss_f[1])])
        print("first_loop :: " + "g_loss: " + str(g_loss) + "d_loss: " + str(0.5 * np.add(d_loss_r[0],d_loss_f[0])))
        make_trainable(discriminator, False)
        caunt = 0
        if g_loss > 0.5:
             n=20
        else:
             n=10
        print("n=>"+str(n))

        while caunt < n:
            # train Generator-Discriminator stack on input noise
            noise_tr = np.random.normal(mu,sigma,size=[2*BATCH_SIZE,1000])
            y2 = np.zeros([2*BATCH_SIZE, 1])
            y2[:] = 1
            g_loss = GAN.train_on_batch(noise_tr, y2)
            caunt += 1

        # stroring the losses
        losses["g"].append(g_loss)
        print("second_loop :: " + "g_loss: " + str(g_loss) + "d_loss: " + str(0.5 * np.add(d_loss_r[0],d_loss_f[0])))
        if (e % 2) == 0:
            noise = np.random.normal(mu,sigma, size=[10, 1000])
            pred = generator.predict(noise)
            decoded_pred = D.predict(pred)
            converted_pred = convert_classes_to_numbers(decoded_pred)
            translated_pred = translate_from_to_smiles(converted_pred, key="dic", dic=dic)
        print(translated_pred)
        #translated_pred = pd.DataFrame(translated_pred)
        if (e % 500) == 0:   
            noise = np.random.normal(mu,sigma, size=[5000, 1000])
            pred = generator.predict(noise)
            decoded_pred = D.predict(pred)
            converted_pred = convert_classes_to_numbers(decoded_pred)
            translated_pred = translate_from_to_smiles(converted_pred, key="dic", dic=dic)
            print(translated_pred)
            translated_pred = pd.DataFrame(translated_pred)
            with open("Latentspace_LSTMGAN_generated_molecules_"+str(name)+".csv", 'a') as f:
                translated_pred.to_csv(f, header=False)
        # saving the models
        GAN.save("latent_space_GAN_" + str(name) + ".hdf5")
        discriminator.save("latent_space_discriminator_" + str(name) + ".hdf5")
        generator.save("latent_space_generator_" + str(name) + ".hdf5")
        # saving the losses
        with open("latent_space_GAN_losses_"+str(name)+".pickle", "wb")as f:
            pickle.dump(losses, f)

    # plot results
    accuracy = [x[1] for x in losses["d"]]
    loss = [x[0] for x in losses["d"]]
    plt.figure(figsize=(15, 20))
    plt.subplot(2, 2, 1)
    plt.title("Generative loss")
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.subplot(2, 2, 2)
    plt.title("Discriminative loss")
    plt.plot(loss, label='discriminitive loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.subplot(2, 2, 3)
    plt.title("Accuracy")
    plt.plot(accuracy, label='accuracy')
    plt.legend()
    plt.xlabel("epochs")
    plt.savefig("AE_LSTM_All_together_" + str(name) + ".png")



def main():

    df_file = sys.argv[1]
    # load dataset and model
    dic_file = sys.argv[sys.argv.index('--dic_file') + 1]
    nb_epochs = int(sys.argv[sys.argv.index('--nb_epochs') + 1])
    title = sys.argv[sys.argv.index('--title') + 1]
    batchsize = int(sys.argv[sys.argv.index('--batchsize') + 1])
    pret_n_epochs = int(sys.argv[sys.argv.index("--pret_ep")+1])
    Gi = sys.argv[sys.argv.index('--G') + 1]
    Di = sys.argv[sys.argv.index('--D') + 1]
    lr0 = float(sys.argv[sys.argv.index('--lr0') + 1])
    lr1 = float(sys.argv[sys.argv.index('--lr1') + 1])
    train = int(sys.argv[sys.argv.index('--train') + 1])

    print("#" * 50)
    print(" " * 20 + "Loading Autoencoder")
    print("#" * 50)
    AE = keras.models.load_model("model_Autoencoder_linear_original.best1.hdf5")
    AE.summary()
    E = Sequential()
    for i in AE.layers[0:5]:
        E.add(i)
    opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, clipvalue=0.04, decay=0)
    E.compile(optimizer=opt, loss='binary_crossentropy')
    D = Sequential()
    D.add(keras.layers.InputLayer(input_shape = (40,1)))
    for i in AE.layers[5:]:
        D.add(i)
    D.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])

    # #Loading the model for the calculaton of the FID
    # print("#" * 50)
    # print(" " * 20 + "Loading the model for the calculaton of the FID")
    # print("#" * 50)
    # net = keras.models.load_model("tox21_FID_NET.h5", custom_objects={
    # 'masked_loss_function': build_masked_loss(K.binary_crossentropy, -1)})
    # FID_model = keras.models.Sequential()
    # FID_model.add(keras.layers.InputLayer(input_shape=(2048,)))
    # for i in net.layers[1:4]:
    #    FID_model.add(i)

    # FID_model.layers[-1].activation = None
    # FID_model.compile(optimizer="adam", loss='binary_crossentropy')

    #loading data, dictionary and
    print("#" * 50)
    print(" " * 20 + "Loading data")
    print("#" * 50)
    print(df_file)
    df = pd.read_csv(df_file, index_col=0, nrows=200000)
    dic = load_dic_form_csv(dic_file,key_col=1)
    print(dic)
    if train:
        data_frame = E.predict(df.values)

    print("#" * 50)
    print(" " * 20 + "Loading generator discriminator and GAN")
    print("#" * 50)
    
    generator, discriminator, GAN = load_GAN_from_G_and_D(Gi, Di,lr0,lr1)

    print("#" * 50)
    print(" " * 20 + "pretraining D")
    print("#" * 50)
    if train:
        loss = pre_train_discriminator(data_frame, discriminator, generator,ntrain=batchsize,E=E,pre_num_epochs = pret_n_epochs)
        print("#" * 50)
        print(" " * 20 + "Train GAN")
        print("#" * 50)
        train_for_n(data_frame, generator, discriminator, GAN, nb_epoch=nb_epochs, BATCH_SIZE=batchsize, title=title,
                name=title,E=E, dic=dic, D=D)

if __name__ == "__main__":
    main()

#CUDA_VISIBLE_DEVICES=4 python3.5 Test_Final_Program X_Chemble_40_15.csv --dic_file DICTIONARY_l2n.csv --lr0 0.001 --lr1 0.001 --g_hidden 0 --d_hidden 6 --nb_epochs 100 --title test --batchsize 5000 --pret_ep 3
