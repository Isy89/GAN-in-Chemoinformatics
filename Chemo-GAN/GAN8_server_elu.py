import numpy as np
import scipy as sp
np.random.seed(42)
from keras import optimizers
from keras import backend as K
import keras
from keras.layers import Input,merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.noise import GaussianNoise
import os
import pandas as pd
import random
import tqdm
import sys
import pickle
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD,Adam, RMSprop
import tensorflow as tf
import argparse
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
#session = tf.Session(config = config)
import collections

#"""This programm can be executed calling it from command line like:
# pytho GAN2.py file epochs caunt generator_learning_rate discriminator_learning_rate generator_hidden_layers
#  discriminator_hidden_layers figure_name --batchsizes [list of batchsizes]
#
# example :
# python GAN6.py X_Chemble_fingerprints.csv 1000 153 1e-04 1e-03 2 2 1000_153_1e-04_1e-03_2 --batchsizes 5000
#  """

##############################################################################################################
#                         FUNCTIONS
##############################################################################################################

def predictions_threshold(df, threshold, below=0, above=1):
    mask = df > threshold
    mask1 = df <= threshold
    df[mask] = above
    df[mask1] = below
    return df

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

def build_masked_metric(metrics, mask_value):
    """Builds a loss function that masks based on targets
    Args:
        metrics: The metrics function to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a metrics function that acts like loss_function with masked inputs
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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Params:
    -- mu1 : Numpy array containing the means of the values of the second layer of the tox21 model
             obtained using the generated molecular fingerprints
    -- mu2   : The sample mean over activations of the second layer of the tox21 model, precalcualted
               on an representative data set.
    -- sigma1: The covariance matrix over the values of the second layer of the tox21 model,
             obtained using the generated molecular fingerprints
    -- sigma2: The covariance matrix over the values of the second layer of the tox21 model,
               precalcualted on an representative data set.
    Returns:
    -- FTOXD  : The Frechet Distance: ||mu_1 - mu_2||^2 Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

    """
    m = np.square(mu1 - mu2).sum()
    s = sp.linalg.sqrtm(np.dot(sigma1, sigma2))
    dist = m + np.trace(sigma1+sigma2 - 2*s)
    return dist

def calculate_FTOXD_generator(mu1,sigma1,generator, model):
    noise_gen = np.random.normal(0, 1, size=[10000, 2048])
    pred = generator.predict(noise_gen)
    pred = predictions_threshold(pred, 0.5)
    pred = model.predict(pred)
    mu2 = np.mean(pred, axis=0)
    sigma2 = np.cov(pred, rowvar=False)
    return  calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def tanimoto(v1, v2):
    """
    Calculates tanimoto similarity for two bit vectors*
    """
    mask1 = v1 > 0.5
    mask2 = v1 <= 0.5
    v1[mask1] = 1
    v1[mask2] = 0
    mask1 = v2 > 0.5
    mask2 = v2 <= 0.5
    v2[mask1] = 1
    v2[mask2] = 0
    #print(v1)
    #print(v2)
    sums = np.array([v1.sum(),v2.sum()])
    v1 = np.asarray(v1,dtype = "bool")
    v2 = np.asarray(v2,dtype = "bool")
    v3 = np.array([(np.bitwise_and(v1, v2).sum() / np.bitwise_or(v1, v2).sum())])
    return v3,sums



# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def plot_loss(losses,title):

    """ generate the plot of the losses """
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.title(title)
    accuracy = [x[1] for x in losses["d"]]
    loss = [x[0] for x in losses["d"]]
    plt.plot(loss, label= 'discriminitive loss')
    plt.plot(accuracy , label='accuracy')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.savefig("Losses and accuracy"+ title + ".png")
    #plt.show()

def create_model(dropout = 0.5,nb_epochs=1000,lr1=0.001,lr0 = 0.0001,g_hidden = 10, d_hidden = 10 ,activation="selu"):
    """function to create the models
       return generator, discriminator and GAN"""
    if activation == "selu":
        kern = "lecun_normal"
    elif activation == "elu":
        kern = "he_normal"
    elif activation == "relu":
        kern = "glorot_normal"
    else:
        kern = "glorot_normal"
    # Build Generative model ...
    g_input = Input(shape=[2048])
    H = Dense(500,activation="selu", kernel_initializer='lecun_normal')(g_input)
    #H = LeakyReLU(alpha=0.3)(H)
    H = Dropout(dropout)(H)
    H = GaussianNoise(0.3)(H)
    for i in range(g_hidden):
        H = Dense(500,activation="selu", kernel_initializer='lecun_normal')(H)
        #H = LeakyReLU(alpha=0.3)(H)
        H = Dropout(dropout)(H)
        H = GaussianNoise(0.3)(H)
    g_V = Dense(2048,activation="sigmoid", kernel_initializer='glorot_normal')(H)
    generator = Model(g_input, g_V)
    opt = optimizers.SGD(lr=lr0)
    #opt = optimizers.RMSprop(lr=lr0)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()
    # Build discriminative model ...

    d_input = Input(shape=[2048])
    H = Dense(2000,activation="selu", kernel_initializer='lecun_normal')(d_input)
    H = Dropout(dropout)(H)
    #H = GaussianNoise(0.3)(H)

    for i in range(d_hidden):
        H = Dense(2000,activation="selu", kernel_initializer='lecun_normal')(H)
        H = Dropout(dropout)(H)

    d_V = Dense(1, activation='sigmoid')(H)
    discriminator = Model(d_input, d_V)
    opt1= optimizers.SGD(lr=lr1)
    #opt1 = optimizers.RMSprop(lr=lr1)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt1 , metrics=["acc"])
    discriminator.summary()
    # Build GAN
    gan_input = Input(shape=[2048])
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='binary_crossentropy', optimizer=opt1)
    GAN.summary()
    return generator, discriminator, GAN


def pre_train_discriminator(dataframe, discriminator, generator, ntrain=10000, nep = 2):
    training = dataframe
    mu, sigma = 0, 1  # mean and standard deviation
    trainidx = random.sample(range(0, training.shape[0]), ntrain)
    TD = training[trainidx, :]
    # Pre-train the discriminator network ...
    # generate random samples coming from a standard uniform distribution
    noise_gen = np.random.normal(mu, sigma, size=[TD.shape[0], 2048])
    # forward pass through the generator to generate the sample sampled from the generator
    generated_vector = generator.predict(noise_gen)
    # first samples coming from the generator and those one coming from the training data concateanted
    X = np.concatenate((TD,generated_vector))
    n = TD.shape[0]
    y = np.zeros([2 * n, 1])
    y[:n] = 1
    make_trainable(discriminator, True)
    d_loss = discriminator.fit(X, y, epochs=nep, batch_size=10000)
    return d_loss.history['loss'][0]

def train_GAN(dataframe,data_frame_for_FID, generator, discriminator, GAN, nb_epoch=5000, BATCH_SIZE=32,name = 7,eval_model="net"):
    tans = np.array([])
    losses = {"d": [], "g": []}
    # distances = []
    distances_container = []
    count_tans = 0
    training = dataframe
    print("Calculating mu1 and sigma1")
    FIDDF = eval_model.predict(data_frame_for_FID)
    mu1 = np.mean(FIDDF, axis=0)
    sigma1 = np.cov(FIDDF, rowvar=False)
    mu, sigma = 0, 1  # mean and standard deviation
    # generate vectors
    print("Generating the fake data and sampling the real one")
    vec_batch = training[random.sample(range(0,training.shape[0]),BATCH_SIZE), :]
    noise_gen = np.random.normal(mu, sigma, size=[BATCH_SIZE, 2048])
    generated_vector = generator.predict(noise_gen)
    # Train discriminator on real data
    print("Training discriminator on real samples")
    make_trainable(discriminator, True)
    X = vec_batch
    y = np.zeros([BATCH_SIZE, 1])
    y[0:BATCH_SIZE] = 1
    d_loss = discriminator.train_on_batch(X, y)
    # print(d_loss)
    # Train discriminator on generated vector
    print("Training discriminator on fake samples")
    X = generated_vector
    y = np.zeros([BATCH_SIZE, 1])
    y[0:BATCH_SIZE] = 0
    d_loss = discriminator.train_on_batch(X, y)
    # train Generator-Discriminator stack on input noise
    print("Training generator")
    make_trainable(discriminator, False)
    noise_gen = np.random.normal(mu, sigma, size=[BATCH_SIZE, 2048])
    y2 = np.zeros([BATCH_SIZE, 1])
    y2[:] = 1
    g_loss = GAN.train_on_batch(noise_gen, y2)
    #print(g_loss)
    for e in tqdm.tqdm(range(nb_epoch)):
        #print("entering for loop")
        make_trainable(discriminator, True)
        #print("enter_for_loop :: " + "g_loss: " + str(g_loss) + "d_loss: " + str(d_loss))
        # I set caunt for the training of the generator and discriminator to train them with different times
        caunt = 0
        n = 1
        while caunt < n:
            # generate vectors
            #print("training discriminator")
            vec_batch = training[random.sample(range(0,training.shape[0]),BATCH_SIZE), :]
            noise_gen = np.random.normal(mu, sigma, size=[BATCH_SIZE, 2048])
            generated_vector = generator.predict(noise_gen)
            #print(generated_vector[0:5,0:20])
            # Train discriminator on samples vector
            print("Training D on real data")
            X = vec_batch
            #y = np.zeros([BATCH_SIZE, 1])
            y[0:BATCH_SIZE] = 1# np.random.uniform(0.9, 1.1) # smoothing the lables
            #y[0:BATCH_SIZE] = 1
            d_loss_r = discriminator.train_on_batch(X, y)
            # Train discriminator on generated vector
            print("Training D on generated data")
            X = generated_vector
            y = np.zeros([BATCH_SIZE, 1])
            y[0:BATCH_SIZE] = 0
            d_loss_f = discriminator.train_on_batch(X, y)
            caunt +=1
        losses["d"].append(0.5*np.add(d_loss_r,d_loss_f))
        print("TRAINING D INTERMEDIATE RESULTS :: " + " g_loss: " + str(g_loss) + " d_loss: " + str(0.5*np.add(d_loss_r[0],d_loss_f[0])) + " binary accuracy: " + str(0.5*np.add(d_loss_r[1],d_loss_f[1])))
        make_trainable(discriminator, False)
        caunt = 0
        n = 1
        # I set caunt for the training of the generator and discriminator to train them with different times
        while caunt < n:
            # train Generator-Discriminator model on Gaussian noise
            print("Training generator")
            noise_tr = np.random.normal(mu, sigma, size=[BATCH_SIZE*2, 2048])
            y2 = np.zeros([BATCH_SIZE*2, 1])
            y2[:] = 1
            g_loss = GAN.train_on_batch(noise_tr, y2)
            caunt += 1
        #Calculating the Freche tox21 distance (FTOXD) each 500 updates of the model
        #to check the the distance between the distribution

        print("Calculating Fréche distance")
        if (e % 2) == 0:
            batch_d=[]
            print("calculating FTOXD for 50")
            for i in range(1): # set to 500
                d = calculate_FTOXD_generator(mu1,sigma1,generator,eval_model)
                batch_d.append(d)
            #print("calculating distance mean and SD")
            distance_mean = np.mean(batch_d)
            #distance_std  = np.std(batch_d)
            #distances.append([distance_mean,distance_std])
            distances_container.append(batch_d)
            print(batch_d)
            print(" "*50)
            print("*"*50)
            print("Fréche distance =>" + str(distance_mean))
            print("*"*50)
            print(" "*50)
        #with open("temp_F_distances_GAN_fingerprints.pickle", 'wb') as f:
        #      pickle.dump(distances, f)
        #stroring the losses
        losses["g"].append(g_loss)
        print("TRAINING G INTERMEDIATE RESULTS :: " + " g_loss: " + str(g_loss) + " d_loss: " + str(0.5*np.add(d_loss_r[0],d_loss_f[0] ))+ " binary accuracy: " + str(0.5*np.add(d_loss_r[1],d_loss_f[1])))
        if (e % 1) == 0: #put to 500 when running for this calculation
            print("Calculating Tanimoto distance")
            count_tans+=1
            for i in range(100):
                noise1 = np.random.normal(mu, sigma, size=[1, 2048])
                noise2 = np.random.normal(mu, sigma, size=[1, 2048])
                pred1 = generator.predict(noise1)
                pred2 = generator.predict(noise2)
                tan, sum_val=tanimoto(pred1, pred2)
                tans = np.concatenate((tans, tan),axis = 0)
            with open("Tanimoto_coefficient"+str(name)+".pickle", 'wb') as f:
               pickle.dump(tans, f)
    #with open("F_distances_GAN_fingerprints_elu.pickle", 'wb') as f:
    #           pickle.dump(distances, f)
    print("Saving distances to => F_distances_container_GAN_fingerprints_elu.pickle")
    with open("FTOXD_"+str(name)+".pickle", 'wb') as f:
               pickle.dump(distances_container, f)
    # saving the models
    GAN.save("GAN_"+str(name)+".hdf5")
    discriminator.save("discriminator_"+str(name)+".hdf5")
    generator.save("generator_"+str(name)+".hdf5")
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.title("FTOXD fingerprints GAN " + str(name))
    # plt.xlabel("Updates")
    # plt.ylabel("FTOXD")
    # errors = [x[1] for x in distances]
    # dist   = [x[0] for x in distances]
    # plt.errorbar([x for x in range(1,len(errors)+1)],dist,errors)
    # plt.savefig("FTOXD_fingerprints_GAN " + str(name) + ".png")
    # plt.figure()
    # plt.title("tanimoto distance"+str(name))
    # plt.xlabel("epochs")
    # plt.ylabel("tanimoto dis")
    # plt.plot(tans)
    # plt.savefig("taanimotos_fingerprints_GAN " +str(name)+ ".png")
    # saving the figure
    # plot_loss(losses, title)
    # plt.figure()
    # plt.hist(tans)
    # plt.xlabel("Tanimoto distance distribution")
    # plt.title("Tanimoto Distribution"+str(name))
    # plt.savefig("tanimotos_dis_fingerprints_GAN " +str(name)+ ".png")
    with open("losses_GAN.pickle","wb")as f:
        pickle.dump(losses,f)
    accuracy = [x[1] for x in losses["d"]]
    loss = [x[0] for x in losses["d"]]
    print("Plotting model performances")
    plt.figure(figsize=(15, 20))
    plt.subplot(3, 2, 1)
    plt.title("Generative loss", fontsize=35)
    plt.plot(losses["g"], label='generative loss')
    plt.grid()
    plt.legend()
    plt.xlabel("updates", fontsize=25)
    plt.subplot(3, 2, 2)
    plt.title("Discriminative loss", fontsize=35)
    plt.plot(loss, label='discriminative loss')
    plt.grid()
    plt.legend()
    plt.xlabel("updates", fontsize=25)
    plt.subplot(3, 2, 3)
    plt.title("Accuracy", fontsize=35)
    plt.plot(accuracy, label='accuracy')
    plt.grid()
    plt.legend()
    plt.xlabel("updates", fontsize=25)
    plt.subplot(3, 2, 4)
    plt.title("Tanimoto coefficient", fontsize=35)
    plt.xlabel("updates", fontsize=25)
    plt.ylabel("Tanimoto coefficient", fontsize=25)
    tans = pd.DataFrame(np.array(tans).reshape(count_tans, 100))
    pd.DataFrame.boxplot(tans.T)
    plt.xticks([x for x in range(0, count_tans +1, 1)], [x for x in range(0, nb_epoch+1000, 500)])
    plt.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.show()
    plt.savefig("all_tog.pdf", format="pdf", bbox_inches='tight')
    os.system("all_tog.pdf")
    return

def main():

    parser = argparse.ArgumentParser(description="""
    Program to train a Chemo-GAN. It returns the learning curves, the Tanimoto coefficient, the binary accuracy and the 
    FTOXD. It prvides also a plot containing the learning curves, the Tanimoto coefficient and the binary accuracy.
    Example how to run the program:
    python GAN8_server_elu.py TARGETS_382688_2048.csv GAN_test_program_ 0.001 0.001 1 2 3 1 selu 10000""")
    parser.add_argument("file", help="Insert the molecular fingerprints csv file having a dimensionality of n moles * 2048 ")
    parser.add_argument("name", help="Name.")
    #parser.add_argument("title", help="Title used to save the results and the plots.")
    parser.add_argument("lr0",type=float, help="Learning rate for the GAN.")
    parser.add_argument("lr1",type=float, help="Learning rate for the discriminator.")
    parser.add_argument("g_hidden",type=int, help="Number of hidden layers in the generator.")
    parser.add_argument("d_hidden",type=int, help="Number of hidden layers in the discriminator.")
    parser.add_argument("nb_epochs",type=int, help="Number of epochs for the training.")
    parser.add_argument("pret_ep",type=int, help="Number of epochs for the pre-training of the discriminator.")
    parser.add_argument("act", help="Activation function used for the hidden layers.")
    parser.add_argument("batchsize",type=int, help="Batch size used.")
    args = parser.parse_args()
    print(args)
    file = args.file
    print(file)
    # caunt is used as number to differenciate the models and figures generated
    #learning rates
    #title = args.title
    lr0 = args.lr0
    lr1 = args.lr1
    #number of hidden layers in the generator and discriminator
    g_hidden = args.g_hidden
    d_hidden = args.d_hidden
    nb_epochs = args.nb_epochs
    pret_n_epochs = args.pret_ep
    #name of the figure
    name = args.name
    batchsize = args.batchsize
    activation = args.act



#########################################################################################################################
    # file = sys.argv[1]
    # print(file)
    # # caunt is used as number to differenciate the models and figures generated
    # #learning rates
    # title = sys.argv[sys.argv.index('--title') + 1]
    # lr0 = float(sys.argv[sys.argv.index('--lr0') + 1])
    # lr1 = float(sys.argv[sys.argv.index('--lr1') + 1])
    # #number of hidden layers in the generator and discriminator
    # g_hidden = int(sys.argv[sys.argv.index('--g_hidden') + 1])
    # d_hidden = int(sys.argv[sys.argv.index('--d_hidden') + 1])
    # nb_epochs = int(sys.argv[sys.argv.index('--nb_epochs') + 1])
    # pret_n_epochs = int(sys.argv[sys.argv.index("--pret_ep")+1])
    # #name of the figure
    # name = sys.argv[sys.argv.index('--name') + 1]
    # batchsize = int(sys.argv[sys.argv.index('--batchsize') + 1])
    # activation = sys.argv[sys.argv.index('--act') + 1]
    #test = 0
    #if test != 0:
    #    file = "TARGETS_382688_2048.csv"
    #    title = "Test_GAN_fingerprints"
    #    lr0 = 0.001
    #    lr1 = 0.001
    #    # number of hidden layers in the generator and discriminator
    #    g_hidden = 1
    #    d_hidden = 5
    #    nb_epochs = 2
    #    pret_n_epochs = 2
    #    # name of the figure
    #    name = "test"
    #    batchsize = 1000

    #Loading the model for the calculaton of the FID
    print("#" * 50)
    print(" " * 20 + "Loading the model for the calculaton of the FTOXD")
    print("#" * 50)
    net = keras.models.load_model("tox21_FID_NET.h5", custom_objects={
    'masked_loss_function': build_masked_loss(K.binary_crossentropy, -1), "masked_metric_function": build_masked_metric(keras.metrics.binary_accuracy,-1)})
    FID_model = keras.models.Sequential()
    FID_model.add(keras.layers.InputLayer(input_shape=(2048,)))
    for i in net.layers[1:4]:
       FID_model.add(i)

    FID_model.layers[-1].activation = None
    FID_model.compile(optimizer="adam", loss='binary_crossentropy')

    #creating and saving the models
    print("#" * 50)
    print(" " * 20 + "creating and saving the models")
    print("#" * 50)
    generator, discriminator, GAN = create_model(nb_epochs=nb_epochs, lr0=lr0, lr1=lr1, g_hidden=g_hidden,d_hidden=d_hidden,activation=activation)

    #loading the dataframe
    print("#"*50)
    print(" "*20+"loading the data")
    print("#" * 50)
    data_frame = pd.read_csv(file,index_col = 0,nrows = 200000)
    data_frame_for_FID = pd.read_csv(file,index_col = 0,nrows = 100000,skiprows  = 200000)
    data_frame = data_frame.values.astype(float)
    data_frame_for_FID=data_frame_for_FID.values.astype(float)
    data_frame[data_frame==1] =0.8 #np.random.uniform(0.8, 1)
    data_frame[data_frame==0] =0.2 #np.random.uniform(0, 0.2)
    #pretraing the discriminator
    print("#" * 50)
    print(" " * 20 + "pretraing the discriminator")
    print("#" * 50)
    loss = pre_train_discriminator(data_frame, discriminator, generator,nep=pret_n_epochs)

    #strarting training
    print("#" * 50)
    print(" " * 20 + "starting training")
    print("#" * 50)
    train_GAN(data_frame, data_frame_for_FID, generator, discriminator, GAN, nb_epoch=nb_epochs, BATCH_SIZE=batchsize,name = name, eval_model = FID_model)

if __name__ == "__main__":
    main()
