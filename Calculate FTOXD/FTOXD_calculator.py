import pandas as pd
import numpy as np
import keras
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
import scipy as sp
import argparse
np.random.seed(42)
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# session = tf.Session(config = config)
import collections
from keras import backend as K
lg = RDLogger.logger()
lg.setLevel(3)


############################################################################
#                                 FUNCTIONS
############################################################################
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


def predictions_threshold(df, threshold, below=0, above=1):
    """Function to replace numbers lower than the given threshold 
    with the below parameter and thos above the given threshold
     with the above parameter
     Params:
     -- df : pandas dataframe containing generated smiles
     -- threshold 
     -- below : number with which the values below the given threshold will be replaced
     -- above : number with which the values above the given threshold will be replaced """
    mask = df > threshold
    mask1 = df <= threshold
    df[mask] = above
    df[mask1] = below
    return df


#######################################################################################


def eval_SMILE_0_1(smile):
    # print(smile)
    # print(type(smile))
    if Chem.MolFromSmiles(smile):
        return 1
    else:
        return 0


def eval_valid_invalid(X2):
    """ Function to evaluate the percentage of valid generated molecules.
    Params:
    --df : pandas dataframe containing the generated smiles
    Returns:
    -- the percentage of generated molecules    """
    evaluation = [eval_SMILE_0_1(smile) for smile in X2]
    valid = sum(evaluation)
    try:
        ratio = valid / len(evaluation)
    except:
        return 0
    return ratio


def modified_FID(mu1, sigma1, df, model, epsilon=1e-13, verbose=False):
    X2 = df
    # print(X2)
    if verbose:
        print("step1")
    FID = calculate_FID_df(mu1, sigma1, model=model, X2=X2)
    if verbose:
        print("step2")
    #p = eval_valid_invalid(X2=X2)
    if verbose:
        print("FInished")
        print(p)
    return FID  # (1/(p + epsilon)) * FID


def calculate_FID_df(mu1, sigma1, model, X2):
    #smiles = [Chem.MolFromSmiles(smile) for smile in X2]
    #smiles = [x for x in smiles if x != None]
    fingerprints = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x[0]), 2, 2048) for x in X2.values if
         str(type(Chem.MolFromSmiles(x[0]))) == "<class 'rdkit.Chem.rdchem.Mol'>"])
    if fingerprints.shape[0]>0:
        #fingerprints = [[y for y in x] for x in fingerprints]
        pred = model.predict(fingerprints)
        mu2 = np.mean(pred, axis=0)
        sigma2 = np.cov(pred, rowvar=False)
        print(mu1,mu2)
        print(sigma1,sigma2)
        return np.real(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))
    return 10 ** 13

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    -- FID  : The Frechet Distance.
    -- mean : The squared norm of the difference of the means: ||mu_1 - mu_2||^2
    -- trace: The trace-part of the FID: Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    m = np.square(mu1 - mu2).sum()
    s = sp.linalg.sqrtm(np.dot(sigma1, sigma2))
    dist = m + np.trace(sigma1 + sigma2 - 2 * s)
    return dist

######################################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("SMILES_ORIGINAL_DATASET",
                        help="csv file containing SMILES belonging to the original dataset")
    parser.add_argument("SMILES_GENERATED_DATASET", help="csv file of the generated SMILES")
    args = parser.parse_args()
    ############################################################################
    #   loading the model for the FID
    ############################################################################
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
    FID_model.summary()
    ############################################################################
    print("#" * 50)
    print(" " * 20 + "Loading original data")
    print("#" * 50)
    SMILES_ORIGINAL_DATASET = pd.read_csv(args.SMILES_ORIGINAL_DATASET,index_col=0,nrows=30)
    print("#" * 50)
    print(" " * 20 + "Loading generated data")
    print("#" * 50)
    SMILES_GENERATED_DATASET = pd.read_csv(args.SMILES_GENERATED_DATASET,index_col=0,nrows=30)
    o_df = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x[0]), 2, 2048) for x in SMILES_ORIGINAL_DATASET.values if
                            str(type(Chem.MolFromSmiles(x[0]))) == "<class 'rdkit.Chem.rdchem.Mol'>"])
    o_df_pred = FID_model.predict(o_df)
    mu1 = np.mean(o_df_pred, axis=0)
    sigma1 = np.cov(o_df_pred, rowvar=False)
    print("#" * 50)
    print(" " * 20 + "Calculating FTOXD")
    print("#" * 50)
    FTOXD = modified_FID(mu1, sigma1, SMILES_GENERATED_DATASET, FID_model, epsilon = 1e-13)
    print("FTOXD = "+str(FTOXD))
if __name__ == "__main__":
    main()