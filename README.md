# GANs for molecular fingerprints and SMILES string generation.

Canonical SMILES strings were retrieved from the ChEMBL data-set through the RDkit package and python and molecular fingerprints were derived from them. A
GAN was implemented using Keras and trained using molecular fingerprints. This model was called Chemo-GAN. Both generator and discriminator were implemented as fully connected deep neural networks.
A second model was implemented using an autoencoder to map SMILES strings from and to a latent space and a GAN trained to generate this latent space representation of SMILES strings. The aim of this second approach was to obtain a generator able to generate latent space representations of SMILES and a decoder able to map them back to SMILES strings. This model was called Latent-Space-GAN. To evaluate the distance between the distributions of original data and generated ones and to assess samples quality, a new distance measure, which was called FTOXD, was designed and calculated.

### Fréchet Tox21 Distance

To evaluate the Chemo-GAN  and the Latent-Space-GAN, the Fréchet Tox21 Distance (FTOXD) was designed. This metric is similar to the FID but instead of using the inception model to generate the conditional probability p(y\|G(z)), it uses a model trained on the Tox21 data-set, which was called Tox21-FTOXD model. 
The Tox21 data-set, which consists of 12,000 training data and 647 test ones, was used. From the chemical structures contained in this data-set, the equivalent of the ECFP4 molecular fingerprints was calculated for each compound through the rdkit package in python. 
The Tox21-FTOXD model was designed as a 3 layer fully connected neural network. 1,024  units were used for each hidden layer and 12 outputs corresponding to the different labels were used. In each hidden layer, the selu activation function with the Lecun weight initialization was used. Many labels provided in the Tox-21 data-set were missing. Therefore, missing values were masked during the training. The binary cross-entropy  was used as loss function. This model obtained an AUC of 0.74 on the test set.
Generated fingerprints were fed in this model and the outputs were extracted from its second hidden layer to have chemical relevant features. Molecular fingerprints derived from SMILES strings belonging to the test set of the ChEMBL  data-set were used to calculate the conditional distribution p(y\|x(data)), while the conditional distribution p(y\|G(z))  was calculated with the generated molecular fingerprints.

![FTOXD formula](https://latex.codecogs.com/gif.latex?%5Crm%7BFTOXD%7D%20%3D%20d%5E2%28%28%5Ctextbf%7Bm%7D%2C%5Ctextbf%7BC%7D%29%2C%28%5Ctextbf%7Bm%7D_%7Bw%7D%2C%5Ctextbf%7BC%7D_%7Bw%7D%29%29%3D%20%5C%7C%20%5Ctextbf%7Bm%7D-%5Ctextbf%7Bm%7D_%7Bw%7D%20%5C%7C_%7B2%7D%5E%7B2%7D%20&plus;%20Tr%28%5Ctextbf%7BC%7D&plus;%5Ctextbf%7BC%7D_%7Bw%7D%2C-2%28%5Ctextbf%7BC%7D%5Ctextbf%7BC%7D_%7Bw%7D%29%5E%5Cfrac%7B1%7D%7B2%7D%29)

The above formula was used to calculate the FTOXD using the means and co-variance matrices derived from the distributions obtained using the Tox21-FTOXD model.



