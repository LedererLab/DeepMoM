# DeepMoM

This repository provides the implementations of the methods described in [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035).

## Estimator

We provide an example code in `TrainNN.R` for computing DeepMoM estimator with Relu activation. Developed for `R 4.1.2`.


## Simulations

We provide an example code in `SimulationStudy.R` for a comparison of least-squars, Huber, and least absolute deviation estimators with ReLU based DeepMoM estimators for regression type problems and another code in `Classifier.R` for comparison of soft-max cross entropy estimators and ReLU based DeepMoM estimators for classification tasks. 

## Applications

We provide an example code in `TCGA.R` to apply the DeepMoM structure on seven TCGA data sets.

The data is available in `TCGAlist.RData`.

## Repository authors 

* Shih-Ting Huang, Ph.D. student in Mathematical Statistics, Ruhr-University Bochum

* Johannes Lederer, Professor in Mathematical Statistics, Ruhr-University Bochum

## Other files

**BackPropN.R** : Compute the derivative of neural network with respect to each weight matrix by back propagation.

**FeedForwardNN.R** : Compute the output of the neural network by feed forwardd method of the neural network.

**l1_derivative.R** : Compute the derivative of L1 loss function.

**huber_derivative.R** : Compute the derivative of Huber loss     function.

**huber_loss.R** : Compute the value of Huber loss function.

**mom.R** : Find the block of data that achieve the median of empirical means. 

**Group.R** : Partition the data into some groups.

## Supported languages and platforms

All of the codes in this repository are written in R and supports all plarforms which are supported by R itself.

## Dependencies

This repository does not depend on any R libraries or external sources.

## Licensing

All codes are licensed under the MIT license. To
view the MIT license please consult `LICENSE.txt`.

## References
 [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035)
 
 Cite as "S. Huang and J. Lederer. DeepMoM: Robust Deep Learning With Median-of-Means.".

