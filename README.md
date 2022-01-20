# DeepMoM

This repository provides the implementations of the methods described in [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035).

## Estimator

We provide an example code in `MOM_ estimator.Rmd` for computing median of means estimator for Relu networks. Developed for `R 4.1.2`.


## Simulations

We provide an example code in `SimulationStudy.Rmd` for a comparison of least squared estimator and median of means estimator for Relu networks. Developed for `R 4.1.2`.

## Repository authors 

* Shih-Ting Huang, Ph.D. student in Mathematical Statistics, Ruhr-University Bochum

* Johannes Lederer, Professor in Mathematical Statistics, Ruhr-University Bochum

## Other files

**Backpropagation** : Applying the back propagation with respect to squared loss for Relu network.

**BackL1** : Applying the back propagation with respect to L1 loss for Relu network.

**BackHuber** : Applying the back propagation with respect to Huber loss for 
Relu network.

**DataGeneration** : Generating sample data.

**Feedforward** : Applying the feed forward computation for a given neural network.

**Loss** : The computation of sum of squared loss for a given neural network.

**MOM** : Implementing the median of means principle.

**Relu** : The Relu activate function.

**ReluDerivative** : The derivative of Relu activate function.

**L1_erivative** : The derivative of L1 loss.

**Huber_erivative** : The derivative of Huber loss.

**my_nn** : A list represents a neural network.

**CVmom** : Selecting the number of blocks for applying median of means principle by cross validation.

## Supported languages and platforms

All of the codes in this repository are written in R and supports all plarforms which are supported by R itself.

## Dependencies

This repository does not depend on any R libraries or external sources.

## Licensing

The HDIM package is licensed under the MIT license. To
view the MIT license please consult `LICENSE.txt`.

## References
 [Robust Deep Learning With Mathematical Guarantees](https://johanneslederer.com/)
 
 Cite as "S. Huang, F. Xie, and J. Lederer. Robust Deep Learning With Mathematical Guarantees.".

