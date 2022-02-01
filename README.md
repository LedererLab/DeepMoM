# DeepMoM

DeepMom is a pipeline for robust deep learning.
It is described here: [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035).
This repository provides the corresponding implementations.

## Simulations

The code in `SimulationsRegression.R` provides a comparison of least-squares, Huber, and least-absolute deviation estimators to our ReLU-based DeepMoM estimators in regression problems;
the code in `SimulationsClassification.R` provides a comparison of soft-max cross-entropy estimators to our ReLU-based DeepMoM estimators in classification problems. 

## Applications

The code in `TcgaApplication.R` applies DeepMoM to seven TCGA data sets.

## Other folders

**AdditionalFunctions**: The source code of the functions required for computing DeepMoM.

**TcgaData**: The TCGA data sets. 

## Repository authors 

* [Shih-Ting Huang](https://johanneslederer.com/team/), Ph.D. student in Mathematical Statistics, Ruhr-University Bochum

* [Johannes Lederer](https://johanneslederer.com), Professor in Mathematical Statistics, Ruhr-University Bochum

## Programing language and supported platforms

The code in this repository is written in R with version `R 4.1.2` and supports all plarforms which are supported by R itself.

## Dependencies

This repository does not depend on any R libraries or external sources.

## Licensing

All codes are licensed under the MIT license. To
view the MIT license please consult `LICENSE.txt`.

## References
 The paper can be found here: [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035)
 
 It should be cited as "Huang, S.T. and Lederer, J., 2021. DeepMoM: Robust Deep Learning With Median-of-Means. arXiv:2105.14035."
