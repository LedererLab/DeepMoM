# DeepMoM

This repository provides the implementations of the methods described in [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035).

## Simulations

We provide an example code in `SimulationStudy.R` for a comparison of least-squars, Huber, and least absolute deviation estimators with ReLU based DeepMoM estimators for regression type problems and another code in `Classifier.R` for comparison of soft-max cross entropy estimators and ReLU based DeepMoM estimators for classification tasks. 

## Applications

We provide an example code in `TcgaApplication.R` to apply the DeepMoM structure on seven TCGA data sets.

## Other folders
**AdditionalFunctions** : The source codes of some functions required for computing DeepMoM estimators.

**TcgaData** : The TCGA data sets. 

## Repository authors 

* Shih-Ting Huang, Ph.D. student in Mathematical Statistics, Ruhr-University Bochum

* Johannes Lederer, Professor in Mathematical Statistics, Ruhr-University Bochum

## Supported languages and platforms

All of the codes in this repository are written in R with version `R 4.1.2` and supports all plarforms which are supported by R itself.

## Dependencies

This repository does not depend on any R libraries or external sources.

## Licensing

All codes are licensed under the MIT license. To
view the MIT license please consult `LICENSE.txt`.

## References
 [DeepMoM: Robust Deep Learning With Median-of-Means](https://arxiv.org/abs/2105.14035)
 
 Cite as "S. Huang and J. Lederer. DeepMoM: Robust Deep Learning With Median-of-Means.".

