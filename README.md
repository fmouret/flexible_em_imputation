# A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data

This repository contains the code that implements the algorithm proposed in "A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data". The proposed algorithm is based on the clustering algorithm proposed in  https://arxiv.org/pdf/1907.01660.pdf. The original algorithm (in the complete data case) can be found here: https://github.com/violetr/fem 

The proposed algorithm is mainly dedicated to the imputation of missing values, but can be used as well for clustering in the presence of missing data.

```python
fem = FEM(K=K)
fem.fit(data_with_missing_values)

X_imputed = fem.X_hat
```
## Copyright

Authors

- Florian Mouret
- Alexandre Hippert-Ferrer
- Frédéric Pascal
- Jean-Yves Tourneret
