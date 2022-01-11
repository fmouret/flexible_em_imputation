# A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data

This repository contains the code that implements the algorithm proposed in "A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data".

The proposed algorithm is mainly dedicated to the imputation of missing values, but can be used as well for clustering in the presence of missing data.

```python
fem = FEM(K=K)
fem.fit(data_with_missing_values)

X_imputed = fem.X_hat
```
