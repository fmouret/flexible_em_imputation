# A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data

This repository contains the code that implements the algorithm proposed in the journal paper "A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data", which is available here: https://arxiv.org/abs/2201.12020. 

The proposed algorithm is based on the clustering algorithm proposed in  https://arxiv.org/pdf/1907.01660.pdf. The original algorithm (in the complete data case) can be found here: https://github.com/violetr/fem 

The proposed algorithm is mainly dedicated to the imputation of missing values, but can be used as well for clustering in the presence of missing data.

```python
fem = FEM(K=K)
fem.fit(data_with_missing_values)

X_imputed = fem.X_hat
```
## Minimal example with synthetic data

A minimal example is provided in the python file "run minimal example.py". Imputatation tasks are conudcted on synthetic data (see paper for details). The FEM algorithm is compared to the KNN and MICE imputation algorithms (using their sklearn implementation).

## Dependencies

- matplotlib==3.2.1
- numpy==1.19.5
- scikit_learn==0.24.2
- scipy==1.5.4

## Copyright

Authors

- Florian Mouret
- Alexandre Hippert-Ferrer
- Frédéric Pascal
- Jean-Yves Tourneret

## Cite this work

      @article{mouret2022fem,
      title={A Robust and Flexible EM Algorithm for Mixtures of Elliptical Distributions with Missing Data}, 
      author={Florian Mouret and Alexandre Hippert-Ferrer and Frédéric Pascal and Jean-Yves Tourneret},
      year={2022},
      journal={Under review},      
      eprint={2201.12020},      
      archivePrefix={arXiv},      
      primaryClass={stat.ML}
      }
