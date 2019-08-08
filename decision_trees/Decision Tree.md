# Decision Tree

``` c
Decision Trees are fairly intuitive and their decisions are easy to interpret, such models are often called *white box models*, In contrast, random forests or neural networks are generally considered *black box models*
```



## CART

*Classification And Regression Tree*

**Cost function**:
$$
J(k,t_k) = {m_{left}  \over m}G_{left} + {m_{right}  \over m}G_{right}
$$
where, G~left~ / G~right~ measures the impurity of the left/right subsets, m~left~ / m~right~ is the number of instances in the left/right subset.

Once is has successfully split the training set in two, it splits the subsets using the same logic, then the sub-subsets and so on, **recursively**. It stops recursing once is reaches the maximum depth (defined by the max_depth hyperparameter), or if it cannot find a split that will reduce impurity.

Actually, the CART algorithm is a **greedy algorithm**: it greedily searches for an optimum split at the top level, then repeats the process at each level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a reasonably good solution, but it is not guaranteed to be the optimal solution.

## Regularization Hyperparameters

A model is called a *nonparametric model*, not because it doesn't have any parameters (it often has a lot) but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data. In contrast, a *parametric model* such as linear model has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).

## Limitation

1. Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation.

   ![]()

   One way to limit this problem is to use PCA, which often results in a better orientation of the training data.

2. Sensitive to small variations in the training data.

   Random Forests can limit this instability by averaging predictions over many trees.

## Exercises

1. what is the approximate depth of a Decision Tree trained (without restriction) on a training set with a million instances?

   20

2. 

   

# Ensemble Learning and Random Forests

