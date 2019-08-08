import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

# iris = datasets.load_iris()
# X = iris["data"][:,(2,3)]
# y = (iris["target"] == 2).astype(np.float64)
# svm_clf = Pipeline((('scale', StandardScaler()),
#                     ('lin_svm', LinearSVC(C=1, loss='hinge'))))
# svm_clf.fit(X,y)
# svm_clf.predict([[5.5, 1.7]])


moons = datasets.make_moons()
X = moons[0]
y = moons[1].astype(np.float64)
polynomial_svm_clf1 = Pipeline((
        ('polynomial', PolynomialFeatures(degree=2)),
        ('scale', StandardScaler()),
        ('linear_svm', LinearSVC(C=1, loss='hinge'))
))
polynomial_svm_clf1.fit(X,y)

polynomial_svm_clf2 = Pipeline((

))