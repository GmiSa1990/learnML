[TOC]

# Classification









## Binary classifier



## confusion matrix

对于二分类，混淆矩阵为2x2，它的每一行代表一个实际的类，比如数字识别，两个类别5和非5，第一行代表类别非5，第二行代表5。第一行第一列代表正确识别为非5的样本个数(true negatives, TN)，第一行第二列代表不是‘非5’被错误识别为非5的样本个数(false negatives, FN)。第二行第一列代表不是5（实际为非5）但被错误识别为5的样本个数(false positives, FP)，第二行第二列代表正确识别为5的样本个数(true positives, TP)。

一个好的混淆矩阵应该只在主对角线上有非零值。

常用的性能评估指标：

- Precision

  precision = TP / (TP + FP)

- Recall, sensitivity, true positives rate TPR

  recall = TP / (TP + FN)

- F~1~ score, combine the above two metrics

  F~1~ = 2 / (1/precision + 1/recall) = 2 * (precision * recall / (precision + recall))

```python
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import cross_val_predict


from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score



```

## Precision/Recall tradeoff



##  ROC curve

receiver operating characteristic

area under the curve AUC

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
```



## Multinomial classification





## Multilabel classification





## Multi-output classification







# SVM

