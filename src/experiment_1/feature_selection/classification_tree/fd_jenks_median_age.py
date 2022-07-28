import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv('../../../../resources/data/dataset_v1/dataset_1.6.csv')

X = np.array(data['median_age']).reshape(-1, 1)

y_fd_jenks = data['cluster_fd_jenks'].values

# FD - JenksFisher clusters
X_train, X_test, y_train, y_test = train_test_split(X, y_fd_jenks, test_size=0.2, random_state=1)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4,
                              min_samples_split=3, min_samples_leaf=5)
tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

dot_data = export_graphviz(tree, out_file=None,
                           filled=True, rounded=True, special_characters=True,
                           feature_names=['median_age'], class_names=True
                           )
graph = graphviz.Source(dot_data)
graph.render('../../../resources/results/fd_jenks_median_age_tree')

acc = metrics.accuracy_score(y_test, y_pred) * 100
print('Accuracy on test set: {accuracy:.2f}% \n'.format(accuracy=acc))
