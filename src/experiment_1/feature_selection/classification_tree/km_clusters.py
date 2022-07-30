import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv('../../../../resources/data/cases/dataset_v1/dataset_1.6.csv')

nonentry_cols = ['location', 'n_peak', 'peak_magnitude_norm', 'cluster_fd_km', 'cluster_fd_agg', 'cluster_fd_jenks', 'cluster_sturges_jenks', 'cluster_scott_jenks']
X = data.loc[:, ~data.columns.isin(nonentry_cols)]

y_km = data['cluster_fd_km'].values

# K-Means clusters
X_train, X_test, y_train, y_test = train_test_split(X, y_km, test_size=0.2, random_state=1)
tree = DecisionTreeClassifier(criterion='gini', min_samples_split=3)
tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

dot_data = export_graphviz(tree, out_file=None,
                           filled=True, rounded=True, special_characters=True,
                           feature_names=X.columns, class_names=True
                           )
graph = graphviz.Source(dot_data)
graph.render('../../../resources/results/km_tree')

acc = metrics.accuracy_score(y_test, y_pred) * 100
print('---Results--- \n')
print('Accuracy on test set: {accuracy:.2f}% \n'.format(accuracy=acc))
print('Feature importance: \n')
importance = tree.feature_importances_
for i, score in enumerate(importance):
    name = X.columns[i]
    f_score = 100*score
    print('{feature}: {percent:.2f}%'.format(feature=name, percent=f_score))

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_tight_layout(True)
ax.bar(X.columns.values, importance)
ax.set_title('Feature importance (MDI)')
ax.tick_params(axis='x', labelrotation=90)
fig.savefig('../../../resources/results/ctree_km_importance_mdi.pdf')
