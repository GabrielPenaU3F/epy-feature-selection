import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../../../../resources/data/cases/dataset_v2/dataset_2.3.csv')

nonentry_cols = ['location', 'n_peak', 'peak_magnitude_norm', 'fd_km_clusters', 'fd_agg_clusters', 'fd_jenks_clusters',
                 'sturges_jenks_clusters', 'scott_jenks_clusters', 'fd_uniform_clusters']
X = data.loc[:, ~data.columns.isin(nonentry_cols)]
feature_names = X.columns.values

y_fd_jenks = data['fd_km_clusters'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_fd_jenks, test_size=0.1, random_state=1)

acc_train_list = []
acc_test_list = []
for i in range(0, 19):
    forest = RandomForestClassifier(n_estimators=20 + i, max_samples=0.8, max_features=0.7, criterion='entropy',
                                    min_samples_split=5, min_samples_leaf=3, oob_score=True)
    forest = forest.fit(X_train, y_train)
    y_pred_train = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_pred_train) * 100
    acc_test = metrics.accuracy_score(y_test, y_pred_test) * 100
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(20, 20 + len(acc_train_list))
ax.plot(x, acc_train_list, label='Accuracy on train set')
ax.plot(x, acc_test_list, label='Accuracy on test set')
ax.set_xlabel('Number of trees')
ax.set_ylabel('Accuracy percent')
ax.grid(True, which="both")
ax.legend()
fig.set_tight_layout(True)

plt.show()
