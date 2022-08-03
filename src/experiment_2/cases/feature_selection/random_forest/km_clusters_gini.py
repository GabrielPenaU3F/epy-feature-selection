import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../../../../../resources/data/cases/dataset_v2/dataset_2.3.csv')

nonentry_cols = ['location', 'n_peak', 'peak_magnitude_norm', 'fd_km_clusters', 'fd_agg_clusters', 'fd_jenks_clusters',
                 'sturges_jenks_clusters', 'scott_jenks_clusters', 'fd_uniform_clusters']
X = data.loc[:, ~data.columns.isin(nonentry_cols)]
feature_names = X.columns.values

y_fd_km = data['fd_km_clusters'].values

# K-Means clusters
X_train, X_test, y_train, y_test = train_test_split(X, y_fd_km, test_size=0.1, random_state=1)
forest = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_features=0.7, criterion='gini',
                                min_samples_split=5, min_samples_leaf=3, oob_score=True)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

crossval_score = cross_val_score(forest, X_train, y_train, cv=5).mean()
acc = metrics.accuracy_score(y_test, y_pred) * 100
print('---Results--- \n')
print('Accuracy on test set: {accuracy:.2f}%'.format(accuracy=acc))
print('Out-of-bag error: {oob:.4f}'.format(oob=forest.oob_score_))
print('Mean cross-validation score: {cv_score:.4f} \n'.format(cv_score=crossval_score))
print('Feature importance (MDI): \n')

importance_mdi_value = forest.feature_importances_
importance_mdi = pd.Series(importance_mdi_value, index=feature_names)
importance_fpermut_value = permutation_importance(forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
importance_fpermut = pd.Series(importance_fpermut_value.importances_mean, index=feature_names)

for i, score in enumerate(importance_mdi_value):
    name = X.columns[i]
    f_score = 100 * score
    print('{feature}: {percent:.2f}%'.format(feature=name, percent=f_score))

sorted_names_mdi = [name for _, name in sorted(zip(importance_mdi_value, X.columns), reverse=True)]
sorted_names_fpermut = [name for _, name in sorted(zip(importance_fpermut_value, X.columns), reverse=True)]
importance_mdi = importance_mdi.sort_values(ascending=False)
importance_fpermut = importance_fpermut.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_tight_layout(True)
importance_mdi.plot.bar(ax=ax)
ax.set_title('Feature importance (MDI)')
fig.savefig('../../../../resources/results/experiment_2/forest_fd_km_gini_importance_mdi.pdf')
ax.clear()
importance_fpermut.plot.bar(ax=ax, yerr=importance_fpermut_value.importances_std)
ax.set_title('Feature importance (feature permutation)')
fig.savefig('../../../../resources/results/experiment_2/forest_fd_km_gini_importance_permutation.pdf')
