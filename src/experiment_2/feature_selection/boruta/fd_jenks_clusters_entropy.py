import numpy as np
import pandas as pd
from boruta import BorutaPy
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../../../../resources/data/cases/dataset_v2/dataset_2.3.csv')

nonentry_cols = ['location', 'n_peak', 'peak_magnitude_norm', 'fd_km_clusters', 'fd_agg_clusters', 'fd_jenks_clusters',
                 'sturges_jenks_clusters', 'scott_jenks_clusters', 'fd_uniform_clusters']
X = data.loc[:, ~data.columns.isin(nonentry_cols)]

y_jenks = data['fd_jenks_clusters'].values

# K-Means clusters
X_train, X_test, y_train, y_test = train_test_split(X, y_jenks, test_size=0.1, random_state=1)
forest = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_features=0.7, criterion='entropy',
                                min_samples_split=5, min_samples_leaf=3, oob_score=True)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', random_state=1, perc=70)
# find all relevant features
feat_selector.fit(np.array(X_train), np.array(y_train))

# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(X_train.columns,
                         feat_selector.ranking_,
                         feat_selector.support_))

# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(np.array(X_train))
X_test_filtered = feat_selector.transform(np.array(X_test))

# Filter the list of columns that were kept
feature_names = X.columns.values
boruta_support = list(feat_selector.support_)
new_feature_names = []
for i in range(len(feature_names)):
    if boruta_support[i]:
        new_feature_names.append(feature_names[i])

forest = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_features=0.7, criterion='entropy',
                                min_samples_split=5, min_samples_leaf=3, oob_score=True)
forest = forest.fit(X_filtered, y_train)
y_pred = forest.predict(X_test_filtered)

crossval_score = cross_val_score(forest, X_filtered, y_train, cv=5).mean()
acc = metrics.accuracy_score(y_test, y_pred) * 100
print('---Results--- \n')
print('Accuracy on test set: {accuracy:.2f}%'.format(accuracy=acc))
print('Out-of-bag error: {oob:.4f}'.format(oob=forest.oob_score_))
print('Mean cross-validation score: {cv_score:.4f} \n'.format(cv_score=crossval_score))
print('Feature importance (MDI): \n')

importance_mdi_value = forest.feature_importances_
importance_mdi = pd.Series(importance_mdi_value, index=new_feature_names)
importance_fpermut_value = permutation_importance(forest, X_test_filtered, y_test, n_repeats=10, random_state=42,
                                                  n_jobs=2)
importance_fpermut = pd.Series(importance_fpermut_value.importances_mean, index=new_feature_names)

for i, score in enumerate(importance_mdi_value):
    name = new_feature_names[i]
    f_score = 100 * score
    print('{feature}: {percent:.2f}%'.format(feature=name, percent=f_score))

sorted_names_mdi = [name for _, name in sorted(zip(importance_mdi_value, new_feature_names), reverse=True)]
sorted_names_fpermut = [name for _, name in sorted(zip(importance_fpermut_value, new_feature_names), reverse=True)]
importance_mdi = importance_mdi.sort_values(ascending=False)
importance_fpermut = importance_fpermut.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
fig.set_tight_layout(True)
importance_mdi.plot.bar(ax=ax)
ax.set_title('Feature importance (MDI)')
fig.savefig('../../../../resources/results/experiment_2/boruta_fd_jenks_entropy_importance_mdi.pdf')
ax.clear()
importance_fpermut.plot.bar(ax=ax, yerr=importance_fpermut_value.importances_std)
ax.set_title('Feature importance (feature permutation)')
fig.savefig('../../../../resources/results/experiment_2/boruta_fd_jenks_entropy_importance_permutation.pdf')
