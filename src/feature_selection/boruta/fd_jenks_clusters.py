import numpy as np
import pandas as pd
from boruta import BorutaPy
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../../../resources/data/peak_magnitude_output_imputed.csv')

nonentry_cols = ['location', 'n_peak', 'peak_magnitude_norm', 'cluster_fd_km', 'cluster_fd_agg', 'cluster_fd_jenks',
                 'cluster_sturges_jenks', 'cluster_scott_jenks']
X = data.loc[:, ~data.columns.isin(nonentry_cols)]

y_jenks = data['cluster_fd_jenks'].values

# K-Means clusters
X_train, X_test, y_train, y_test = train_test_split(X, y_jenks, test_size=0.2, random_state=1)
forest = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_features=0.7,
                                min_samples_split=5, min_samples_leaf=3, oob_score=True)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', random_state=1)
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

forest = RandomForestClassifier(n_estimators=30, max_samples=0.8, max_features=0.7,
                                min_samples_split=5, min_samples_leaf=3, oob_score=True)
forest = forest.fit(X_filtered, y_train)
y_pred = forest.predict(X_test_filtered)

crossval_score = cross_val_score(forest, X_filtered, y_train, cv=5).mean()
acc = metrics.accuracy_score(y_test, y_pred)
print('---Results--- \n')
print('Accuracy on test set: {accuracy:.2f}%'.format(accuracy=acc))
print('Out-of-bag error: {oob:.4f}'.format(oob=forest.oob_score_))
print('Mean cross-validation score: {cv_score:.4f} \n'.format(cv_score=crossval_score))
print('Feature importance (MDI): \n')

importance_mdi_value = forest.feature_importances_
importance_mdi = pd.Series(importance_mdi_value, index=new_feature_names)
importante_fpermut_value = permutation_importance(forest, X_test_filtered, y_test, n_repeats=10, random_state=42, n_jobs=2)
importance_fpermut = pd.Series(importante_fpermut_value.importances_mean, index=new_feature_names)

for i, score in enumerate(importance_mdi_value):
    name = new_feature_names[i]
    f_score = 100 * score
    print('{feature}: {percent:.2f}%'.format(feature=name, percent=f_score))

fig, axes = plt.subplots(2, 1)
importance_mdi.plot.bar(ax=axes[0])
axes[0].set_title('Feature importance (MDI)')
importance_fpermut.plot.bar(ax=axes[1], yerr=importante_fpermut_value.importances_std)
axes[1].set_title('Feature importance (feature permutation)')
fig.tight_layout()
plt.show()
