import pandas as pd

# Cases
# dataset = pd.read_csv('../resources/data/cases/dataset_v2/dataset_2.2.csv')
# dataset = dataset.drop(columns=['location', 'n_peak', 'peak_magnitude_norm', 'fd_jenks_clusters',
#                                'sturges_jenks_clusters', 'scott_jenks_clusters', 'fd_agg_clusters',
#                                'fd_km_clusters', 'fd_uniform_clusters'])

# Deaths
dataset = pd.read_csv('../resources/data/deaths/dataset_deaths_2.2.csv')
dataset = dataset.drop(columns=['location', 'n_peak', 'peak_magnitude_norm', 'fd_jenks_clusters',
                                'sturges_jenks_clusters', 'scott_jenks_clusters', 'fd_agg_clusters',
                                'fd_km_clusters', 'fd_uniform_clusters'])

# for col in dataset.columns:
#     full_column = dataset[col]
#     nas = full_column.isna().sum()
#     print(str(col) + ': ' + str(nas))

df_size = len(dataset) * len(dataset.columns)
number_nas = dataset.isna().sum().sum()

print('Number of NAs: ' + str(number_nas))
print('Total number of data: ' + str(df_size))
print('NA percentage: ' + str(number_nas/df_size * 100) + '%')
