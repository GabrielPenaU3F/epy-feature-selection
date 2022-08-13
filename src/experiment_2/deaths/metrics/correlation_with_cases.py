import pandas as pd
import scipy.stats as stats

dataset = pd.read_csv('../../../../resources/data/cases/dataset_v2/dataset_2.3.csv')

peak_magnitude = dataset['peak_magnitude_norm'].to_numpy()
total_deaths = dataset['total_deaths_norm'].to_numpy()

pearson_r, pearson_pv = stats.pearsonr(peak_magnitude, total_deaths)
spearman_rho, spearman_pv = stats.spearmanr(peak_magnitude, total_deaths)
kendall_tau, kendall_pv = stats.kendalltau(peak_magnitude, total_deaths)

print('---- Correlation metrics ----')
print('Pearson r: {metric:.4f}    -    p-value: {pv}'.format(metric=pearson_r, pv=pearson_pv))
print('Spearman \u03C1: {metric:.4f}    -    p-value: {pv}'.format(metric=spearman_rho, pv=spearman_pv))
print('Kendall \u03C4: {metric:.4f}    -    p-value: {pv}'.format(metric=kendall_tau, pv=kendall_pv))
