import pandas as pd

from src.binning.binning_utils import fisher_jenks_breaks, calculate_number_of_bins, cut_data_in_column

dataset = pd.read_csv('../../resources/data/dataset.csv')

norm_peaks = dataset['peak_magnitude_norm'].values

n_freedman = calculate_number_of_bins(norm_peaks, method='fd')
n_sturges = calculate_number_of_bins(norm_peaks, method='sturges')
n_scott = calculate_number_of_bins(norm_peaks, method='scott')

breaks_freedman = fisher_jenks_breaks(norm_peaks, n_freedman)
breaks_sturges = fisher_jenks_breaks(norm_peaks, n_sturges)
breaks_scott = fisher_jenks_breaks(norm_peaks, n_scott)

binned_fd = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_freedman)
binned_sturges = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_sturges)
binned_scott = cut_data_in_column(dataset, 'peak_magnitude_norm', breaks_scott)

dataset.insert(loc=6, column='bins_fd', value=binned_fd)
dataset.insert(loc=7, column='bins_sturges', value=binned_sturges)
dataset.insert(loc=8, column='bins_scott', value=binned_scott)

''' # Histograms
plt.hist(dataset['bins_fd'], bins=n_freedman)
plt.show()

plt.hist(dataset['bins_sturges'], bins=n_sturges)
plt.show()

plt.hist(dataset['bins_scott'], bins=n_scott)
plt.show()
'''

dataset.to_csv('../../resources/data/dataset_binned.csv', sep=',', encoding='utf-8', index=False)
