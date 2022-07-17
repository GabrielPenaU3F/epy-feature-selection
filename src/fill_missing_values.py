import pandas as pd
from sklearn.impute import KNNImputer

dataset = pd.read_csv('../resources/data/peak_magnitude_output_dataset.csv')

# Fill the vaccination columns with zeros
dataset['vaccination_1_dose'] = dataset['vaccination_1_dose'].fillna(0)
dataset['vaccination_full'] = dataset['vaccination_full'].fillna(0)

# Impute the rest of the values with KNN

location_col = dataset['location'].values
numerical_df = dataset.drop(columns=['location'])

imputer = KNNImputer(n_neighbors=3, weights="uniform")
imputed_dataset = pd.DataFrame(imputer.fit_transform(numerical_df), columns=numerical_df.columns)
imputed_dataset.insert(loc=0, column='location', value=location_col)

imputed_dataset.to_csv('../resources/data/peak_magnitude_output_imputed.csv', sep=',', encoding='utf-8', index=False)
