# Libraries
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('./datasets/format03_data_300.csv')

# Show
print(df)
print(df.columns)
print(df.dtypes)
print(df['0'].isna().sum())

# Convert string to numpy
print(df['0'])
#df['0'] = df['0'].map(np.fromstring)

#print(df['0'])

aux = df.loc[0, '0']
print("aaa")
print(aux)
print(type(aux))

a = np.fromstring(aux, dtype=np.complex_)
print(a)