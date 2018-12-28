import pandas as pd

dframe = pd.read_csv('train.csv')
print(dframe.Cloth_label.values)