import pandas as pd

data_path = 'C:\\Users\\daidd\\Documents\\GitHub\\601-Proj\\data\\data1.csv'

data = pd.read_csv(data_path)

print(data.head)