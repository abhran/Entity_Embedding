import pandas as pd 
import numpy as np 




test=pd.read_csv("teste.csv")
train=pd.read_csv("trainee.csv")
data = pd.concat([train, test]).reset_index(drop=True)
data=data[["InvoiceNo","StockCode","Quantity","CustomerID","Country","year","months","day","hour","frq_inv","frq_custo"]]
print(data.head(50))
print(data.info())
print(data.describe())