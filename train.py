import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Embedding,Concatenate
# models = []
import pandas as pd 
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
import numpy as np
from sklearn.model_selection import StratifiedKFold
from  sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.preprocessing import PowerTransformer

from keras.callbacks import ModelCheckpoint
import keras.objectives
from keras.callbacks import CSVLogger




import pandas as pd
from sklearn import preprocessing
# # from pygments.layers import data
# # from pygments.lexers import data
# # read training data
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

# test.loc[:, "UnitPrice"] = -1000000000
# # concatenate both training and test data
# data = pd.concat([train, test]).reset_index(drop=True)
# for feat in ["InvoiceNo","StockCode","Quantity","CustomerID","Country","year","months","day","hour","frq_inv","frq_custo"]:
#     lbl_enc = preprocessing.LabelEncoder()
#     temp_col = data[feat].fillna("NONE").astype(str).values
#     data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
# train2= data[data.UnitPrice != -1000000000].reset_index(drop=True)
# test2 = data[data.UnitPrice == -1000000000].reset_index(drop=True)
# # print(train2.head(50))
# # print(test2.head(50))
# train2.to_csv('trainee.csv')
# test2.to_csv('teste.csv')



# tensorboard_cb = keras.callbacks.TensorBoard(
#     log_dir='tensorboard!!',
#     histogram_freq=1,
#     write_graph=True,
#     write_images=True
# )
csv_logger = CSVLogger(f'swap/loss.csv', append=True, separator=',')


# data=pd.read_csv("useful_combined.csv")
train=pd.read_csv("trainee.csv")
test=pd.read_csv("teste.csv")

q=train[(train["UnitPrice"]>5000) ].index
print(q)
train=train.drop(q)
train= shuffle(train, random_state=20)

# print(data.info())
# print(data.describe())
# print(train.info())
# print(train.describe())
# print(test.info())
# print(test.describe())

tx=train[["InvoiceNo","StockCode","Quantity","CustomerID","Country","year","months","day","hour","frq_inv","frq_custo"]]
ty=train["UnitPrice"].astype(float)
ty=np.array(ty)
# print(tx)
# power = PowerTransformer(method='yeo-johnson', standardize=True)
# ty=np.reshape(ty,(-1, 1))

# ty = power.fit_transform(ty)
# print(np.max(ty))
# print(np.min(ty))
# ty=np.reshape(ty,(1, -1))
# data_trans = power.inverse_transform(ty)
# x_train, x_test, y_train, y_test = train_test_split(tx, ty, test_size=0.2, random_state=42)
x_train=tx
y_train=ty
x1=x_train["InvoiceNo"].astype(int)
x1=np.array(x1)
x2=x_train["StockCode"].astype(int)
x2=np.array(x2)
x3=x_train["Quantity"].astype(int)
x3=np.array(x3)
x4=x_train["CustomerID"].astype(int)
x4=np.array(x4)
x5=x_train["Country"].astype(int)
x5=np.array(x5)
x6=x_train["year"].astype(int)
x6=np.array(x6)
# print("sdfghjkjhgfddfghjkjhgcxcvbnm,mnbvc---------------------------------------------",x3)
x7=x_train["months"].astype(int)
x7=np.array(x7)

# InvoiceNo,StockCode,Quantity,UnitPrice,CustomerID,Country,year,months,day,hour,frq_inv,frq_custo
x8=x_train["day"].astype(int)
x8=np.array(x8)
x9=x_train["hour"].astype(int)
x9=np.array(x9)
x10=x_train["frq_inv"].astype(int)
x10=np.array(x10)
x11=x_train["frq_custo"].astype(int)
x11=np.array(x11)
# x12=x_train["Country"].astype(int)
# x12=np.array(x7)


# x11=x_test["InvoiceNo"].astype(int)
# x11=np.array(x11)
# x22=x_test["StockCode"].astype(int)
# x22=np.array(x22)
# x33=x_test["Description"].astype(int)
# x33=np.array(x33)
# x44=x_test["Quantity"].astype(int)
# x44=np.array(x44)
# x55=x_test["InvoiceDate"].astype(int)
# x55=np.array(x55)
# x66=x_test["CustomerID"].astype(int)
# x66=np.array(x66)
# x77=x_test["Country"].astype(int)
# x77=np.array(x77)

inputs = []
embeddings = []

invoice = Input(shape=(1,))
embedding = Embedding(22188, 200, input_length=1)(invoice)
embedding = Reshape(target_shape=(200,))(embedding)
inputs.append(invoice)
embeddings.append(embedding)
  
stockcode = Input(shape=(1,))
embedding = Embedding(3916,80, input_length=1)(stockcode)
embedding = Reshape(target_shape=(80,))(embedding)
inputs.append(stockcode)
embeddings.append(embedding)

quantity = Input(shape=(1,))
embedding = Embedding(436,25, input_length=1)(quantity)
embedding = Reshape(target_shape=(25,))(embedding)
inputs.append(quantity)
embeddings.append(embedding)
    
customerid = Input(shape=(1,))
embedding = Embedding(4372,75, input_length=1)(customerid)
embedding = Reshape(target_shape=(75,))(embedding)
inputs.append(customerid)
embeddings.append(embedding)

country = Input(shape=(1,))
embedding = Embedding(37,8, input_length=1)(country)
embedding = Reshape(target_shape=(8,))(embedding)
inputs.append(country)
embeddings.append(embedding)
  
  
year = Input(shape=(1,))
embedding = Embedding(2,2, input_length=1)(year)
embedding = Reshape(target_shape=(2,))(embedding)
inputs.append(year)
embeddings.append(embedding)
  
months = Input(shape=(1,))
embedding = Embedding(12,6, input_length=1)(months)
embedding = Reshape(target_shape=(6,))(embedding)
inputs.append(months)
embeddings.append(embedding)
  
day = Input(shape=(1,))
embedding = Embedding(6,4, input_length=1)(day)
embedding = Reshape(target_shape=(4,))(embedding)
inputs.append(day)
embeddings.append(embedding)
  
hour = Input(shape=(1,))
embedding = Embedding(15,5, input_length=1)(hour)
embedding = Reshape(target_shape=(5,))(embedding)
inputs.append(hour)
embeddings.append(embedding)

  
f_in = Input(shape=(1,))
embedding = Embedding(199,25, input_length=1)(f_in)
embedding = Reshape(target_shape=(25,))(embedding)
inputs.append(f_in)
embeddings.append(embedding)
  
f_cus = Input(shape=(1,))
embedding = Embedding(474,50, input_length=1)(f_cus)
embedding = Reshape(target_shape=(50,))(embedding)
inputs.append(f_cus)
embeddings.append(embedding)

x = Concatenate()(embeddings)
x = Dense(128, activation='relu')(x)

x=tf.keras.layers.BatchNormalization()(x)
x=tf.expand_dims(x,axis=-1)
x=tf.keras.layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2)(x)
x = Dense(64, activation='relu')(x)
x=tf.keras.layers.BatchNormalization()(x)
x = Dropout(.15)(x)
x=tf.expand_dims(x,axis=-1)
x=tf.keras.layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2)(x)
x = Dropout(.15)(x)
x = Dense(8, activation='relu')(x)
x = Dropout(.15)(x)
output = Dense(1)(x)
model = Model(inputs, output)
# model = Model(inputs, output)


from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))




def lr_scheduler(epoch, lr):
        if epoch%20==0:
                return lr * 0.9
        return lr

callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)





# opt=keras.optimizers.RMSprop(lr=1e-4,decay=0.95,momentum=0.9, epsilon=1e-8, name="RMSprop")
opt=keras.optimizers.Adam(lr=4e-4)
model.compile(loss = 'mean_absolute_error',optimizer=opt, metrics=[root_mean_squared_error])
# model.compile(loss =root_mean_squared_error ,optimizer="adam", metrics=['mean_absolute_error'])
print(model.summary())


cp1= ModelCheckpoint(filepath="swap/save_best.h5", monitor='loss',save_best_only=True, mode='min',verbose=1,save_weights_only=True)
cp2= ModelCheckpoint(filepath='swap/save_all.h5', monitor='loss',save_best_only=False ,verbose=1,save_weights_only=True)
# callbacks_list = [callback,cp1,cp2,csv_logger]
callbacks_list = [callback,cp1,cp2,csv_logger]



# X = shuffle([x1,x2,x3], random_state=20)
# ty = shuffle(ty, random_state=20)
# X_train, X_test, y_train, y_test = train_test_split([x1,x2,x3], ty, test_size=0.2, random_state=42)


# validation_data=([x11,x22,x33,x44,x55,x66,x77],y_test),
# model.fit(X_train,y_train, batch_size =1024, epochs = 1000, validation_split = 0.2,validation_data=(xval, yval))
model.fit([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11],y_train, batch_size =2048, epochs =1,shuffle=True)#,callbacks=[callbacks_list])
model_json = model.to_json()
with open(f"swap/model.json", "w") as json_file:
    json_file.write(model_json)
# model.save_weights(f"modelswap.h5")