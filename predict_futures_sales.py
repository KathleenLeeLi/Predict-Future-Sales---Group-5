import pandas as pd
#print(pd.__version__)
import numpy as np
# from googletrans import Translator
# import re
import datetime
from sklearn import tree
from matplotlib import pyplot as plt
import matplotlib
from pmdarima.arima import auto_arima


sales_train_df = pd.read_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\sales_train_v2.csv', sep=',',header=0)
shops_df = pd.read_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\shops.csv', sep=',',header=0)
item_category_df = pd.read_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\item_categories.csv', sep=',',header=0)
items_df = pd.read_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\items.csv', sep=',',header=0)
test_df = pd.read_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\test.csv', sep=',',header=0)

shops_df.head()
item_category_df.head()
items_df.head()

sales_train_df.info()
sales_train_df.shape
sales_train_df.columns
sales_train_df.dropna(inplace=True)

test_df.shape
test_df.head()
len(test_df['item_id'].unique())  #-- 5100
len(test_df['item_id'])  #-- 214200

#----------------------------------------------------
# Train, Test modified
#----------------------------------------------------

#---- Modification of train set until 31-10-2015 (to respect kaggle mentioned)
# sales_train_df['date2']=pd.to_datetime(sales_train_df['date'])
# sales_train_df2=sales_train_df[sales_train_df['date2']<='2015-10-31']
# sales_train_df2['date2'].max()  # Oct 15
# sales_train_df2['date2'].min()  # Jan 13

#Elan way -- works better!
raw_date = sales_train_df.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

raw_date.head()
min(raw_date)
max(raw_date)

sales_train_df['day'] = raw_date.dt.day
sales_train_df['month'] = raw_date.dt.month
sales_train_df['year'] = raw_date.dt.year
sales_train_df['date2']=raw_date

sales_train_df.head()

sales_train_df.to_csv('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\sales_train_prueba.csv')



min(sales_train_df['date2'])
max(sales_train_df['date2'])

# fecha=sales_train_df['date2'][sales_train_df['date2']>='2015-11-01']
# fecha.to_excel('D:\\Mis Documentos\\Data Science Certificate\\Assignments\\Group Assignment\\fecha.xlsx')

#----- Training modified with sum grouped

sales_train_df=sales_train_df.sort_values(by=['date2'])

sales_train_df.head()

len(sales_train_df2)
len(sales_train_df2['shop_id'].unique())

sales_train_df.head()


sales_train_df3=sales_train_df.groupby(['shop_id','month','year']).count()

sales_train_df4=sales_train_df3.add_suffix('_total').reset_index()

sales_train_df4=sales_train_df4.sort_values(by=['year','month'])

# sales_train_df4.drop('month_year_total',axis=1)

sales_train_df4['month_year']=sales_train_df4['month'].astype(str)+'-'+sales_train_df4['year'].astype(str)

vec_date=sales_train_df4['month_year'].unique()

vec_date=pd.DataFrame(vec_date)

vec_date['uniq_date']=vec_date.index.get_values()+1

vec_date.columns=['month_year','uniq_date']

sales_train_df5=pd.merge(sales_train_df4,vec_date,on='month_year')

sales_train_df5

max(sales_train_df5['uniq_date'])
min(sales_train_df5['uniq_date'])
max(sales_train_df5['item_cnt_day_total'])
min(sales_train_df5['item_cnt_day_total'])

#----- Test modified 

test_df.head()
test_df=test_df.drop('ID',axis=1)
test_df2=test_df.groupby('shop_id').count()

test_df2['uniq_date']=35

test_df3=test_df2.reset_index()

#----- In case of require prices of the items
# item_price_2015=sales_train_df[['item_price','item_id','date2']][(sales_train_df['date2']>='2015-11-01') & (sales_train_df['date2']<='2015-11-12')]
# item_price_2015['date2'].max()
# sales_train_df['date2'].min()
# sales_train_df2=pd.merge(test_df,item_price_2015,on='item_id',how='left')

# len(item_price_2015['item_id'].unique())
# len(item_price_2015['item_id'])


#----------------------------------------------------
# Translation of russian names
#----------------------------------------------------

# gt=Translator()

# list_shop=shop['shop_name'].tolist()
# list_rus_items=[((items['item_name'].get_values()[i])) for i in range(0,len(items['item_name'].get_values()))]

# rus_shop=gt.translate(list_shop)

# rus_items=gt.translate(['Hola'])

# len(items['item_name'])
# rus_items.shape
# rus_items.head()

# trans_items=[]

# rus_items.text

# for trans in rus_items:
#     # trans_items.append(trans.text)
#     print(trans.text)


# re.sub('\W+',' ','???? ???????? 1?:???????????  [???????? ??????]')


# gt.translate('???? ???????? 1?:???????????  [???????? ??????]').text
# print(gt.translate(items['item_name']).text)

#----------------------------------------------------
# Regression tree
#----------------------------------------------------

def mape(real,pred):
    result=100*np.mean(abs((real-pred)/real))
    return result

def mae(real,pred):
    result=np.mean(abs((real-pred)))
    return result    

X=sales_train_df5[['shop_id','uniq_date']]
y=np.array(sales_train_df5['item_cnt_day_total'])
x1=test_df3[['shop_id','uniq_date']]

tree_sales = tree.DecisionTreeRegressor(max_depth=30,min_samples_split=2)
tree_sales = tree_sales.fit(X,y)
tree_sales = tree_sales.predict(x1)

len(tree_sales)
len(test_df3['item_id'])

mape(test_df3['item_id'],tree_sales)
np.sqrt(mae(test_df3['item_id'],tree_sales))


#----------------------------------------------------
# ARIMA
#----------------------------------------------------

min(sales_train_df5['shop_id'].unique())
min(test_df3['shop_id'].unique())

pred_final={}

i=2
model=auto_arima(np.array(sales_train_df5['item_cnt_day_total'][sales_train_df5['shop_id']==i]))
model.fit(np.array(sales_train_df5['item_cnt_day_total'][sales_train_df5['shop_id']==i]))
pred=model.predict(n_periods=len(test_df3['item_id'][test_df3['shop_id']==i]))
pred_final[i]=pred

test_df3['item_id'][test_df3['shop_id']==2]


for i in test_df3['shop_id'].unique():
    
    model=auto_arima(sales_train_df5['item_cnt_day_total'][sales_train_df5['shop_id']==i])
    model.fit(sales_train_df5['item_cnt_day_total'][sales_train_df5['shop_id']==i])
    pred=model.predict(n_periods=len(test_df3['item_id'][test_df3['shop_id']==i]))
    pred_final[i]=pred


model=auto_arima(y)
model.fit(y)

pred=model.predict(n_periods=len(test_df3['item_id']))
pred=pd.DataFrame(pred,index=test_df3['item_id'].index,columns=['Pred'])

len(pred)
len(test_df3['item_id'])

plt.plot(y,label='train',color='b')
plt.plot(test_df3['item_id'],label='real',color='g')
plt.plot(pred,label='pred',color='r')
plt.show()

mape(test_df3['item_id'],pred['Pred'])