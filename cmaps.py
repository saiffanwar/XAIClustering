#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

# ## 1. Import dataset

# In[3]:


index_col_names=['unit_id','time_cycle']
operat_set_col_names=['oper_set{}'.format(i) for i in range(1,4)]
sensor_measure_col_names=['sm_{}'.format(i) for i in range(1,22)]
all_col=index_col_names+operat_set_col_names+sensor_measure_col_names
print(all_col)


# In[4]:


train_df=pd.read_csv('Data/CMAP/train_FD001.txt',delim_whitespace=True,names=all_col)
train_df


# In[5]:


train_df[train_df.unit_id==1]


# In[6]:


train_df.info()


# The dimensions of test set and RUL data are different, because RUL file assigns one value per unit. Then, we assign the RUL value to the last data point for each unit and adding one to each row above

# In[7]:


test_df=pd.read_csv('Data/CMAP/test_FD001.txt',delim_whitespace=True,names=all_col)
test_df.head()


# In[8]:


test_df.info()


# In[9]:


y_true=pd.read_csv('Data/CMAP/RUL_FD001.txt',delim_whitespace=True,names=['RUL'])
y_true['unit_id']=y_true.index+1
y_true.head()


# In[10]:


y_true


# In[11]:


test_df[test_df.unit_id==1]


# ## 2. Obtain RUL

# ## 2.1 training set

# In[12]:


#find maximum time cycle for each unit d
max_time_cycle=train_df.groupby('unit_id')['time_cycle'].max()
print(max_time_cycle)
rul = pd.DataFrame(max_time_cycle).reset_index()
rul.columns = ['unit_id', 'max']
rul.head()


# In[13]:


rul


# In[14]:


#merge train_df with rul dataframe based on number unit, that is the column join, using the left join
train_df = train_df.merge(rul, on=['unit_id'], how='left')


# In[15]:


train_df['RUL'] = train_df['max'] - train_df['time_cycle']
train_df.drop('max', axis=1, inplace=True)


# In[16]:


train_df[train_df.unit_id==1].iloc[:,[1,-1]]


# ## 2.2 test set

# In[17]:


test_df['RUL']=0
for i in range(1,101):
    test_df.loc[test_df.unit_id==i,'RUL']=range(int(y_true.RUL[y_true.unit_id==i])+len(test_df[test_df.unit_id==i])-1,
                                      int(y_true.RUL[y_true.unit_id==i])-1,-1)


# In[18]:


test_df.iloc[:,[0,1,-1]]


# In[19]:


y_true


# In[20]:


#check if it's correct changing unit_id
y_true
test_df.loc[test_df.unit_id==5,['unit_id','time_cycle','RUL']]


# ## 3. Feature Selection

# In[21]:


train_df.head()


# In[22]:


train_df[index_col_names].corr()


# In[23]:


train_df[operat_set_col_names].corr()


# In[24]:


train_df[sensor_measure_col_names].corr()


# From correlation matrix, we notice:
# * oper_set3 is not correlated with the other variables
# * sensor 1, 5,10,16,18,19

# In[25]:


train_df.iloc[:,1:-1].corr()


# In[26]:


for v_idx,v in enumerate(all_col[1:]):
    for i in train_df['unit_id'].unique():
        plt.plot('RUL',v,data=train_df[train_df['unit_id']==i])
    plt.xlabel('RUL')
    plt.ylabel(v)
#    plt.show()


# In[27]:


fig,ax=plt.subplots(13,2,figsize=(12,20))
fig.tight_layout()
r,c=0,0
for v_idx,v in enumerate(all_col[1:]):
    for i in train_df['unit_id'].unique():
        ax[r][c].plot('RUL',v,data=train_df[train_df['unit_id']==i])

    ax[r][c].set_xlabel('RUL')
    ax[r][c].set_ylabel(v)
    if c<1:
        c+=1
    elif c==1:
         r+=1
         c-=1


# In[28]:



#cols_drop=['oper_set3','sm_1','sm_5','sm_6','sm_10','sm_14','sm_16','sm_18','sm_19']
#train_df = train_df.drop(cols_drop, axis = 1)
#test_df = test_df.drop(cols_drop, axis = 1)


# In[29]:


train_df.head()


# In[30]:


test_df.head()


# In[31]:


train_df.corr()


# ## 4. Data Preprocessing

# In[32]:


from sklearn.preprocessing import MinMaxScaler


# In[33]:


features=list(train_df.columns[1:-1])


# In[34]:


features


# In[35]:


min_max_scaler = MinMaxScaler(feature_range=(0,1))

train_df[features] = min_max_scaler.fit_transform(train_df[features])
test_df[features] = min_max_scaler.fit_transform(test_df[features])


# In[36]:


test_df.head()


# In[37]:


test_df.describe()


# In[38]:


train_df


# In[39]:


features = train_df.drop(['unit_id','RUL'],axis=1).columns


# In[40]:


X_train = train_df.drop(['unit_id','RUL'],axis=1).values
y_train = train_df['RUL'].values


# In[41]:


X_test = test_df.drop(['unit_id','RUL'],axis=1).values
y_test = test_df['RUL'].values

with open('processed_cmaps.pck', 'wb') as f:
    pickle.dump([X_train, y_train, X_test, y_test, features], f)


# ## 5. Gradient Boosting

# In[42]:


len(X_train[0])


# In[100]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[101]:


X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train2


# In[105]:


reg = GradientBoostingRegressor(max_depth=5,n_estimators=500,random_state=42)
reg.fit(X_train, y_train)

with open('saved/CMAPS_model.pck', 'wb') as f:
    pickle.dump(reg, f)
print('model saved')

# predict and evaluate
y_hat_train = reg.predict(X_train)
print('training RMSE: ',np.sqrt(mean_squared_error(y_train, y_hat_train)))

print()

#y_hat_val = reg.predict(X_val)
#print('val RMSE: ',np.sqrt(mean_squared_error(y_val, y_hat_val)))
#

y_hat_test = reg.predict(X_test)
print('test RMSE: ',np.sqrt(mean_squared_error(y_test, y_hat_test)))


# In[106]:


reg = GradientBoostingRegressor(max_depth=5,n_estimators=500,random_state=42)
reg.fit(X_train, y_train)

# predict and evaluate
y_hat_train = reg.predict(X_train)
print('training RMSE: ',np.sqrt(mean_squared_error(y_train, y_hat_train)))

print()

y_hat_test = reg.predict(X_test)
print('test RMSE: ',np.sqrt(mean_squared_error(y_test, y_hat_test)))


# In[85]:


X = pd.concat([pd.DataFrame(X_train),pd.DataFrame(X_test)]).values
y = pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_test)]).values.ravel()


# In[77]:


from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_squared_error


# In[108]:


gb = GradientBoostingRegressor(random_state=42)

cv = KFold(n_splits=5, random_state=0,shuffle=True)

scores={'RMSE':[]}

for train_index, test_index in cv.split(X, y.ravel()):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        gb.fit(xtrain, ytrain)
        pred = gb.predict(xtest)
        scores['RMSE'].append(np.sqrt(mean_squared_error(ytest, pred)))

print('RMSE of the 5 splits: {}'.format(scores['RMSE']))
print('average RMSE: {}'.format(np.mean(scores['RMSE'])))


# In[90]:


print('average RMSE: {}'.format(np.mean(scores['RMSE'])))


# In[92]:


reg.feature_importances_


# In[93]:


importances = reg.feature_importances_
sorted_index=np.argsort(importances)[::-1]
x=range(len(importances))
labels=np.array(train_df.drop(['unit_id','RUL'],axis=1).columns)[sorted_index]

plt.bar(x,importances[sorted_index],tick_label=labels)

plt.xticks(rotation=90)
plt.show()


# In[47]:


from sklearn.linear_model import LinearRegression


# In[54]:


lm = LinearRegression()
lm.fit(X_train, y_train)

# predict and evaluate
y_hat_train = lm.predict(X_train)
print(np.sqrt(mean_squared_error(y_train, y_hat_train)))

y_hat_test = lm.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_hat_test)))


# In[49]:


from sklearn.ensemble import RandomForestRegressor


# In[50]:


m = RandomForestRegressor(max_depth=7,n_estimators=448,random_state=42)
m.fit(X_train, y_train)

y_hat_train = m.predict(X_train)
print(np.sqrt(mean_squared_error(y_train, y_hat_train)))

y_hat_test = m.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_hat_test)))

