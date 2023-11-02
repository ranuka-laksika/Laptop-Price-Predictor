
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
url="laptop_price.csv"
data = pd.read_csv(url, encoding='latin-1')
data.head()

# we have to change the column type to numerical types
data['Ram']=data['Ram'].str.replace('GB','').astype('int32')
data['Weight']=data['Weight'].str.replace('kg','').astype('float32')

# Now we have to do one-hot encoding. But the no of Companies is much higher. So we want reduce it.
def add_company(inpt):
    if inpt == 'Samsung' or inpt =='Razer' or inpt == 'Mediacom' or inpt == 'Microsoft' or inpt == 'Xiaomi' or inpt == 'Vero' or inpt == 'Chuwi' or inpt == 'Google' or inpt == 'Fujitsu' or inpt == 'LG' or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt
# Add this method to the Company column
data['Company']=data['Company'].apply(add_company)

# Get some features from ScreenResolution column
# Build new column from these features
data['Touchscreen']=data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
data['Ips']=data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

data['cpu_name']=data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'
data['cpu_name']=data['cpu_name'].apply(set_processor)

data['gpu_name']=data['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))

# ARM Gpu count is 1 we can delete entire Row
data=data[data['gpu_name']!='ARM']

def set_os(inpt):
    if inpt== 'Windows 10' or inpt== 'Windows 7' or inpt== 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'
data['OpSys']=data['OpSys'].apply(set_os)

# Delete the unwanted columns
data = data.drop(columns=['laptop_ID','Inches','Product','ScreenResolution','Cpu','Gpu'])

# Get numerical values by one-hot encoding in pandas
data=pd.get_dummies(data)



# Building the model

# Drop the Price_euros column from the dataset as y
X=data.drop('Price_euros', axis=1)
y=data['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Since we have planned to train the model through different algorithm 
# Find the model accuracy
def model_acc(model):
    model.fit(X_train, y_train)
    acc=model.score(X_test, y_test)
    print(str(model)+ ' --> ' +str(acc))

# Build a models
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_acc(rf)

# Since, highest score value has been given by RandomForestRegressor, so now we can use this model 

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10, 50, 100],
              'criterion':['squared_error','absolute_error','poisson']}

grid_obj = GridSearchCV(estimator=rf, param_grid=parameters)

grid_fit = grid_obj.fit(X_train, y_train)

best_model = grid_fit.best_estimator_

best_model.score(X_test, y_test)



x=(best_model.predict([[8,1.3,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0]]))
print(x[0])
# Save the model
# import pickle
# with open('predictor.pickle','wb') as file:
#     pickle.dump(best_model,file)









