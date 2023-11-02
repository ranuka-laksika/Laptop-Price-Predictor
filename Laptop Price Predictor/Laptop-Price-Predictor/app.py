import tkinter as tk


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
    #print(str(model)+ ' --> ' +str(acc))

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




Display=[0,0]
Company=[0,0,0,0,0,0,0,0,0]
Type=[0,0,0,0,0,0]
OS=[0,0,0,0]
CPU=[0,0,0,0,0]
GPU=[0,0,0]
#from Project import set_os
# Function to calculate and display the sum


def predict_value():
    ram = float(ram_entry.get())
    weight = float(weight_entry.get())
    touchscreen = touchscreen_var.get()
    ips = ips_var.get()
    if touchscreen:
        Display[0]=1
    if ips:
        Display[1]=1
    company_type = company_type_var.get()
    if company_type=="Acer":
        Company[0]=1
    elif company_type=="Apple":
        Company[1]=1
    elif company_type=="Asus":
        Company[2]=1
    elif company_type=="Dell":
        Company[3]=1
    elif company_type=="HP":
        Company[4]=1
    elif company_type=="Lenovo":
        Company[5]=1
    elif company_type=="MSI":
        Company[6]=1
    elif company_type=="Other":
        Company[7]=1
    else:
        Company[8]=1

    type = type_var.get()
    if type=="2 in 1 Convertible":
        Type[0]=1
    elif type=="Gaming":
        Type[1]=1
    elif type=="Netbook":
        Type[2]=1
    elif type=="Notebook":
        Type[3]=1
    elif type=="Ultraboo":
        Type[4]=1
    else:
        Type[5]=1
    
    OS1 = OS_var.get()
    if OS1=="Linux":
        OS[0]=1
    elif OS1=="Mac":
        OS[1]=1
    elif OS1=="Other":
        OS[2]=1
    else:
        OS[3]=1 
    
    CPU1 = CPU_var.get()
    if CPU1=="AMD":
        CPU[0]=1
    elif CPU1=="Intel Core i3":
        CPU[1]=1
    elif CPU1=="Intel Core i5":
        CPU[2]=1
    elif CPU1=="Intel Core i7":
        CPU[3]=1
    else:
        CPU[4]=1

    GPU1 = GPU_var.get()
    if GPU1=="AMD":
        GPU[0]=1
    elif GPU1=="Intel":
        GPU[1]=1
    else:
       GPU[2]=1

    array=[ram]+[weight]+Display+Company+Type+OS+CPU+GPU
    value=(best_model.predict([array]))
    print(array)
    #return str(value[0])
    EurotoLKR=340.75
    result_label.config(text=f"Estimated Price: LKR = {EurotoLKR*value[0]}")

# Function to reset input fields and result label
def reset_inputs():
    ram_entry.delete(0, tk.END)
    weight_entry.delete(0, tk.END)
    touchscreen_var.set(False)
    ips_var.set(False)
    result_label.config(text="")
# Create the main application window
app = tk.Tk()
app.title("myApp")

# Title label at the top
title_label = tk.Label(app, text="LAPTOP PRICE PREDICTOR", font=("Arial", 16))
title_label.grid(row=0, column=0, columnspan=2)  # Span two columns for title

# RAM input
ram_label = tk.Label(app, text="RAM (GB)", font=("Arial", 12))
ram_label.grid(row=1, column=0)
ram_entry = tk.Entry(app)
ram_entry.grid(row=1, column=1)

# Weight input
weight_label = tk.Label(app, text="Weight (kg)", font=("Arial", 12))
weight_label.grid(row=2, column=0)
weight_entry = tk.Entry(app)
weight_entry.grid(row=2, column=1)

# Screen type selection
# Touchscreen Checkbutton
touchscreen_label = tk.Label(app, text="Touchscreen", font=("Arial", 12))
touchscreen_label.grid(row=3, column=0)
touchscreen_var = tk.BooleanVar()
touchscreen_checkbutton = tk.Checkbutton(app, text="Yes", variable=touchscreen_var)
touchscreen_checkbutton.grid(row=3, column=1)

# IPS Checkbutton
ips_label = tk.Label(app, text="IPS", font=("Arial", 12))
ips_label.grid(row=4, column=0)
ips_var = tk.BooleanVar()
ips_checkbutton = tk.Checkbutton(app, text="Yes", variable=ips_var)
ips_checkbutton.grid(row=4, column=1)

# Company name selection
company_type_label = tk.Label(app, text="Company Name", font=("Arial", 12))
company_type_label.grid(row=5, column=0)

company_type_var = tk.StringVar()
company_type_var.set("SELECT")  # Default selection
company_type_option = tk.OptionMenu(app, company_type_var, "Acer", "Apple","Asus","Dell","HP","Lenovo","MSI","Toshiba","Other")
company_type_option.grid(row=5, column=1)

# Type selection
type_label = tk.Label(app, text="Type Name", font=("Arial", 12))
type_label.grid(row=6, column=0)

type_var = tk.StringVar()
type_var.set("SELECT")  # Default selection
type_option = tk.OptionMenu(app, type_var, "2 in 1 Convertible", "Gaming","Netbook","Notebook","Ultrabook","Workstation")
type_option.grid(row=6, column=1)

# OS type selection
OS_label = tk.Label(app, text="OS", font=("Arial", 12))
OS_label.grid(row=7, column=0)

OS_var = tk.StringVar()
OS_var.set("SELECT")  # Default selection
OS_option = tk.OptionMenu(app, OS_var, "Linux", "Mac","Windows","Other")
OS_option.grid(row=7, column=1)

# CPU Type selection
CPU_label = tk.Label(app, text="CPU", font=("Arial", 12))
CPU_label.grid(row=8, column=0)

CPU_var = tk.StringVar()
CPU_var.set("SELECT")  # Default selection
CPU_option = tk.OptionMenu(app, CPU_var, "AMD", "Intel Core i3","Intel Core i5","Intel Core i7","Other")
CPU_option.grid(row=8, column=1)

# GPU Type selection
GPU_label = tk.Label(app, text="GPU", font=("Arial", 12))
GPU_label.grid(row=9, column=0)

GPU_var = tk.StringVar()
GPU_var.set("SELECT")  # Default selection
GPU_option = tk.OptionMenu(app, GPU_var, "AMD", "Intel","Nvidia")
GPU_option.grid(row=9, column=1)

# Calculate button
calculate_button = tk.Button(app, text="Predict Price", font=("Arial", 13), command=predict_value)
calculate_button.grid(row=10, column=0, columnspan=2)  # Span two columns for button

# Result label
result_label = tk.Label(app, text="")
result_label.grid(row=11, column=0, columnspan=2)  # Span two columns for result

# Reset button
reset_button = tk.Button(app, text="Reset", command=reset_inputs)
reset_button.grid(row=12, column=0, columnspan=2)  # Span two columns for result

# Start the main application loop
app.mainloop()





