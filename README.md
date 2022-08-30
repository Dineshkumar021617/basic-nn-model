# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## Neural Network Model
![Screenshot (389)](https://user-images.githubusercontent.com/75243072/187078981-2aafe51a-eaff-4dd6-a902-e6f6bc567333.png)

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Dataset1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df.dtypes
df=df.astype({'A':'int'})
df=df.astype({'B':'float'})
df.dtypes
from sklearn.model_selection import train_test_split
X=df[['A']].values
Y=df[['B']].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=20)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai_brain = Sequential([
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_train_scaled,y=y_train,epochs=20000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test_scaled=scaler.transform(x_test)
ai_brain.evaluate(x_test_scaled,y_test)
input=[[185]]
input_scaled=scaler.transform(input)
ai_brain.predict(input_scaled)
```

## Dataset Information
![image](https://user-images.githubusercontent.com/75234807/187390697-6f32e1ed-7380-45cb-bc57-f385e7c6371f.png)


## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot (246)](https://user-images.githubusercontent.com/75234807/187431193-e0aa687b-5596-45d3-80c2-5527b21f6d68.png)


### Test Data Root Mean Squared Error
![Screenshot (245)](https://user-images.githubusercontent.com/75234807/187391778-73df420a-ba69-4e8c-9d99-a86f8980da85.png)


### New Sample Data Prediction
![Screenshot (244)](https://user-images.githubusercontent.com/75234807/187391740-d745a676-5bde-42d6-a375-778c227f7340.png)


## RESULT:
Thus the Neural network for Regression model is Implemented successfully.
