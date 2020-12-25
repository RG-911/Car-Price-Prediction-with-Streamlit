
import pandas as pd
import numpy as np
import datetime
import streamlit as st


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# UPLOAD THE DATASET
path = '/Users/LAMAN/Google Drive/Data Science/DataProfessor/car_data.csv'
cars = pd.read_csv(path)
cars.dropna(inplace=True)


#DEPLOY THE ML MODEL
st.write("""
# SIMPLE CAR PRICE PREDICTION APP

With help of this app you can find  either right price to sell your old car or buy a new one.

""")

st.sidebar.header('**INPUT PARAMETRS**')

def set_input_parameters():
    brand = st.sidebar.text_input('Brand')
    model = st.sidebar.text_input('Model')
    fuel_type = st.sidebar.selectbox('Fuel Type', ('Benzin', 'Diesel'))
    transmission = st.sidebar.selectbox('Transmission', ('Schaltgetriebe', 'Automatik'))
    condition = st.sidebar.selectbox('Condition', ('Gebraucht', 'Neu'))
    #start_date = st.sidebar.date_input('Star_date', datetime.date(2010, 1, 1))
    first_registration_year = st.sidebar.text_input('First registration year')
    milleage_km = st.sidebar.slider('Milleage', 0, 100000, 45000)
    engine_power_PS = st.sidebar.slider('Engine Power', 5, 1000, 100)

    parameters = {'brand': brand,
                'model': model,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'condition': condition,
                'first_registration_year': first_registration_year,
                'milleage_km': milleage_km,
                'engine_power_PS': engine_power_PS
                }

    features = pd.DataFrame(parameters, index=[0])
    return features

df = set_input_parameters()

st.subheader('Car Parameters')
st.write(df)

cars['Brand'] = cars['Brand'].astype('category')
cars['Model'] = cars['Model'].astype('category')
cars['Conditon'] = cars['Conditon'].astype('category')
cars['Fuel type'] = cars['Fuel type'].astype('category')
cars['Transmission'] = cars['Transmission'].astype('category')

#SPLIT THE DATA SET
X = cars.drop(columns=['Price (EUR)'], axis = 1)
y = cars['Price (EUR)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

#BUILD MODEL
pipe_rm = make_pipeline(OneHotEncoder(handle_unknown='ignore') ,RandomForestRegressor())
n_estimators_range = np.arange(10,210,10)
param_grid = dict( randomforestregressor__n_estimators=n_estimators_range)
grid = GridSearchCV(estimator=pipe_rm, param_grid=param_grid, cv=5)

#FIT THE MODEL
grid.fit(X_test, y_test)

#GET THE PREDICTION
st.subheader('The predicted price of the car')
pred_rm=grid.predict(df)
st.write(pred_rm)


#PREDICTION TECHNICAL DETAILS
mae = metrics.mean_absolute_error(y_test, pred_rm)
mse = metrics.mean_squared_error(y_test, pred_rm)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_rm))
score = metrics.r2_score(y_test, pred_rm)*100
probability = grid.predict_proba(df)

st.subheader('Model evaluation parameters')
st.write(mae)
st.write(mse)
st.write(rmse)
st.write(probability)




