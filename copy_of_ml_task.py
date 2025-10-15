
import streamlit as st
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 200)
df = pd.read_csv("f1_dnf.csv")
dt = pd.read_csv("f1_dnf.csv")
df = df[[
    'resultId', 'year', 'round', 'grid', 'positionOrder', 'points', 'laps', 'dob', 'date', 'target_finish',
    'driverRef', 'forename', 'surname', 'nationality_x',
    'constructorRef', 'name', 'nationality_y',
    'circuitRef', 'name_y', 'location', 'country', 'lat', 'lng', 'alt'
]].copy()
df['dob'] = pd.to_datetime(df['dob'])
df['date'] = pd.to_datetime(df['date'])
df['driver_age_at_race'] = (df['date'] - df['dob']).dt.days / 365.25
df['driver_name'] = df['forename'] + ' ' + df['surname']
df = df.drop(columns=['forename', 'surname'])
df = df.loc[~df.duplicated(subset=["year", "round", "driverRef", "constructorRef"])]\
.reset_index(drop=True).copy()
df.head()

df = df.drop(columns=['resultId', 'dob','positionOrder', 'date', 'driver_name','name', 'name_y','location','circuitRef', 'country', 'alt','nationality_y','nationality_x','laps','points'])
df.head()

df.isna().sum()

from sklearn.preprocessing import LabelEncoder

le_constructor = LabelEncoder()
df['constructorRef_encoded'] = le_constructor.fit_transform(df['constructorRef'])

le_driver = LabelEncoder()
df['driverRef_encoded'] = le_driver.fit_transform(df['driverRef'])
df.drop(columns=['constructorRef', 'driverRef'], inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x = df.drop(columns=['target_finish'])
y = df['target_finish']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

model = RandomForestClassifier(n_estimators=700, max_depth=12, random_state=1)
model.fit(X_train, y_train)

st.title('Formula 1 Finish Prediction')
lookup_table = pd.read_csv("f1_dnf.csv")[['driverRef','constructorRef','lat','lng']].drop_duplicates()

year = st.number_input("Year", min_value=1950, max_value=2030, value=2023)
round_num = st.number_input("Round", min_value=1, max_value=30, value=1)
grid = st.number_input("Grid", min_value=1, max_value=50, value=1)

driver_choice = st.selectbox("Driver", sorted(lookup_table['driverRef'].unique()))
constructor_choice = st.selectbox("Constructor", sorted(lookup_table['constructorRef'].unique()))
lat = st.number_input("Latitude", value=float(lookup_table['lat'].mean()))
lng = st.number_input("Longitude", value=float(lookup_table['lng'].mean()))
driver_age = st.number_input("Driver Age at Race (years)", min_value=16.0, max_value=60.0, value=28.0)

driver_encoded = le_driver.transform([driver_choice])[0]
constructor_encoded = le_constructor.transform([constructor_choice])[0]

input_data = pd.DataFrame([{
    "year": year,
    "round": round_num,
    "grid": grid,
    "lat": lat,
    "lng": lng,
    "driver_age_at_race": driver_age,
    "constructorRef_encoded": constructor_encoded,
    "driverRef_encoded": driver_encoded
}])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.write("DNF")
    else:
        st.write("Finished")

circuits = dt[['circuitRef','lat','lng']].drop_duplicates().reset_index(inplace = true)
st.dataframe(circuits)
#y_pred = model.predict(X_test)
#y_pred_proba = model.predict_proba(X_test)[:, 1]

'''from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)'''

'''print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
print(f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))'''
