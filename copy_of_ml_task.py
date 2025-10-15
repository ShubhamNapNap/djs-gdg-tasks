
import streamlit as st
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 200)
df = pd.read_csv("f1_dnf.csv")
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
grid = st.sidebar.number_input("Grid Position", min_value=1, max_value=20, value=1)
position_order = st.sidebar.number_input("Position Order", min_value=1, max_value=20, value=1)
driver_age = st.sidebar.number_input("Driver Age at Race", min_value=18, max_value=50, value=25)
constructor = st.sidebar.selectbox("Constructor", options=df['constructorRef_encoded'].unique())
driver = st.sidebar.selectbox("Driver", options=df['driverRef_encoded'].unique())
input_data = np.array([[grid, position_order, driver_age, constructor, driver]])

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.write("DNF")
    else:
        st.write("Finished")

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