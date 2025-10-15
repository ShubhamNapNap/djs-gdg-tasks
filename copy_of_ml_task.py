
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

year = st.number_input("Year", min_value=1950, max_value=2030, value=2025)
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

if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if prediction[0] == 1:
        st.write("DNF")
    else:
        st.write("Finished")

st.title("Circuit-lat-lng")
dt = dt[['circuitRef','lat','lng']].drop_duplicates().reset_index(drop=True)

st.dataframe(dt)

st.title("Model's performance")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
st.write(f'Precision: {precision_score(y_test, y_pred):.3f}')
st.write(f'Recall: {recall_score(y_test, y_pred):.3f}')
st.write(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
st.write(f'ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')


st.sidebar.write("Created by Shubham Nagpure\n")
st.sidebar.write("Provide Lat and Lng according to the circuit table given below.\n")

import matplotlib.pylab as plt
import seaborn as sns
plt.style.use("ggplot")
dl = pd.read_csv("f1_dnf.csv")
dl['finish_status'] = dl['target_finish'].map({1: 'Finished', 0: 'DNF/Retired'})
top_10_constructors = dl['constructorRef'].value_counts().head(10).index
df_top_10 = dl[dl['constructorRef'].isin(top_10_constructors)]
ax = sns.countplot(
    y='constructorRef',
    data=df_top_10,
    order=top_10_constructors,
    hue='finish_status',
    palette={'Finished': '#2ecc71', 'DNF/Retired': '#e74c3c'},
    saturation=0.8
)
plt.title('Total Finishes vs. DNFs for Top 10 Constructors (by Race Count)', fontsize=18)
plt.xlabel('Total Race Results (Count)', fontsize=14)
plt.ylabel('Constructor Reference ID', fontsize=14)
plt.tick_params(axis='y', labelsize=12)

plt.legend(title='Race Outcome', loc='lower right', fontsize=12, title_fontsize=12)

plt.tight_layout()
st.pyplot(plt)

st.sidebar.write("Total Finishes vs. DNFs for Top 10 Constructors (by Race Count) given at the end.")
