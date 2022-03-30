import streamlit as st
import pandas as pd
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#page = st.selectbox("Choose your page", ["Text data preview", "Machine Learning for text data"]) 


st.write("""
# Classification for OASIS MRI Data
This app predicts the dementia based on biological index.
""")


st.sidebar.header('User Input Parameters')

def user_input_features():
    ASF = st.sidebar.slider('ASF', 0.80, 1.60, 0.88)
    Age = st.sidebar.slider('Age', 60, 99, 87)
    CDR = st.sidebar.slider('CDR', 0.0, 2.0, 0.0)
    EDUC = st.sidebar.slider('EDUC', 11.0, 23.0, 14.0)
    MMSE = st.sidebar.slider('MMSE', 4.0, 30.0,27.0)
    SES = st.sidebar.slider('SES', 0.50,5.25,2.00)
    eTIV = st.sidebar.slider('eTIV', 1100.0,2050.0,1987.0)
    gender = st.sidebar.slider('gender(should only choose between 0 and 1)', 0, 1,0)
    nWBV = st.sidebar.slider('nWBV', 0.63, 0.85,0.69)

    data = {'ASF': ASF,
            'Age': Age,
            'CDR': CDR,
            'EDUC': EDUC,
            'MMSE':MMSE,
            'SES':SES,
            'eTIV':eTIV,
            'gender':gender,
            'nWBV':nWBV}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#iris = datasets.load_iris()
#X = iris.data
#Y = iris.target

#load data
test_url = 'https://finalproject-42236-default-rtdb.firebaseio.com/test.json'
response = requests.get(test_url)
text_test = response.json()
test_t = pd.DataFrame.from_dict(text_test, orient='columns')

train_url = 'https://finalproject-42236-default-rtdb.firebaseio.com/train.json'
response = requests.get(train_url)
text_train = response.json()
train_t = pd.DataFrame.from_dict(text_train, orient='columns')

# split the data into X and y
def xy_split(dat):
    y = dat['Group']
    X = dat.loc[:, dat.columns != 'Group']
    return y, X

y_train, X_train = xy_split(train_t)
y_test, X_test = xy_split(test_t)


st.dataframe(X_train)

#Random Forest
clf = RandomForestClassifier(random_state=57)
clf.fit(X_train, y_train)


# The interactive part
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictionNB = gnb.predict(df)
prediction_probaNB = gnb.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction Using Random Forest From User Input')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Using Naive Bayes From User Input')
st.write(predictionNB)

st.subheader('Prediction Probability Using Random Forest From User Input')
st.write(prediction_proba)
st.subheader('Prediction Probability Using Naive Bayes From User Input')
st.write(prediction_probaNB)


#Using test data
t_pred = clf.predict(X_test)
tnb_pred = gnb.predict(X_test)
st.subheader('Prediction Result Using Random Forest From Test Data')
st.write(t_pred)
st.subheader('Prediction Result Using Naive Bayes From Test Data')
st.write(tnb_pred)
st.subheader('True Test Data Label')
st.write(y_test)

st.write('The accuracy score if using random forest classifier is: ',clf.score(X_test, y_test))
st.write('The accuracy score if using naive bayes classifier is: ',clf.score(X_test, y_test))





@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


csv_train = convert_df(train_t)
csv_test = convert_df(test_t)

st.download_button(
   "Press to Download Training Data",
   csv_train,
   "train.csv",
   "text/csv",
   key='download-csv'
)

st.download_button(
   "Press to Download Testing Data",
   csv_test,
   "test.csv",
   "text/csv",
   key='download-csv'
)