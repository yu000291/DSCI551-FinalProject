import streamlit as st
import pandas as pd
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#page = st.selectbox("Choose your page", ["Text data preview", "Machine Learning for text data"]) 


st.write("""
# Classification for OASIS MRI Data
This app predicts the dementia based on current biological index and situation.
""")
st.subheader('How it works?')
st.write('You can fiddle with the input data on sidebar, classifiers will predict the results based on your input!')
st.write('Your input data will also be shown below in the User Input Parameters box.')
st.write('The classifiers are trained with training data. Thus, besides the result from your input data, it will also show the results from testing data.')
st.write('Here we provide two classifiers to learn the input data and build two models based on them: Random Forest Classifier and Naive Bayes Classifier')
st.write('Random Forest Classifier: "A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting." (scikit learn)')
st.write('Naive Bayes Classifier: "Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable." (scikit learn)' )
st.subheader('Prediction Problem')
st.write('How we can use known patient information to predict the probability of dementia using classification for this patient?')

st.sidebar.header('User Input Parameters')

def user_input_features():
    ASF = st.sidebar.slider('ASF(Atlas Scaling Factor)', 0.80, 1.60, 0.88)
    Age = st.sidebar.slider('Age', 60, 99, 87)
    CDR = st.sidebar.slider('CDR(Clinical Dementia Rating)', 0.0, 2.0, 0.0)
    EDUC = st.sidebar.slider('EDUC(Years of Education)', 11.0, 23.0, 14.0)
    MMSE = st.sidebar.slider('MMSE(Mini-Mental State Exam)', 4.0, 30.0,27.0)
    SES = st.sidebar.slider('SES(Social Economic Status)', 0.50,5.25,2.00)
    eTIV = st.sidebar.slider('eTIV(Estimated Total Inctracranial Volume)', 1100.0,2050.0,1987.0)
    gender = st.sidebar.slider('gender(0-Male, 1-Female)', 0, 1,0)
    nWBV = st.sidebar.slider('nWBV(Normalize Whole Brain Volume)', 0.63, 0.85,0.69)

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


#st.dataframe(X_train)

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

st.subheader('Prediction From User Input')
st.write('Using Random Forest')
st.write(prediction)
st.write('Based on your input data, the patient is diagnosed as: ', prediction[0])
st.write('Using Naive Bayes')
st.write(predictionNB)
st.write('Based on your input data, the patient is diagnosed as: ', predictionNB[0])

st.subheader('Prediction Probability From User Input')
st.write('How to understand the result?')
st.write('The (0,0) box shows the probability of being diagnosed as demented; the (1,0) box shows the probability of being diagnosed as nondemented')
st.write('Using Random Forest Classifier')
st.write(prediction_proba)
st.write(' Using Naive Bayes')
st.write(prediction_probaNB)


#Using test data
t_pred = clf.predict(X_test)
tnb_pred = gnb.predict(X_test)


st.subheader('Prediction Result From Test Data')
st.write('Using Random Forest')
st.write(t_pred)
#st.subheader('Prediction Result Using Naive Bayes From Test Data')
st.write('Using Naive Bayes')
st.write(tnb_pred)
st.subheader('True Test Data Label')
st.write(y_test)

st.write('The accuracy score if using random forest classifier is: ',clf.score(X_test, y_test))
st.write('The accuracy score if using naive bayes classifier is: ',clf.score(X_test, y_test))




#uploading data
urlpred_t = 'https://finalproject-42236-default-rtdb.firebaseio.com/rlt_random_forest.json'
urlpred_tnb = 'https://finalproject-42236-default-rtdb.firebaseio.com/rlt_naive_bayes.json'

t_j = json.dumps(t_pred.tolist())
tnb_j = json.dumps(tnb_pred.tolist())

r1 = requests.put(urlpred_t, t_j)
r2 = requests.put(urlpred_tnb, tnb_j)


#@st.cache
#def convert_df(df):
#   return df.to_csv().encode('utf-8')


#csv_train = convert_df(train_t)
#csv_test = convert_df(test_t)

#st.download_button(
#   "Press to Download Training Data",
#   csv_train,
#   "train.csv",
#   "text/csv",
#   key='download-csv'
#)

#st.download_button(
#   "Press to Download Testing Data",
#   csv_test,
#   "test.csv",
#   "text/csv",
#   key='download-csv'
#)
