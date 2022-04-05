from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pickle
from scipy.sparse import hstack
import math

st.title("FINANCE PRO's")
st.title("Loan Grant Prediction")
st.subheader("By Sudeesh Reddy- Vijay Madhav - Sai Deep chand ")







st.subheader('Enter your details:')

with st.form("my_form"):
    Gender = st.selectbox('Gender:',('Male','Female'))
    
    Married = st.selectbox('Married:',('Yes','No'))
    
    Dependents = st.selectbox('Dependents:',('Zero','One','Two','Three'))
    
    Education = st.selectbox('Education:',('Graduate','Not Graduate'))
    
    Self_Employed = st.selectbox('Self_Employed:',('Yes','No'))
    
    ApplicantIncome = st.number_input('ApplicantIncome:')

    CoapplicantIncome = st.number_input('CoapplicantIncome:')
    
    LoanAmount = st.number_input('LoanAmount:')
    
    Loan_Amount_Term = st.number_input('Loan_Amount_Term:')
    
    Credit_History = st.selectbox('Credit_History: (1:all debts paid, 0: not paid)',('One','Zero'))
    
    Property_Area = st.selectbox('Property_Area:',('Urban','Semiurban','Rural'))

    submitted = st.form_submit_button("Predict")
    

    Gender = np.array(Gender).reshape(1, -1)
    Married=np.array(Married).reshape(1, -1)
    Dependents=np.array(Dependents).reshape(1, -1)
    Education=np.array(Education).reshape(1, -1)
    Self_Employed=np.array(Self_Employed).reshape(1, -1)
    ApplicantIncome=np.array(round(ApplicantIncome)).reshape(1, -1)
    CoapplicantIncome=np.array(round(CoapplicantIncome)).reshape(1, -1)
    LoanAmount=np.array(round(LoanAmount)).reshape(1, -1)
    Loan_Amount_Term=np.array(round(Loan_Amount_Term)).reshape(1, -1)
    Credit_History=np.array(Credit_History).reshape(1, -1)
    Property_Area=np.array(Property_Area).reshape(1, -1)

    model= open('final_model.pkl', 'rb')
    model = pickle.load(model)

    ApplicantIncome_vectorizer = pickle.load(open('ApplicantIncome_vectorizerl.pkl', 'rb'))

    CoapplicantIncome_vectorizer = pickle.load(open('CoapplicantIncome.pkl', 'rb'))

    LoanAmount_vectorizer = pickle.load(open('LoanAmount.pkl', 'rb'))

    Loan_Amount_Term_vectorizer = pickle.load(open('SLoan_Amount_Term.pkl', 'rb'))

    Gender_vectorizer = pickle.load(open('Gender_vectorizerl.pkl', 'rb'))

    Married_vectorizer = pickle.load(open('Married_vectorizer.pkl', 'rb'))

    Education_vectorizer = pickle.load(open('Education_vectorizer.pkl', 'rb'))

    Self_Employed_vectorizer = pickle.load(open('Self_Employed_vectorizerl.pkl', 'rb'))

    Property_Area_vectorizer = pickle.load(open('Property_Area_vectorizer.pkl', 'rb'))

    Dependents_vectorizer = pickle.load(open('Dependents_vectorizer.pkl', 'rb'))

    Credit_History_vectorizer = pickle.load(open('Credit_History_vectorizer.pkl', 'rb'))

    if submitted:
        ApplicantIncome_vectorizer=ApplicantIncome_vectorizer.transform(ApplicantIncome)
        CoapplicantIncome_vectorizer=CoapplicantIncome_vectorizer.transform(CoapplicantIncome) 
        LoanAmount_vectorizer=LoanAmount_vectorizer.transform(LoanAmount)
        Loan_Amount_Term_vectorizer=Loan_Amount_Term_vectorizer.transform(Loan_Amount_Term)
        Gender_vectorizer=Gender_vectorizer.transform(Gender.ravel())
        Married_vectorizer=Married_vectorizer.transform(Married.ravel())
        Education_vectorizer=Education_vectorizer.transform(Education.ravel())
        Self_Employed_vectorizer=Self_Employed_vectorizer.transform(Self_Employed.ravel())
        Property_Area_vectorizer=Property_Area_vectorizer.transform(Property_Area.ravel())
        Dependents_vectorizer=Dependents_vectorizer.transform(Dependents.ravel())
        Credit_History_vectorizer=Credit_History_vectorizer.transform(Credit_History.ravel())
        input = hstack((Credit_History_vectorizer,
                        Dependents_vectorizer,
                        Loan_Amount_Term_vectorizer,
                        LoanAmount_vectorizer,
                        CoapplicantIncome_vectorizer,
                        ApplicantIncome_vectorizer,
                        Gender_vectorizer,
                        Married_vectorizer,
                        Education_vectorizer,
                        Self_Employed_vectorizer,
                        Property_Area_vectorizer,)).tocsr()
        prediction = model.predict(input)
        pre= model.predict_proba(input)
        pre=pre[0]
        pre=pre[1]
        pre = math.trunc(pre*100)
        pre = str(pre)
        c ="Chances of Loan Getting Approved :" + pre + "%"

        if str(prediction[0])=="N":
            st.warning("Rejected")
            st.warning(c)
        elif str(prediction[0])=="Y": 
            st.success("Approved")
            st.success(c)



