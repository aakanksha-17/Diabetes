#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[17]:


#pip install streamlit_jupyter


# In[19]:



import numpy as np
import pickle 
import streamlit as st
#sp = StreamlitPatcher()
#sp.jupyter()  # register patcher with streamlit
from sklearn.preprocessing import StandardScaler


# In[3]:


loaded_model=pickle.load(open('C:/Users/HP/trained_model.sav','rb'))


# In[6]:


#creating a function for prediciton : 
def diabetes_prediction(input_data):
#whatever data we get from user - we put it in input data
    input_data_as_numpy_array=np.asarray(input_data)

    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1) 


    #input_data_standard=scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction) #this is a list and first value is our ans

    if (prediction[0] == 0 ): 
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic' 


# In[ ]:


def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    


# In[ ]:


#jupytext --to py myapp.ipynb

