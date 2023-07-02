import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('gb_class.sav', 'rb'))


# creating a function for Prediction
def prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Displeased'
    else:
        return 'Satisfied'


def main():
    # giving a title
    st.title('Customer Churn Prediction Web App')
    st.markdown('<h1 style="font-size:large;">Enter numeric data only!. Use examples.</h1>', unsafe_allow_html=True)

    # getting the input data from the user
    SeniorCitizen = st.text_input('Senior Citizen (example >>>	Yes=1, No=0)')
    Partner = st.text_input('Partner (example >>> Yes=1, No=0)')
    Dependents = st.text_input('Dependents (example >>> Yes=1, No=0)')
    tenure = st.text_input('Tenure (example >>> 1-100)')
    StreamingTV = st.text_input('Streaming TV (example >>> No=1, Yes=2, No internet service=0)')
    StreamingMovies = st.text_input('Streaming Movies (example >>> No=1, Yes=2, No internet service=0)')
    Contract = st.text_input('Contract (example >>> Month-to-month=0, One year=1, Two year=2)')
    PaperlessBilling = st.text_input('Paperless Billing (example >>> Yes=1, No=0)')
    PaymentMethod = st.text_input('Payment Method (example >>> Electronic check=0, Mailed check=1, Bank transfer(automatic)=2,Credit card(automatic)=3)')
    MonthlyCharges = st.text_input('Monthly Charges (example >>> 1.00-120.00)')
    TotalCharges = st.text_input('Total Charges (example >>> 15.00-10000.00)')

    # code for Prediction
    churn = ''

    # creating a button for Prediction
    if st.button('Submit'):
        churn = prediction([SeniorCitizen, Partner, Dependents, tenure, StreamingTV, StreamingMovies, Contract,	PaperlessBilling,	PaymentMethod, MonthlyCharges, TotalCharges])

    with st.container():
        st.success(churn)
        st.markdown('<h4 style="font-size:large;">Reminder!!!. Model accuracy did not exceed 80%. Taking this into account, the model may show an error result.</h4>', unsafe_allow_html=True)
        st.title('My Social Links')
        st.markdown('[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sitbayevalibek)')
        st.markdown('[![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?&style=flat-square&logo=github&logoColor=white)](https://github.com/sitbayevalibek)')
        st.markdown('[![Kaggle](https://img.shields.io/badge/Kaggle-%2320BEFF.svg?&style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/sitbayevalibek)')


if __name__ == '__main__':
    main()
