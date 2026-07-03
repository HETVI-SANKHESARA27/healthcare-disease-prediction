import streamlit as st
import pickle
import numpy as np
import base64

# Load Model
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_model.sav", "rb"))
parkinsons_model =pickle.load(open("parkinsons_model.sav","rb"))
st.set_page_config(page_title="Healthcare Disease Predicion",
                   page_icon="588hospital_100778.png",
                   layout="wide",
                   initial_sidebar_state="collapsed")
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:

    st.image("hospital.png", use_container_width=True)

    st.title("🏥 Healthcare Disease Prediction System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid Username or Password")
else:

    st.sidebar.success("✅ Logged In")

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.rerun()

    selected = st.sidebar.selectbox(
        "Select Disease",
        [
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinson's Prediction"
        ]
    )
    if selected == "Diabetes Prediction":
          st.markdown("""
          <h1 style='text-align:center;color:#0E76A8;'>
          🩸 Diabetes Prediction System
          </h1>
            """, unsafe_allow_html=True)

          st.markdown("""
          <p style='text-align:center;font-size:20px;'>
          AI Powered Healthcare Prediction
          </p>
          """, unsafe_allow_html=True)
          st.write("---")


          preg = st.number_input("Pregnancies", min_value=0)
          glucose = st.number_input("Glucose")
          bp = st.number_input("Blood Pressure")
          skin = st.number_input("Skin Thickness")
          insulin = st.number_input("Insulin")
          bmi = st.number_input("BMI")
          dpf = st.number_input("Diabetes Pedigree Function")
          age = st.number_input("Age")

          if st.button("Predict"):

            data = np.array([[
              preg,
              glucose,
              bp,
              skin,
              insulin,
              bmi,
              dpf,
              age
            ]])
            prediction = diabetes_model.predict(data)

            if prediction[0] == 1:
                st.error("⚠️ Diabetic")
            else:
                st.success("✅ Non-Diabetic")
          st.write("Model Accuracy:78%")

    elif selected == "Heart Disease Prediction":

          st.title("❤️ Heart Disease Prediction System")

          age = st.number_input("Age")
          sex = st.number_input("Sex (1=Male, 0=Female)")
          cp = st.number_input("Chest Pain Type")
          trestbps = st.number_input("Resting Blood Pressure")
          chol = st.number_input("Cholesterol")
          fbs = st.number_input("Fasting Blood Sugar")
          restecg = st.number_input("Rest ECG")
          thalach = st.number_input("Max Heart Rate")
          exang = st.number_input("Exercise Induced Angina")
          oldpeak = st.number_input("Old Peak")
          slope = st.number_input("Slope")
          ca = st.number_input("Major Vessels")
          thal = st.number_input("Thal")

          if st.button("Predict Heart Disease"):
            data = np.array([[age, sex, cp, trestbps, chol,
                          fbs, restecg, thalach,
                          exang, oldpeak, slope,
                          ca, thal
                          
                        ]])

            prediction = heart_model.predict(data)

            if prediction[0] == 1:
               st.error("⚠️ Heart Disease Detected")
            else:
               st.success("✅ No Heart Disease")
          st.write("Model Accuracy:80%")
    elif selected == "Parkinson's Prediction":

        st.title("🧠 Parkinson's Disease Prediction")

        fo = st.number_input("MDVP:Fo(Hz)")
        fhi = st.number_input("MDVP:Fhi(Hz)")
        flo = st.number_input("MDVP:Flo(Hz)")
        jitter = st.number_input("Jitter (%)")
        shimmer = st.number_input("Shimmer")
        nhr = st.number_input("NHR")
        hnr = st.number_input("HNR")
        rpde = st.number_input("RPDE")
        dfa = st.number_input("DFA")
        spread1 = st.number_input("Spread1")
        spread2 = st.number_input("Spread2")
        ppe = st.number_input("PPE")

        if st.button("Predict Parkinson's"):

          data = np.array([[
            fo, fhi, flo,
            jitter, shimmer,
            nhr, hnr,
            rpde, dfa,
            spread1, spread2,
            ppe
           ]])

          prediction = parkinsons_model.predict(data)

          if prediction[0] == 1:
            st.error("⚠️ Parkinson's Disease Detected")
          else:
            st.success("✅ Healthy")
        st.write("Model Accuracy:87%")



st.markdown("_____")
st.markdown("Developed By HETVI")



