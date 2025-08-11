# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# — Load model yang sudah kamu latih dan simpan, misalnya model_overweight.pkl

with open('model.pkl', 'rb') as file:
  model = pickle.load(file)

st.set_page_config(page_title="Overweight Prediction", layout="centered")

st.title("🏋️ Overweight Risk Prediction")

st.write("""
Masukkan data sesuai kolom berikut,  
lalu klik Predict untuk melihat prediksi risiko obesitas.
""")

def run():
    with st.form(key="input_form"):
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=29)
        height = st.number_input("Height (m)", value=1.78, format="%.2f")
        weight = st.number_input("Weight (kg)",min_value=1, value=60)
        family = st.selectbox("Family History With Overweight?", options=["yes", "no"])
        favc = st.selectbox("Frequent High Calorie Foods (FAVC)?", options=["yes", "no"])
        fcvc = st.number_input("Daily Vegetable Consumption (FCVC)", min_value=1, value=3)
        ncp = st.number_input("Number of Main Meals (NCP)", min_value=1, value=2)
        caec = st.selectbox("Eat Any Food Between Meals (CAEC)", options=["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Smoke?", options=["yes", "no"])
        ch2o = st.number_input("Daily Water Intake (L)", min_value=0.1, value=1.0, format="%.1f")
        scc = st.selectbox("Calories Monitoring (SCC)?", options=["yes", "no"])
        faf = st.number_input("Physical Activity Frequency", min_value=0, value=2)
        tue = st.number_input("Time Using Tech at EOD (TUE) hours", min_value=0, value=0)
        calc = st.selectbox("Consumption of Alcohol (CALC)?", options=["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation Mode (MTRANS)", options=["Automobile", "Motorbike", "Public_Transportation", "Walking", "Bike"])

        submitted = st.form_submit_button("Predict")

    data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }
    data = pd.DataFrame([data])

    link1 = ""
    link2 = ""


    if submitted:

        prediction2 = model.predict(data)

        idx = int(prediction2[0])

    # Ubah ke label dengan if‑else
        if idx == 0:
            label = 'Insufficient_Weight'
            link1 = "[cara terhindar dari obesitas](https://ayosehat.kemkes.go.id/cegah-obesitas-dengan-5-langkah)"
        elif idx == 1:
            label = 'Normal_Weight'
            link1 = "[cara terhindar dari obesitas](https://ayosehat.kemkes.go.id/cegah-obesitas-dengan-5-langkah)"
        elif idx == 2:
            label = 'Overweight_Level_I'
            link1 = "[Olahraga yang bisa mengatasi berat berlebih](https://www.halodoc.com/artikel/5-olahraga-untuk-bantu-atasi-obesitas-pada-orang-dewasa)"
            link2 = "[Cara Hidup Sehat untuk Mencegah Obesitas](https://hellosehat.com/nutrisi/obesitas/kebiasaan-makan-cegah-obesitas/)"
        elif idx == 3:
            label = 'Overweight_Level_II'
            link1 = "[Olahraga yang bisa mengatasi berat berlebih](https://www.halodoc.com/artikel/5-olahraga-untuk-bantu-atasi-obesitas-pada-orang-dewasa)"
            link2 = "[Cara Hidup Sehat untuk Mencegah Obesitas](https://hellosehat.com/nutrisi/obesitas/kebiasaan-makan-cegah-obesitas/)"
        elif idx == 4:
            label = 'Obesity_Type_I'
        elif idx == 5:
            label = 'Obesity_Type_II'
        elif idx == 6:
            label ='Obesity_Type_III'

        st.write(f'# Predicted risiko obesitas: **{label}**')
        links_md = f"{link1}  \n{link2}  \n"
        st.markdown(links_md, unsafe_allow_html=True)

if __name__ == "__main__":
    run()

    