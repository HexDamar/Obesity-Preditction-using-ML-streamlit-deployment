import prediction, eda
import streamlit as st


# "with" notation
with st.sidebar:
    st.write ('# Page Navigation')

    page = st.selectbox('pilih halaman',
                        ['EDA','PredictRating'])
    
    st.write('# About')
    st.markdown('Program yang akan dibuat kali ini, akan menentukan apakah seseorang berpotensi terkena obesitas ( beserta type obesitasnya ) atau tidak')

if page == 'EDA':
    eda.run()

else:
    prediction.run()