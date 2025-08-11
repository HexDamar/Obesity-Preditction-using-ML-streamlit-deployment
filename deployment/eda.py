import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np



def run():
    st.write('# Prediksi Obesitas')

    # HEADER GIF
    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://tv-fanatic-res.cloudinary.com/iu/s--dUZQFaW7--/t_full/cs_srgb,f_auto,fl_strip_profile.lossy,q_auto:420/v1489866460/fattening-food-family-guy.jpg' width='900'>
    </div>
    """, unsafe_allow_html=True)

    st.write('# Description')

    st.write('''Obesitas telah berkembang menjadi krisis kesehatan global — pada 2022, sekitar 2,5 miliar orang dewasa mengalami kelebihan berat badan, dan 890 juta di antaranya tergolong obesitas (BMI ≥ 30) The Times of India. 
                Penyakit ini bukanlah masalah kosmetik semata, melainkan kondisi kronis yang serius: pada 2021, kelebihan berat badan langsung menyebabkan 1,6 juta kematian prematur akibat penyakit tidak menular seperti diabetes, penyakit jantung, dan kanker.
                Program yang akan dibuat kali ini, akan menentukan apakah seseorang berpotensi terkena obesitas ( beserta type obesitasnya ) atau tidak.Project ini akan dibuat menggunakan algoritma Gradients Boosting Classifier dan dievaluasi dengan metrics F1 Score Macro sebagai pertimbangan.''')

    # load data
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    st.write('# Dataset')
    # menampilkan dataframe
    st.write(df)

    # membuat visualisasi (grafik)
    st.write('# Exploratory Data Analysis')

# EDA 1

    st.write('# 1. Apakah mereka yang memonitor kalori memiliki aktivitas fisik  yang berbeda dari mereka yang tidak memonitor kalori')

    df['SCC_bin'] = df['SCC'].map({'yes':1, 'no':0})
    # buat plot untuk tahu uji korelasi 
    # Create the stripplot
    fig, ax = plt.subplots()
    sns.stripplot(x='SCC_bin', y='FAF', data=df, jitter=True, alpha=0.5, ax=ax)

    # Set labels and title
    ax.set_xlabel('Monitor Kalori (SCC): 0=no, 1=yes')
    ax.set_ylabel('Aktivitas Fisik (FAF)')
    ax.set_title('Scatter SCC vs FAF')

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.write('''insight:
            setelah dilakuakan pengujian antara 2 kolom aktifitas fisik(FAF), dengan monitor kalori (SCC), didapati bahwa terdapat korelasi positif meskipun sangat lemah (0.054) tapi karena p value nya (0.000) yang berarti korelasi itu bersifat signifikan. jadi kesimpulan yang didapat adalah mereka yang memonitor kalori kalori cenderung memiliki aktifitas fisik yang lebih tinggi dibanding dengan yang tidak.
            ''')
# ===============================================================================================================================================================
#EDA 2
    st.write('# 2. Apakah mereka yang merokok memiliki tingkat obesitas yang lebih tinggi dibanding mereka yang tidak merokok?')

    # lakukan filtering memisahakan perokok dan tindak perokok
    df2 = df.query("SMOKE == 'yes' or SMOKE == 'no'")

    ct = pd.crosstab(df2['NObeyesdad'], df2['SMOKE'])

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the grouped bar chart
    ct.plot(kind='bar', rot=45, ax=ax)

    # Set labels and title
    ax.set_xlabel("Kategori Obesitas (NObeyesdad)")
    ax.set_ylabel("Jumlah Individu")
    ax.set_title("Distribusi NObeyesdad per Status Merokok")

    # Add legend
    ax.legend(title="Merokok")

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.write('''insight:
            setelah melihat bar barplot diatas. korelasi apakah mereka yang merokok memiliki tingkat obesitas yang lebih tinggi dibanding mereka yang tidak merokok sudah terjawab. tidak ada korelasi dari apakah dia mereka merokok atau tidak dengan tingkat obesitas yang dimiliki.
            ''')

#================================================================================================================================================================
#EDA 3
    st.write('# 3. apakah mereka yang keluarganya memiliki riwayat obesitas bisa terkena obesitas juga meskipun memiliki pola hidup yang sehat ')

    # Anggap kategori obesitas: Obesity_Type_I & II
    df['obesity_bin'] = df['NObeyesdad'].isin(['Obesity_Type_I', 'Obesity_Type_II']).astype(int)
    df['fam_bin'] = (df['family_history_with_overweight']=='yes').astype(int)

    ct = pd.crosstab(df['fam_bin'], df['obesity_bin'])

    a = ct.loc[1,1]
    b = ct.loc[1,0]
    c = ct.loc[0,1]
    d = ct.loc[0,0]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figsize as needed
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax)

    # Set labels and title
    ax.set_xlabel('Obesitas (0=no, 1=yes)')
    ax.set_ylabel('Riwayat Keluarga (0=no, 1=yes)')
    ax.set_title('Kontingensi Riwayat Keluarga vs Obesitas\n(SMOKE=no & SCC=yes)') # Use \n for multiline title

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.write('''insight:
            setelah dilakukan uji korelasi dan menampilkan heatmap didapati bahwa orang yang keluarganya memiliki riwayat obesitas masih beresiko lebih tinggi terkena obesitas meskipun memiliki pola hidup sehat. dari heatmap diatas terbukti bahwa 640 orang terkena dari 1722 yang memiliki family history obesitas dibanding 8 orang yang tidak memiliki obesitas.
            ''')


#================================================================================================================================================================
#EDA 4
    st.write('# 4. apakah ada korelasi antara seberapa sering minum alkohol dengan tingkat obsistas yang dimiliki ')

    # Encoding frekuensi minum alkohol (CAEC)
    mapping_caec = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'yes': 1}
    df['CAEC_num'] = df['CAEC'].map(mapping_caec)

    # Encoding status obesitas berdasarkan NObeyesdad
    # Kelompok "Obese": Obesity_Type_I, Obesity_Type_II, Obesity_Type_III
    obese_values = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    df['Obese'] = df['NObeyesdad'].apply(lambda x: 1 if x in obese_values else 0)

    # Set seaborn style
    sns.set(style='whitegrid')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create scatter plot
    sns.scatterplot(
        data=df,
        x='CAEC_num', # Use column names directly with data argument
        y='Obese',    # Use column names directly with data argument
        hue='Obese',
        palette={0: 'blue', 1: 'red'},
        alpha=0.6,
        s=60,
        ax=ax # Pass the axes object to seaborn
    )

    # Add logistic regression line
    sns.regplot(
        data=df,
        x='CAEC_num', # Use column names directly with data argument
        y='Obese',    # Use column names directly with data argument
        scatter=False,
        logistic=True, # Use logistic regression for binary outcome
        ax=ax,         # Pass the axes object to seaborn
        color='black',
        ci=None        # Do not show confidence interval
    )

    # Set labels and title
    ax.set_xlabel('Frekuensi Minum Alkohol (encoded)')
    ax.set_ylabel('Status Obesitas (0 = tidak, 1 = ya)')
    ax.set_title('Scatter Plot: Frekuensi Minum Alkohol vs Obesitas')

    # Add legend
    ax.legend(title='Obese')

    # Adjust layout
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)


    st.write('''insight: 
            dilihat dari plot distribusi diatas dan uji korelasi sebelumnya dapat disimpulkan bahwa frequensi minum alkohol tidak ada kaitannya dengan kenaikan tingkat obesitas ini bisa dilihat dari score korelasi ( - 0.25 )
            ''')










#================================================================================================================================================================
#EDA 5
    st.write('# 5. apakah ada korelasi antara Berapa banyak waktu yang Anda gunakan untuk menggunakan perangkat teknologi seperti ponsel, videogame, televisi, komputer, dan lainnya dengan tingkat obsistas ')

    #encode obesity level untuk diuji
    df['obesity_label'] = df['NObeyesdad'].astype('category').cat.codes   

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the regression plot (scatter plot with trendline)
    sns.regplot(x='TUE', y='obesity_label', data=df, scatter_kws={'alpha':0.6}, ax=ax)

    # Set labels and title
    ax.set_xlabel('Time Using Technology (jam/hari)')
    ax.set_ylabel('Kode Obesitas (0=Under, ..., 6=Obesity III)')
    ax.set_title(f"Scatter & Trendline ")

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.write('''insight: 
            dari hasil korelasi dan plot dapat disimpulkan bahwa korelasi lemah antara lamanya penggunaan teknologi dengan kenaikan tingkat obesitas. bisa dilihat dari hasil korelasi (-0.06) dan dari trendline yang disajikan (korelasi negatif). jadi kolom menggunakan teknologi tidak bisa bisa berdiri sendiri untuk dilakukan uji korelasi harus digabung dengan kolom lain misalkan konsumsi alkohol.
            ''')

#================================================================================================================================================================
#EDA 6
    # value counts of a specific column in your DataFrame.
    labels = [
        'Obesity_Type_I',
        'Obesity_Type_III',
        'Obesity_Type_II',
        'Overweight_Level_II',
        'Normal_Weight',
        'Overweight_Level_I',
        'Insufficient_Weight'
    ]
    sizes = np.array([351, 324, 297, 290, 282, 276, 267])
    # --- End of data target ---

    st.write("# 6. Diagram lingkaran ini menggambarkan distribusi kategori NObeyesdad (obesitas) yang berbeda di dalam dataset.")

    # Define explode for emphasis (e.g., pulling out the largest slice)
    explode = [0.1] + [0] * (len(labels) - 1)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8)) # Create figure and axes
    ax.pie(
        sizes,
        labels=labels,
        explode=explode,
        autopct='%1.1f%%',
        startangle=90,
        shadow=True
    )

    # Ensure the pie chart is circular
    ax.axis('equal')

    # Set title
    ax.set_title('Distribusi Kategori NObeyesdad')

    # Add legend (loc='best' tries to put it in an optimal location)
    ax.legend(loc='best')

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

#================================================================================================================================================================

    st.write('# 7. apakah mereka yang memakan kalori tinggi tapi minum banyak bisa mempengaruhi tingkat obesitas? ')

    ord_map = {"Insufficient_Weight":0,"Normal_Weight":1,"Overweight_Level_I":2,
           "Overweight_Level_II":3,"Obesity_Type_I":4,"Obesity_Type_II":5,"Obesity_Type_III":6}
    df["N_ob"] = df["NObeyesdad"].map(ord_map)
    df["F"] = df["FAVC"].map({"no":0,"yes":1})
    df["C"] = df["CH2O"].astype(float)


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # Adjust figsize as needed

    # Plot 1: Scatter plot
    sns.scatterplot(x="C", y="N_ob", data=df, ax=ax1)
    ax1.set_title(f"C vs N_ob)")
    ax1.set_xlabel("Calorie Consumption ('C')")
    ax1.set_ylabel("Obesity Code ('N_ob')")


    # Plot 2: Boxplot
    sns.boxplot(x="FAVC", y="N_ob", data=df, ax=ax2)
    ax2.set_title(f"FAVC vs N_ob")
    ax2.set_xlabel("Consumes High-Caloric Food (FAVC)")
    ax2.set_ylabel("Obesity Code ('N_ob')")
    # Optionally, set x-tick labels for FAVC if it's 0/1
    ax2.set_xticklabels(['No', 'Yes'])

    # Adjust layout to prevent overlaps
    plt.tight_layout()

    # Display the plots in Streamlit
    st.pyplot(fig)


    st.write ('Minum banyak air sendiri tidak cukup sebagai faktor uji terhadap obesitas.')
    st.write ('Mengonsumsi makanan tinggi kalori memang berkaitan dengan peningkatan obesitas, namun hanya memberikan efek kecil – artinya, faktor lain seperti  aktivitas fisik, usia, dan genetik kemungkinan bermain peran lebih besar.')  









#================================================================================================================================================================


if __name__ == '__main__':
    run()
