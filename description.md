# Milestone 2

## Repository Outline

```
 |
    ├── deployment/ -> berisi tentang file file yang digunakan untuk model deployment
    │   ├── app.py -> merupakan home page dari eda dan prediction 
    │   └── eda.py  -> berisi plot yang ada pada notebook yang kemudian akan ditampilkan di dashboard
    │   └── prediction.py -> berisi fitur prediks yang  ada pada notebook yang kemudian akan ditampilkan di dashboard
    │   └── model.pkl ->berisi model yang dipilih dan dilakukan parameter tuning untuk membuat model prediction
    ├── description.md ->  deskripsi dokumentasi repository project yang dibuat 
    ├── P1M2_Nugroho_conceptual.txt -> menjawab pertanyaan seputar concept dari FE pada machine learning
    ├── P1M2_Nugroho_Wicaksono.ipynb -> berisi code yang membuat model dari awal sampai akhir
    ├── P1M2_Nugroho_Wicaksono_inf.ipynb -> berisi pengujian terhadap model yang berhasil dibuat
    ├── url.txt -> berisi url dari dashboard  model deploy
    └── README.md -> berisi cara cara serta metriks dari pengerjaan milestone 2
```

## Problem Background
Obesitas telah berkembang menjadi krisis kesehatan global — pada 2022, sekitar 2,5 miliar orang dewasa mengalami kelebihan berat badan, dan 890 juta di antaranya tergolong obesitas (BMI ≥ 30) The Times of India.
Penyakit ini bukanlah masalah kosmetik semata, melainkan kondisi kronis yang serius: pada 2021, kelebihan berat badan langsung menyebabkan 1,6 juta kematian prematur akibat penyakit tidak menular seperti diabetes, penyakit jantung, dan kanker 

Program yang akan dibuat kali ini, akan menentukan apakah seseorang berpotensi terkena obesitas ( beserta type obesitasnya ) atau tidak.

## Project Output
program ini akan menghasilkan predict terhadap seseorang berpotensi terkena obesitas ( beserta type obesitasnya ) atau tidak. serta akan membuat dashboard untuk model deployment.

## Data

Dataset ini mencakup data untuk estimasi tingkat obesitas pada individu dari negara Meksiko, Peru, dan Kolombia, berdasarkan kebiasaan makan dan kondisi fisik mereka.

Kode tersebut mencetak ringkasan struktur DataFrame yang terdiri dari 2.087 baris (indeks 0–2.086) dan 17 kolom, terdapat delapan kolom numerik (float64) dan sembilan kolom kategori/teks (object).

dari 17 kolom tidak ada missing value


## Method
dataset yang saya gunakan berupa multiclass classification jadi penilaian yang cocok adalah F1 score macro dan gradientBoostingClassifier.

## Stacks
``` py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.outliers import Winsorizer

from scipy.stats import kendalltau, chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline as sklPipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report, ConfusionMatrixDisplay,f1_score

from imblearn.over_sampling import SMOTENC

import streamlit as st

```


## Reference
```
1.	Romero Corral et al. (2008)
“Accuracy of body mass index in diagnosing obesity in the adult general population.”
•	Hasil utama: BMI ≥ 30 punya sensitivitas rendah (~36% pria, ~49% wanita), menyebabkan lebih dari setengah orang dengan lemak tubuh berlebih tidak terdeteksi sebagai obesitas 
```
Link: [PubMed – artikel](https://pubmed.ncbi.nlm.nih.gov/18283284/)

```
    2.	Prentice & Jebb, 2001 (Obesity Reviews)
    ‘Beyond body mass index.’
    •	Hasil utama: BMI tidak membedakan massa lemak vs massa bebas lemak dan distribusi lemak tubuh (visceral vs subkutan), padahal ini penting untuk risiko penyakit metabolik.
```
Link: [Beyond body mass index - Prentice - 2001 - Obesity Reviews - Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1467-789x.2001.00031.x)


```
    3.	D O Okorodudu 1, M F Jumean, V M Montori, A Romero-Corral, V K Somers, P J Erwin, F Lopez-Jimenez DOI: 10.1038/ijo.2010.5
    Diagnostic performance of body mass index to identify obesity as defined by body adiposity: a systematic review and meta-analysisIsi: 
    •	Meta-analisis menunjukkan sensitivitas BMI dalam mendeteksi obesitas berdasarkan % body fat hanya sekitar 50%, sehingga banyak orang dengan obesitas (tinggi lemak tubuh) terklasifikasi non-obesitas oleh BMI..

```
Link: [Diagnostic performance of body mass index to identify obesity as defined by body adiposity: a systematic review and meta-analysis - PubMed](https://pubmed.ncbi.nlm.nih.gov/20125098/)



