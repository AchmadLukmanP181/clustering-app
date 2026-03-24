import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(layout="wide")

# =============================
# SIDEBAR
# =============================
menu = st.sidebar.selectbox("Menu", [
    "Upload Data",
    "Preprocessing",
    "Pemodelan",
    "Evaluasi",
    "Interpretasi"
])

# =============================
# SESSION STATE
# =============================
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_median" not in st.session_state:
    st.session_state.df_median = None
if "X_std" not in st.session_state:
    st.session_state.X_std = None
if "labels" not in st.session_state:
    st.session_state.labels = None

# =============================
# 1. UPLOAD
# =============================
if menu == "Upload Data":
    st.header("Upload Data")

    file_timbulan = st.file_uploader("Timbulan", type="xlsx")
    file_capaian = st.file_uploader("Capaian", type="xlsx")

    if file_timbulan and file_capaian:
        df_timbulan = pd.read_excel(file_timbulan)
        df_capaian = pd.read_excel(file_capaian)

        df = df_timbulan.merge(
            df_capaian,
            on=["Tahun", "Provinsi", "Kabupaten/Kota"],
            how="outer"
        )

        df.rename(columns={
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan'
        }, inplace=True)

        st.session_state.df_raw = df
        st.success("Data berhasil diupload")
        st.dataframe(df.head())

# =============================
# 2. PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df_raw is None:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df_raw.copy()

        st.subheader("Missing Value")
        st.write(df.isnull().sum())

        if st.button("Isi Missing dengan 0"):
            df[['sampah_tahunan','pengurangan','penanganan']] = df[['sampah_tahunan','pengurangan','penanganan']].fillna(0)

        if st.button("Hitung Median"):
            df_median = df.groupby(["Kabupaten/Kota"])[
                ['sampah_tahunan','pengurangan','penanganan']
            ].median().reset_index()

            df_median["perc_pengurangan"] = df_median["pengurangan"] / df_median["sampah_tahunan"] * 100
            df_median["perc_penanganan"] = df_median["penanganan"] / df_median["sampah_tahunan"] * 100

            st.session_state.df_median = df_median
            st.dataframe(df_median)

        if st.session_state.df_median is not None:
            df_median = st.session_state.df_median

            features = st.multiselect(
                "Pilih variabel",
                ['perc_pengurangan','perc_penanganan'],
                default=['perc_pengurangan','perc_penanganan']
            )

            if st.button("Standarisasi"):
                scaler = StandardScaler()
                X_std = scaler.fit_transform(df_median[features])

                st.session_state.X_std = X_std

                st.subheader("Sebelum")
                st.dataframe(df_median[features].describe())

                st.subheader("Sesudah")
                st.dataframe(pd.DataFrame(X_std, columns=features).describe())

# =============================
# 3. PEMODELAN
# =============================
elif menu == "Pemodelan":
    if st.session_state.X_std is None:
        st.warning("Preprocessing dulu")
    else:
        X = st.session_state.X_std

        st.subheader("Elbow")
        inertia = []
        K_range = range(2,10)

        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            inertia.append(model.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        st.pyplot(fig)

        st.subheader("Silhouette")
        sil = []
        for k in K_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            sil.append(silhouette_score(X, labels))

        fig, ax = plt.subplots()
        ax.plot(K_range, sil, marker='o')
        st.pyplot(fig)

        k = st.slider("Jumlah Cluster", 2, 6, 3)

        if st.button("Jalankan KMeans"):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            st.session_state.labels = labels
            st.success("Clustering selesai")

# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        X = st.session_state.X_std
        labels = st.session_state.labels

        st.write("Silhouette:", silhouette_score(X, labels))
        st.write("DBI:", davies_bouldin_score(X, labels))

# =============================
# 5. INTERPRETASI
# =============================
elif menu == "Interpretasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        df = st.session_state.df_median.copy()
        df['Cluster'] = st.session_state.labels

        mean = df.groupby('Cluster')[['perc_pengurangan','perc_penanganan']].mean()
        st.dataframe(mean)

        for i, row in mean.iterrows():
            st.markdown(f"### Cluster {i}")

            if row['perc_penanganan'] >= 70 and row['perc_pengurangan'] >= 30:
                ket = "Kinerja Baik"
            else:
                ket = "Perlu peningkatan"

            st.write(ket)
