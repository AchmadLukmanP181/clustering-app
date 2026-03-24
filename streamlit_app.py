import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(page_title="Clustering Sampah", layout="wide")

# =============================
# SIDEBAR MENU
# =============================
menu = st.sidebar.selectbox("Menu", [
    "Upload Data",
    "Preprocessing",
    "Pemodelan K-Means",
    "Evaluasi Model",
    "Ringkasan Hasil"
])

# =============================
# SESSION STATE
# =============================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_std' not in st.session_state:
    st.session_state.X_std = None
if 'labels' not in st.session_state:
    st.session_state.labels = None

# =============================
# 1. UPLOAD DATA
# =============================
if menu == "Upload Data":
    st.header("📤 Upload Data")

    file_timbulan = st.file_uploader("Upload Data Timbulan", type="xlsx")
    file_capaian = st.file_uploader("Upload Data Capaian", type="xlsx")

    if file_timbulan and file_capaian:
        df_timbulan = pd.read_excel(file_timbulan)
        df_capaian = pd.read_excel(file_capaian)

        # MERGE
        df = df_timbulan.merge(
            df_capaian,
            on=["Tahun", "Provinsi", "Kabupaten/Kota"],
            how="outer"
        )

        # RENAME
        df.rename(columns={
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan'
        }, inplace=True)

        # HANDLE MISSING
        volume_cols = ['sampah_tahunan', 'pengurangan', 'penanganan']
        df[volume_cols] = df[volume_cols].fillna(0)

        # =========================
        # 🔥 MEDIAN (3 TAHUN)
        # =========================
        df_median = df.groupby(
            ["Provinsi", "Kabupaten/Kota"]
        )[volume_cols].median().reset_index()

        # =========================
        # 🔥 HITUNG PERSENTASE
        # =========================
        df_median["perc_pengurangan"] = (
            df_median["pengurangan"] / df_median["sampah_tahunan"]
        ) * 100

        df_median["perc_penanganan"] = (
            df_median["penanganan"] / df_median["sampah_tahunan"]
        ) * 100

        st.session_state.df = df_median

        st.success("✅ Data berhasil diproses (merge + median)")
        st.dataframe(df_median.head())

# =============================
# 2. PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df is None:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df.copy()

        st.subheader("📊 Data Awal")
        st.dataframe(df.head())

        features = ['sampah_tahunan', 'pengurangan', 'penanganan']

        X = df[features]

        # Standarisasi
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        st.session_state.X_std = X_std

        st.subheader("📏 Standarisasi")
        st.write(pd.DataFrame(X_std, columns=features).describe())

# =============================
# 3. PEMODELAN KMEANS
# =============================
elif menu == "Pemodelan K-Means":
    if st.session_state.X_std is None:
        st.warning("Lakukan preprocessing dulu")
    else:
        X = st.session_state.X_std

        st.header("⚙️ K-Means")

        # ELBOW
        inertia = []
        K_range = range(2, 10)

        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            inertia.append(model.inertia_)

        st.subheader("📉 Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        ax.set_xlabel("k")
        ax.set_ylabel("SSE")
        st.pyplot(fig)

        # SILHOUETTE
        sil_scores = []
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            sil_scores.append(silhouette_score(X, labels))

        st.subheader("📈 Silhouette Score")
        fig, ax = plt.subplots()
        ax.plot(K_range, sil_scores, marker='o')
        st.pyplot(fig)

        # PILIH K
        k = st.slider("Pilih jumlah klaster", 2, 6, 3)

        if st.button("🚀 Jalankan KMeans"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            st.session_state.labels = labels

            st.success("✅ Clustering selesai")

            dist = pd.Series(labels).value_counts()
            st.write(dist)

# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi Model":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        X = st.session_state.X_std
        labels = st.session_state.labels

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        st.header("🧪 Evaluasi")
        st.write(f"Silhouette Score: {sil:.3f}")
        st.write(f"Davies-Bouldin Index: {dbi:.3f}")

# =============================
# 5. RINGKASAN
# =============================
elif menu == "Ringkasan Hasil":
    if st.session_state.labels is None:
        st.warning("Belum ada hasil")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels

        df['Cluster'] = labels

        st.header("📌 Hasil Clustering")
        st.dataframe(df)

        # RATA-RATA
        st.subheader("📊 Rata-rata per Cluster")
        mean_cluster = df.groupby('Cluster')[['perc_pengurangan', 'perc_penanganan']].mean()
        st.dataframe(mean_cluster)

        # INTERPRETASI
        st.subheader("🧠 Interpretasi")

        for i, row in mean_cluster.iterrows():
            st.markdown(f"### Cluster {i}")

            if row['perc_pengurangan'] < 15:
                ket = "Kinerja rendah"
            else:
                ket = "Kinerja tinggi"

            st.write(f"Pengurangan: {row['perc_pengurangan']:.2f}")
            st.write(f"Penanganan: {row['perc_penanganan']:.2f}")
            st.write(f"👉 {ket}")
