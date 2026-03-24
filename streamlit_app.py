import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(layout="wide")
st.title("Aplikasi Pengelompokan Wilayah Berdasarkan Pengelolaan Sampah")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### Trash Achievement")

    tabs = ["Halaman Utama", "Upload Data", "Preprocessing", "Pemodelan", "Visualisasi"]
    for tab in tabs:
        if st.button(tab):
            st.session_state.selected_tab = tab

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Halaman Utama"

# ===============================
# HALAMAN UTAMA
# ===============================
if st.session_state.selected_tab == "Halaman Utama":
    st.write("""
    Aplikasi ini digunakan untuk clustering wilayah berdasarkan pengelolaan sampah.

    🔹 Tahapan:
    1. Upload data capaian
    2. Preprocessing (missing value, median, seleksi variabel, standardisasi)
    3. Clustering K-Means
    """)

# ===============================
# UPLOAD DATA
# ===============================
elif st.session_state.selected_tab == "Upload Data":

    uploaded_file = st.file_uploader("Upload Data Capaian", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        # Rename kolom
        df.rename(columns={
            'Timbulan Sampah Harian(ton)': 'sampah_harian',
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            '%Pengurangan Sampah(B/A)': 'perc_pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan',
            '%Penanganan Sampah(C/A)': 'perc_penanganan',
            'Sampah Terkelola Tahunan (ton/tahun)(B+C)': 'sampah_terkelola',
            '%Sampah Terkelola(B+C)/A': 'perc_sampah_terkelola'
        }, inplace=True)

        st.success("Data berhasil diupload")
        st.dataframe(df.head())

        st.session_state.df_raw = df


# ===============================
# PREPROCESSING
# ===============================
elif st.session_state.selected_tab == "Preprocessing":

    if 'df_raw' not in st.session_state:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df_raw.copy()

        st.subheader("Preprocessing Data")

        # =========================
        # 1. CEK MISSING
        # =========================
        if st.checkbox("Tampilkan Missing Value"):
            st.write(df.isnull().sum())

        # =========================
        # 2. HANDLE MISSING
        # =========================
        if st.button("Tangani Missing Value (Isi 0)"):
            volume_cols = ['sampah_harian','sampah_tahunan','pengurangan','penanganan','sampah_terkelola']
            df[volume_cols] = df[volume_cols].fillna(0)

            st.success("Missing value berhasil ditangani")
            st.session_state.df_clean = df

        # =========================
        # 3. MEDIAN 3 TAHUN
        # =========================
        if 'df_clean' in st.session_state:
            if st.button("Hitung Median 3 Tahun"):

                df_clean = st.session_state.df_clean

                volume_cols = ['sampah_harian','sampah_tahunan','pengurangan','penanganan','sampah_terkelola']

                df_median = df_clean.groupby(["Provinsi","Kabupaten/Kota"])[volume_cols].median().reset_index()

                # Hitung ulang persentase
                df_median["perc_pengurangan"] = (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
                df_median["perc_penanganan"] = (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100
                df_median["perc_sampah_terkelola"] = (df_median["sampah_terkelola"] / df_median["sampah_tahunan"]) * 100

                df_median = df_median.fillna(0)

                st.success("Median berhasil dihitung")
                st.dataframe(df_median.head())

                st.session_state.df_median = df_median

        # =========================
        # 4. PEMILIHAN VARIABEL + SCALING
        # =========================
        if 'df_median' in st.session_state:

            df_median = st.session_state.df_median

            st.subheader("Pemilihan Variabel & Standardisasi")

            numeric_cols = [
                'sampah_harian',
                'sampah_tahunan',
                'pengurangan',
                'penanganan',
                'perc_pengurangan',
                'perc_penanganan',
                'sampah_terkelola',
                'perc_sampah_terkelola'
            ]

            selected_features = st.multiselect(
                "Pilih Variabel",
                options=numeric_cols,
                default=['sampah_tahunan', 'pengurangan', 'penanganan']
            )

            if len(selected_features) >= 2:

                X = df_median[selected_features]

                use_scaling = st.checkbox("Gunakan StandardScaler", value=True)

                if use_scaling:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    st.success("Data berhasil distandardisasi")
                    st.dataframe(pd.DataFrame(X_scaled, columns=selected_features).head())
                else:
                    X_scaled = X.values
                    st.info("Tanpa standardisasi")

                st.session_state.X_final = X_scaled
                st.session_state.selected_features = selected_features


# ===============================
# PEMODELAN
# ===============================
elif st.session_state.selected_tab == "Pemodelan":

    if 'X_final' not in st.session_state:
        st.warning("Selesaikan preprocessing dulu")
    else:
        X_used = st.session_state.X_final
        df = st.session_state.df_median.copy()

        st.subheader("Clustering K-Means")

        k = st.slider("Jumlah Klaster", 2, 6, 3)

        if st.button("Jalankan Clustering"):

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_used)

            df['cluster_labels'] = kmeans.labels_

            sil = silhouette_score(X_used, kmeans.labels_)
            dbi = davies_bouldin_score(X_used, kmeans.labels_)

            st.success(f"Jumlah klaster: {k}")
            st.write(f"Silhouette Score: {sil:.4f}")
            st.write(f"Davies-Bouldin Index: {dbi:.4f}")

            st.session_state.df_cluster = df

            # Interpretasi
            st.subheader("Interpretasi Klaster")

            for cluster in sorted(df['cluster_labels'].unique()):
                cluster_df = df[df['cluster_labels'] == cluster]

                avg = cluster_df[['perc_pengurangan','perc_penanganan']].mean()

                st.markdown(f"### Klaster {cluster}")
                st.write(avg)

                if avg['perc_pengurangan'] < 15:
                    st.warning("Kinerja rendah")
                else:
                    st.success("Kinerja lebih baik")


# ===============================
# VISUALISASI
# ===============================
elif st.session_state.selected_tab == "Visualisasi":

    if 'df_cluster' not in st.session_state:
        st.warning("Lakukan clustering dulu")
    else:
        df = st.session_state.df_cluster

        st.title("Visualisasi Klaster")

        for cluster in sorted(df['cluster_labels'].unique()):
            cluster_df = df[df['cluster_labels'] == cluster]

            st.subheader(f"Klaster {cluster}")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Sampah", int(cluster_df['sampah_tahunan'].sum()))
            col2.metric("Total Pengurangan", int(cluster_df['pengurangan'].sum()))
            col3.metric("Total Penanganan", int(cluster_df['penanganan'].sum()))

            # Bar chart
            avg = cluster_df[['perc_pengurangan','perc_penanganan']].mean()

            fig, ax = plt.subplots()
            avg.plot(kind='bar', ax=ax)
            ax.set_title("Rata-rata Persentase")
            st.pyplot(fig)

            st.dataframe(cluster_df[['Kabupaten/Kota','sampah_tahunan','pengurangan','penanganan']])
