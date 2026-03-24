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
    st.image("https://raw.githubusercontent.com/awalidya/TugasAkhir/main/logo%20sampah.png", width=200)
    st.markdown("### Trash Achievement")

    tabs = ["Halaman Utama", "Upload Data", "Pemodelan", "Visualisasi"]
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
    Aplikasi ini digunakan untuk mengelompokkan wilayah di Indonesia 
    berdasarkan karakteristik pengelolaan sampah menggunakan metode K-Means.

    Tahapan:
    1. Merge data timbulan & capaian
    2. Agregasi median 3 tahun
    3. Clustering K-Means
    """)

# ===============================
# UPLOAD DATA
# ===============================
elif st.session_state.selected_tab == "Upload Data":

    uploaded_timbulan = st.file_uploader("Upload Data Timbulan", type=["xlsx"])
    uploaded_capaian = st.file_uploader("Upload Data Capaian", type=["xlsx"])

    if uploaded_timbulan and uploaded_capaian:

        df_timbulan = pd.read_excel(uploaded_timbulan)
        df_capaian = pd.read_excel(uploaded_capaian)

        # Merge
        df = df_timbulan.merge(
            df_capaian,
            on=["Tahun","Provinsi","Kabupaten/Kota"],
            how="outer"
        )

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

        # Drop kolom tidak dipakai (jika ada)
        drop_cols = [
            'Timbulan Sampah Tahunan(ton)',
            'Bahan baku Sampah Tahunan (ton/tahun)(E)',
            'Recycling Rate(D+E)/A',
            'Daur ulang Sampah Tahunan (ton/tahun)(D)',
            'P1/P2'
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Missing value
        volume_cols = ['sampah_harian','sampah_tahunan','pengurangan','penanganan','sampah_terkelola']
        df[volume_cols] = df[volume_cols].fillna(0)

        # =====================
        # MEDIAN 3 TAHUN
        # =====================
        df_median = df.groupby(["Provinsi","Kabupaten/Kota"])[volume_cols].median().reset_index()

        # Hitung ulang persentase
        df_median["perc_pengurangan"] = (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
        df_median["perc_penanganan"] = (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100
        df_median["perc_sampah_terkelola"] = (df_median["sampah_terkelola"] / df_median["sampah_tahunan"]) * 100

        df_median = df_median.fillna(0)

        st.success("Preprocessing berhasil!")

        st.subheader("Data Median 3 Tahun")
        st.dataframe(df_median.head())

        st.session_state.df_median = df_median

# ===============================
# PEMODELAN
# ===============================
elif st.session_state.selected_tab == "Pemodelan":

    if 'df_median' not in st.session_state:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df_median.copy()

        st.subheader("Clustering K-Means")

        features = ['sampah_tahunan', 'pengurangan', 'penanganan']
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Jumlah Klaster", 2, 6, 3)

        if st.button("Jalankan Clustering"):

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)

            df['cluster_labels'] = kmeans.labels_

            # Evaluasi
            sil = silhouette_score(X_scaled, kmeans.labels_)
            dbi = davies_bouldin_score(X_scaled, kmeans.labels_)

            st.success(f"Jumlah klaster: {k}")
            st.write(f"Silhouette Score: {sil:.4f}")
            st.write(f"Davies-Bouldin Index: {dbi:.4f}")

            st.session_state.df_cluster = df

            # =====================
            # INTERPRETASI
            # =====================
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

            # Pie chart provinsi
            prov = cluster_df['Provinsi'].value_counts().head(5)

            fig2, ax2 = plt.subplots()
            ax2.pie(prov, labels=prov.index, autopct='%1.1f%%')
            ax2.set_title("Top Provinsi")
            st.pyplot(fig2)

            st.dataframe(cluster_df[['Kabupaten/Kota','sampah_tahunan','pengurangan','penanganan']])
