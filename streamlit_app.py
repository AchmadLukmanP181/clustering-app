import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

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
            'Timbulan Sampah Harian(ton)': 'sampah_harian',
'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            '%Pengurangan Sampah(B/A)': 'perc_pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan',
            '%Penanganan Sampah(C/A)': 'perc_penanganan',
            'Sampah Terkelola Tahunan (ton/tahun)(B+C)': 'sampah_terkelola',
            '%Sampah Terkelola(B+C)/A': 'perc_sampah_terkelola'
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan'
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
        # HANDLE MISSING
        volume_cols = ['sampah_tahunan', 'pengurangan', 'penanganan']
        df[volume_cols] = df[volume_cols].fillna(0)

# =========================
        # 3. MEDIAN 3 TAHUN
        # 🔥 MEDIAN (3 TAHUN)
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
        df_median = df.groupby(
            ["Provinsi", "Kabupaten/Kota"]
        )[volume_cols].median().reset_index()

# =========================
        # 4. PEMILIHAN VARIABEL + SCALING
        # 🔥 HITUNG PERSENTASE
# =========================
        if 'df_median' in st.session_state:
        df_median["perc_pengurangan"] = (
            df_median["pengurangan"] / df_median["sampah_tahunan"]
        ) * 100

            df_median = st.session_state.df_median
        df_median["perc_penanganan"] = (
            df_median["penanganan"] / df_median["sampah_tahunan"]
        ) * 100

            st.subheader("Pemilihan Variabel & Standardisasi")
        st.session_state.df = df_median

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
        st.success("✅ Data berhasil diproses (merge + median)")
        st.dataframe(df_median.head())

            selected_features = st.multiselect(
                "Pilih Variabel",
                options=numeric_cols,
                default=['sampah_tahunan', 'pengurangan', 'penanganan']
            )

            if len(selected_features) >= 2:

                X = df_median[selected_features]
# =============================
# 2. PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df is None:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df.copy()

                use_scaling = st.checkbox("Gunakan StandardScaler", value=True)
        st.subheader("📊 Data Awal")
        st.dataframe(df.head())

                if use_scaling:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
        features = ['sampah_tahunan', 'pengurangan', 'penanganan']

                    st.success("Data berhasil distandardisasi")
                    st.dataframe(pd.DataFrame(X_scaled, columns=selected_features).head())
                else:
                    X_scaled = X.values
                    st.info("Tanpa standardisasi")
        X = df[features]

                st.session_state.X_final = X_scaled
                st.session_state.selected_features = selected_features
        # Standarisasi
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        st.session_state.X_std = X_std

# ===============================
# PEMODELAN
# ===============================
elif st.session_state.selected_tab == "Pemodelan":
        st.subheader("📏 Standarisasi")
        st.write(pd.DataFrame(X_std, columns=features).describe())

    if 'X_final' not in st.session_state:
        st.warning("Selesaikan preprocessing dulu")
# =============================
# 3. PEMODELAN KMEANS
# =============================
elif menu == "Pemodelan K-Means":
    if st.session_state.X_std is None:
        st.warning("Lakukan preprocessing dulu")
else:
        X_used = st.session_state.X_final
        df = st.session_state.df_median.copy()

        st.subheader("Clustering K-Means")

        k = st.slider("Jumlah Klaster", 2, 6, 3)

        if st.button("Jalankan Clustering"):

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
            kmeans.fit(X_used)

            df['cluster_labels'] = kmeans.labels_

            sil = silhouette_score(X_used, kmeans.labels_)
            dbi = davies_bouldin_score(X_used, kmeans.labels_)

            st.success(f"Jumlah klaster: {k}")
            st.write(f"Silhouette Score: {sil:.4f}")
            st.write(f"Davies-Bouldin Index: {dbi:.4f}")

            st.session_state.df_cluster = df
            labels = kmeans.fit_predict(X)

            # Interpretasi
            st.subheader("Interpretasi Klaster")
            st.session_state.labels = labels

            for cluster in sorted(df['cluster_labels'].unique()):
                cluster_df = df[df['cluster_labels'] == cluster]
            st.success("✅ Clustering selesai")

                avg = cluster_df[['perc_pengurangan','perc_penanganan']].mean()
            dist = pd.Series(labels).value_counts()
            st.write(dist)

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
# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi Model":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
else:
        df = st.session_state.df_cluster

        st.title("Visualisasi Klaster")
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

        for cluster in sorted(df['cluster_labels'].unique()):
            cluster_df = df[df['cluster_labels'] == cluster]
        df['Cluster'] = labels

            st.subheader(f"Klaster {cluster}")
        st.header("📌 Hasil Clustering")
        st.dataframe(df)

            col1, col2, col3 = st.columns(3)
        # RATA-RATA
        st.subheader("📊 Rata-rata per Cluster")
        mean_cluster = df.groupby('Cluster')[['perc_pengurangan', 'perc_penanganan']].mean()
        st.dataframe(mean_cluster)

            col1.metric("Total Sampah", int(cluster_df['sampah_tahunan'].sum()))
            col2.metric("Total Pengurangan", int(cluster_df['pengurangan'].sum()))
            col3.metric("Total Penanganan", int(cluster_df['penanganan'].sum()))
        # INTERPRETASI
        st.subheader("🧠 Interpretasi")

            # Bar chart
            avg = cluster_df[['perc_pengurangan','perc_penanganan']].mean()
        for i, row in mean_cluster.iterrows():
            st.markdown(f"### Cluster {i}")

            fig, ax = plt.subplots()
            avg.plot(kind='bar', ax=ax)
            ax.set_title("Rata-rata Persentase")
            st.pyplot(fig)
            if row['perc_pengurangan'] < 15:
                ket = "Kinerja rendah"
            else:
                ket = "Kinerja tinggi"

            st.dataframe(cluster_df[['Kabupaten/Kota','sampah_tahunan','pengurangan','penanganan']])
            st.write(f"Pengurangan: {row['perc_pengurangan']:.2f}")
            st.write(f"Penanganan: {row['perc_penanganan']:.2f}")
            st.write(f"👉 {ket}")
