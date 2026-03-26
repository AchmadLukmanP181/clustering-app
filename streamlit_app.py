import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(layout="wide", page_title="Clustering Sampah")

# =============================
# SIDEBAR
# =============================
menu = st.sidebar.selectbox("📌 Menu", [
    "Upload Data",
    "Preprocessing",
    "Pemodelan",
    "Evaluasi",
    "Interpretasi"
])

# =============================
# SESSION STATE
# =============================
for key in ["df_raw","df_median","df_model","X_std","labels"]:
    if key not in st.session_state:
        st.session_state[key] = None

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

        df = df_timbulan.merge(
            df_capaian,
            on=["Tahun","Provinsi","Kabupaten/Kota"],
            how="outer"
        )

        df.rename(columns={
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan'
        }, inplace=True)

        st.session_state.df_raw = df

        st.success("✅ Data berhasil diupload")
        st.dataframe(df.head())

# =============================
# 2. PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if st.session_state.df_raw is None:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df_raw.copy()

        st.subheader("🔍 Missing Value")
        st.write(df.isnull().sum())

        # =========================
        # HANDLE MISSING VALUE
        # =========================
        if st.button("Isi Missing Value"):

            st.subheader("📊 Data Sebelum Handling Missing Value")
            st.dataframe(df.head())

            st.write("Jumlah Missing Value Sebelum:")
            st.write(df.isna().sum())

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)

            st.subheader("✅ Data Setelah Handling Missing Value")
            st.dataframe(df.head())

            st.write("Jumlah Missing Value Setelah:")
            st.write(df.isna().sum())

            st.session_state.df_raw = df

            st.success("Missing value berhasil diisi")

        # =========================
        # MEDIAN
        # =========================
        if st.button("Hitung Median"):

            volume_cols = ['sampah_tahunan','pengurangan','penanganan']

            df_median = df.groupby(
                ['Provinsi','Kabupaten/Kota']
            )[volume_cols].median().reset_index()

            # HITUNG PERSENTASE
            df_median["perc_pengurangan"] = (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
            df_median["perc_penanganan"] = (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100
            df_median["perc_sampah_terkelola"] = (
                (df_median["pengurangan"] + df_median["penanganan"]) / df_median["sampah_tahunan"]
            ) * 100

            df_median = df_median.round(2)

            # =========================
            # CEK SEBELUM CLEAN
            # =========================
            st.subheader("🧹 Sebelum Cleaning")
            st.write("NaN:")
            st.write(df_median.isna().sum())

            # =========================
            # CLEAN
            # =========================
            df_median = df_median.replace([np.inf, -np.inf], 0)
            df_median = df_median.fillna(0)

            # =========================
            # SETELAH CLEAN
            # =========================
            st.subheader("✅ Setelah Cleaning")
            st.write(df_median.isna().sum())

            st.subheader("📊 Preview Data")
            st.dataframe(df_median.head())

            st.session_state.df_median = df_median
            st.session_state.df_model = df_median.copy()

            st.success("✅ Median berhasil")

        # =========================
        # SCALING
        # =========================
        if st.session_state.df_model is not None:

            st.subheader("📊 Data untuk Modeling")

            X = st.session_state.df_model[['sampah_tahunan','pengurangan','penanganan']]

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            st.subheader("📊 Sebelum Standardisasi")
            st.dataframe(X.head())
            st.write(X.describe())

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            st.subheader("⚖️ Setelah Standardisasi")
            st.dataframe(X_scaled_df.head())
            st.write(X_scaled_df.describe())

            st.session_state.X_std = X_scaled

            st.success("✅ Data siap untuk modeling")

# =============================
# 3. PEMODELAN
# =============================
elif menu == "Pemodelan":
    if st.session_state.X_std is None:
        st.warning("Preprocessing dulu")
    else:
        X = pd.DataFrame(st.session_state.X_std)

        st.header("⚙️ K-Means")

        inertia = []
        sil_scores = []
        K_range = range(2,10)

        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)

            inertia.append(model.inertia_)
            sil_scores.append(silhouette_score(X, model.labels_))

        # ELBOW
        st.subheader("📉 Elbow")
        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        st.pyplot(fig)

        # SILHOUETTE
        st.subheader("📈 Silhouette")
        fig, ax = plt.subplots()
        ax.plot(K_range, sil_scores, marker='o')
        st.pyplot(fig)

        k = st.slider("Jumlah Cluster", 2, 6, 3)

        if st.button("🚀 Jalankan KMeans"):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            st.session_state.labels = labels
            st.session_state.df_model['Cluster'] = labels

            st.success("✅ Clustering selesai")

# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        X = pd.DataFrame(st.session_state.X_std)
        labels = st.session_state.labels

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        st.header("🧪 Evaluasi Model")
        st.write(f"Silhouette Score: {sil:.3f}")
        st.write(f"Davies-Bouldin Index: {dbi:.3f}")

# =============================
# 5. INTERPRETASI
# =============================
elif menu == "Interpretasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        df = st.session_state.df_model.copy()

        st.header("🧠 Interpretasi")

        mean_cluster = df.groupby('Cluster')[['perc_pengurangan','perc_penanganan']].mean()

        st.subheader("📊 Rata-rata Cluster")
        st.dataframe(mean_cluster)

        # SCATTER
        fig, ax = plt.subplots()
        ax.scatter(df['perc_pengurangan'], df['perc_penanganan'], c=df['Cluster'])
        ax.set_xlabel("Pengurangan (%)")
        ax.set_ylabel("Penanganan (%)")
        st.pyplot(fig)

        # RANKING
        df['skor'] = 0.3*df['perc_pengurangan'] + 0.7*df['perc_penanganan']
        df_sorted = df.sort_values(by="skor", ascending=False)

        st.subheader("🏆 Top 10")
        st.dataframe(df_sorted[['Kabupaten/Kota','skor']].head(10))

        st.subheader("🔻 Bottom 10")
        st.dataframe(df_sorted[['Kabupaten/Kota','skor']].tail(10))

        # INTERPRETASI
        for i, row in mean_cluster.iterrows():
            if row['perc_penanganan'] >= 70 and row['perc_pengurangan'] >= 30:
                ket = "✅ Kinerja Baik"
            elif row['perc_penanganan'] >= 50:
                ket = "⚠️ Cukup"
            else:
                ket = "❌ Rendah"

            st.markdown(f"""
            ### Cluster {i}
            - Pengurangan: {row['perc_pengurangan']:.2f}%
            - Penanganan: {row['perc_penanganan']:.2f}%
            - 👉 {ket}
            """)
