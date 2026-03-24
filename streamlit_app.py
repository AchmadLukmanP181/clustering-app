import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(page_title="Clustering Sampah", layout="wide")

# =============================
# SESSION STATE STEP NAVIGATION
# =============================
if "step" not in st.session_state:
    st.session_state.step = 1

def next_step():
    st.session_state.step += 1

def prev_step():
    st.session_state.step -= 1

# =============================
# STEP 1: UPLOAD DATA
# =============================
if st.session_state.step == 1:
    st.title("📤 Upload Data Capaian")

    file = st.file_uploader("Upload file capaian (.xlsx)", type="xlsx")

    if file:
        df = pd.read_excel(file)

        # Rename sesuai sempro
        df.rename(columns={
            'Kabupaten/Kota': 'wilayah',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan',
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan'
        }, inplace=True)

        # HANDLE MISSING
        df[['pengurangan','penanganan','sampah_tahunan']] = df[['pengurangan','penanganan','sampah_tahunan']].fillna(0)

        # =========================
        # 🔥 MEDIAN 3 TAHUN
        # =========================
        df_median = df.groupby("wilayah")[['pengurangan','penanganan','sampah_tahunan']].median().reset_index()

        # Hitung persentase
        df_median["perc_pengurangan"] = (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
        df_median["perc_penanganan"] = (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100

        st.session_state.df = df_median

        st.success("✅ Data berhasil diproses")
        st.dataframe(df_median.head())

        st.button("Next ➡️", on_click=next_step)

# =============================
# STEP 2: PREPROCESSING
# =============================
elif st.session_state.step == 2:
    st.title("🧹 Preprocessing Data")

    df = st.session_state.df.copy()

    # Missing Value Check
    st.subheader("🔍 Cek Missing Value")
    st.dataframe(df.isnull().sum())

    # Pilih variabel
    st.subheader("🎯 Pilih Variabel")
    features = st.multiselect(
        "Pilih variabel untuk clustering",
        ['perc_pengurangan','perc_penanganan'],
        default=['perc_pengurangan','perc_penanganan']
    )

    # Standarisasi
    if st.button("⚙️ Standarisasi Data"):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(df[features])

        st.session_state.X_std = X_std
        st.session_state.features = features

        st.subheader("📊 Sebelum Standarisasi")
        st.dataframe(df[features].describe())

        st.subheader("📏 Setelah Standarisasi")
        st.dataframe(pd.DataFrame(X_std, columns=features).describe())

    st.button("⬅️ Back", on_click=prev_step)
    st.button("Next ➡️", on_click=next_step)

# =============================
# STEP 3: KMEANS
# =============================
elif st.session_state.step == 3:
    st.title("⚙️ Pemodelan K-Means")

    X = st.session_state.X_std

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
    k = st.slider("Jumlah klaster", 2, 6, 2)

    if st.button("🚀 Jalankan KMeans"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        st.session_state.labels = labels
        st.session_state.k = k

        st.success("✅ Clustering selesai")

    st.button("⬅️ Back", on_click=prev_step)
    st.button("Next ➡️", on_click=next_step)

# =============================
# STEP 4: EVALUASI
# =============================
elif st.session_state.step == 4:
    st.title("🧪 Evaluasi Model")

    X = st.session_state.X_std
    labels = st.session_state.labels
    df = st.session_state.df.copy()

    df['Cluster'] = labels

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)

    st.write(f"Silhouette Score: **{sil:.3f}**")
    st.write(f"Davies-Bouldin Index: **{dbi:.3f}**")

    # Wilayah per cluster
    st.subheader("📍 Wilayah Tiap Klaster")
    wilayah_cluster = df.groupby('Cluster')['wilayah'].apply(list)
    st.dataframe(wilayah_cluster)

    st.button("⬅️ Back", on_click=prev_step)
    st.button("Next ➡️", on_click=next_step)

# =============================
# STEP 5: INTERPRETASI
# =============================
elif st.session_state.step == 5:
    st.title("🧠 Interpretasi Klaster (JAKSTRANAS)")

    df = st.session_state.df.copy()
    labels = st.session_state.labels

    df['Cluster'] = labels

    mean_cluster = df.groupby('Cluster')[['perc_pengurangan','perc_penanganan']].mean()

    st.dataframe(mean_cluster)

    for i, row in mean_cluster.iterrows():
        st.markdown(f"### Cluster {i}")

        pengurangan = row['perc_pengurangan']
        penanganan = row['perc_penanganan']

        # JAKSTRANAS RULE
        if penanganan >= 70 and pengurangan >= 30:
            ket = "✅ Kinerja Baik (sesuai target nasional)"
        elif penanganan >= 50:
            ket = "⚠️ Cukup Baik"
        else:
            ket = "❌ Rendah"

        st.write(f"Pengurangan: {pengurangan:.2f}%")
        st.write(f"Penanganan: {penanganan:.2f}%")
        st.write(f"👉 {ket}")

    st.button("⬅️ Back", on_click=prev_step)
