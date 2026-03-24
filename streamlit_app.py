import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(layout="wide")

# =============================
# SIDEBAR MENU
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
for key in ["df_raw","df_median","X_std","labels"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# 1. UPLOAD DATA
# =============================
if menu == "Upload Data":
    st.header("📤 Upload Data")

    file_timbulan = st.file_uploader("Upload Timbulan", type="xlsx")
    file_capaian = st.file_uploader("Upload Capaian", type="xlsx")

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

        if st.button("Isi Missing dengan 0"):
            df[['sampah_tahunan','pengurangan','penanganan']] = df[['sampah_tahunan','pengurangan','penanganan']].fillna(0)
            st.success("Missing value diisi 0")

        # =====================
        # MEDIAN
        # =====================
        if st.button("Hitung Median"):
            df_median = df.groupby("Kabupaten/Kota")[
                ['sampah_tahunan','pengurangan','penanganan']
            ].median().reset_index()

            # =====================
            # FIX PEMBAGIAN NOL
            # =====================
            df_median["perc_pengurangan"] = np.where(
                df_median["sampah_tahunan"] == 0,
                0,
                (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
            )

            df_median["perc_penanganan"] = np.where(
                df_median["sampah_tahunan"] == 0,
                0,
                (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100
            )

            # =====================
            # CLEAN DATA (ANTI ERROR)
            # =====================
            df_median = df_median.replace([np.inf, -np.inf], 0)
            df_median = df_median.fillna(0)

            st.session_state.df_median = df_median

            st.success("✅ Median & persentase berhasil")
            st.dataframe(df_median)

        # =====================
        # FEATURE + SCALING
        # =====================
        if st.session_state.df_median is not None:
            df_median = st.session_state.df_median

            features = st.multiselect(
                "Pilih Variabel",
                ['perc_pengurangan','perc_penanganan'],
                default=['perc_pengurangan','perc_penanganan']
            )

            if st.button("Standarisasi"):
                X = df_median[features]

                # CLEAN LAGI (DOUBLE SAFETY)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(0)

                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                st.session_state.X_std = X_std

                st.subheader("Sebelum")
                st.dataframe(X.describe())

                st.subheader("Sesudah")
                st.dataframe(pd.DataFrame(X_std, columns=features).describe())

# =============================
# 3. PEMODELAN
# =============================
elif menu == "Pemodelan":
    if st.session_state.X_std is None:
        st.warning("Preprocessing dulu")
    else:
        X = pd.DataFrame(st.session_state.X_std)

        # CLEAN FINAL (ANTI CRASH)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        st.header("⚙️ K-Means")

        # =====================
        # ELBOW
        # =====================
        inertia = []
        K_range = range(2,10)

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

        # =====================
        # SILHOUETTE
        # =====================
        sil_scores = []
        for k in K_range:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            sil_scores.append(silhouette_score(X, labels))

        st.subheader("📈 Silhouette")
        fig, ax = plt.subplots()
        ax.plot(K_range, sil_scores, marker='o')
        st.pyplot(fig)

        # =====================
        # PILIH K
        # =====================
        k = st.slider("Jumlah Cluster", 2, 6, 3)

        if st.button("🚀 Jalankan KMeans"):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            st.session_state.labels = labels
            st.success("✅ Clustering selesai")

# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        X = pd.DataFrame(st.session_state.X_std)
        X = X.fillna(0)

        labels = st.session_state.labels

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        st.header("🧪 Evaluasi")
        st.write(f"Silhouette Score: {sil:.3f}")
        st.write(f"Davies-Bouldin Index: {dbi:.3f}")

# =============================
# 5. INTERPRETASI
# =============================
elif menu == "Interpretasi":
    if st.session_state.labels is None:
        st.warning("Belum clustering")
    else:
        df = st.session_state.df_median.copy()
        df['Cluster'] = st.session_state.labels

        st.header("🧠 Interpretasi JAKSTRANAS")

        mean_cluster = df.groupby('Cluster')[['perc_pengurangan','perc_penanganan']].mean()
        st.dataframe(mean_cluster)

        for i, row in mean_cluster.iterrows():
            st.markdown(f"### Cluster {i}")

            if row['perc_penanganan'] >= 70 and row['perc_pengurangan'] >= 30:
                ket = "✅ Kinerja Baik"
            elif row['perc_penanganan'] >= 50:
                ket = "⚠️ Cukup"
            else:
                ket = "❌ Rendah"

            st.write(f"Pengurangan: {row['perc_pengurangan']:.2f}%")
            st.write(f"Penanganan: {row['perc_penanganan']:.2f}%")
            st.write(f"👉 {ket}")
            st.subheader("📊 Visualisasi Cluster (Scatter Plot)")

fig, ax = plt.subplots()

scatter = ax.scatter(
    df['perc_pengurangan'],
    df['perc_penanganan'],
    c=df['Cluster'],
)

ax.set_xlabel("Persentase Pengurangan (%)")
ax.set_ylabel("Persentase Penanganan (%)")
ax.set_title("Distribusi Cluster")

st.pyplot(fig)
st.subheader("🏆 Ranking Wilayah")

df['skor_kinerja'] = (
    (df['perc_pengurangan'] * 0.3) +
    (df['perc_penanganan'] * 0.7)
)

df_sorted = df.sort_values(by="skor_kinerja", ascending=False)

st.markdown("### 🔝 Top 10 Wilayah Terbaik")
st.dataframe(df_sorted[['Kabupaten/Kota','skor_kinerja']].head(10))

st.markdown("### 🔻 Bottom 10 Wilayah Terburuk")
st.dataframe(df_sorted[['Kabupaten/Kota','skor_kinerja']].tail(10))
st.subheader("🧠 Narasi Otomatis")

for i, row in mean_cluster.iterrows():

    if row['perc_penanganan'] >= 70 and row['perc_pengurangan'] >= 30:
        narasi = "Wilayah dalam cluster ini memiliki kinerja pengelolaan sampah yang sangat baik karena telah memenuhi target nasional (JAKSTRANAS)."
    
    elif row['perc_penanganan'] >= 50:
        narasi = "Wilayah dalam cluster ini menunjukkan kinerja yang cukup baik, namun masih belum sepenuhnya memenuhi target nasional."
    
    else:
        narasi = "Wilayah dalam cluster ini memiliki kinerja pengelolaan sampah yang rendah dan perlu mendapatkan perhatian khusus dari pemerintah."

    st.markdown(f"""
    ### Cluster {i}
    - Rata-rata Pengurangan: **{row['perc_pengurangan']:.2f}%**
    - Rata-rata Penanganan: **{row['perc_penanganan']:.2f}%**
    - 📌 **Kesimpulan:** {narasi}
    """)
