import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(page_title="Clustering Sampah", layout="wide")

# =============================
# CSS STYLE
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #fff5f5, #ffe3e3);
}
[data-testid="stSidebar"] {
    background-color: #2b2d42;
    color: white;
}
h1, h2, h3 {
    color: #d62828;
}
.stButton>button {
    background-color: #d62828;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #a61e1e;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
menu_list = [
    "Upload Data",
    "Preprocessing",
    "Pemodelan K-Means",
    "Evaluasi Model",
    "Interpretasi"
]

icons = {
    "Upload Data": "📤",
    "Preprocessing": "🧹",
    "Pemodelan K-Means": "⚙️",
    "Evaluasi Model": "🧪",
    "Interpretasi": "🧠"
}

if "menu" not in st.session_state:
    st.session_state.menu = menu_list[0]

st.sidebar.title("📊 Dashboard Clustering")
menu = st.sidebar.radio(
    "Navigasi",
    menu_list,
    format_func=lambda x: f"{icons[x]} {x}"
)

st.session_state.menu = menu

# Progress
step_idx = menu_list.index(menu) + 1
st.sidebar.progress(step_idx / len(menu_list))
st.sidebar.caption(f"Step {step_idx} / {len(menu_list)}")

# =============================
# NAVIGATION FUNCTION
# =============================
def next_menu():
    idx = menu_list.index(st.session_state.menu)
    if idx < len(menu_list) - 1:
        st.session_state.menu = menu_list[idx + 1]

def prev_menu():
    idx = menu_list.index(st.session_state.menu)
    if idx > 0:
        st.session_state.menu = menu_list[idx - 1]

# =============================
# 1. UPLOAD
# =============================
if menu == "Upload Data":
    st.title("📤 Upload Data Capaian")

    file = st.file_uploader("Upload Excel", type="xlsx")

    if file:
        df = pd.read_excel(file)

        df.rename(columns={
            'Kabupaten/Kota': 'wilayah',
            'Pengurangan Sampah Tahunan (ton/tahun)(B)': 'pengurangan',
            'Penanganan Sampah Tahunan (ton/tahun)(C)': 'penanganan',
            'Timbulan Sampah Tahunan (ton/tahun)(A)': 'sampah_tahunan'
        }, inplace=True)

        df[['pengurangan','penanganan','sampah_tahunan']] = df[['pengurangan','penanganan','sampah_tahunan']].fillna(0)

        # MEDIAN
        df_median = df.groupby("wilayah")[['pengurangan','penanganan','sampah_tahunan']].median().reset_index()

        # PERSENTASE
        df_median["perc_pengurangan"] = (df_median["pengurangan"] / df_median["sampah_tahunan"]) * 100
        df_median["perc_penanganan"] = (df_median["penanganan"] / df_median["sampah_tahunan"]) * 100

        st.session_state.df = df_median

        st.success("✅ Data siap")
        st.dataframe(df_median.head())

    col1, col2 = st.columns([1,1])
    with col2:
        st.button("Next ➡️", on_click=next_menu)

# =============================
# 2. PREPROCESSING
# =============================
elif menu == "Preprocessing":
    if "df" not in st.session_state:
        st.warning("Upload data dulu")
    else:
        df = st.session_state.df.copy()

        st.title("🧹 Preprocessing")

        st.subheader("Missing Value")
        st.dataframe(df.isnull().sum())

        features = st.multiselect(
            "Pilih variabel",
            ['perc_pengurangan','perc_penanganan'],
            default=['perc_pengurangan','perc_penanganan']
        )

        if st.button("Standarisasi"):
            scaler = StandardScaler()
            X_std = scaler.fit_transform(df[features])

            st.session_state.X_std = X_std
            st.session_state.features = features

            st.subheader("Sebelum")
            st.dataframe(df[features].describe())

            st.subheader("Sesudah")
            st.dataframe(pd.DataFrame(X_std, columns=features).describe())

    col1, col2 = st.columns([1,1])
    with col1:
        st.button("⬅️ Back", on_click=prev_menu)
    with col2:
        st.button("Next ➡️", on_click=next_menu)

# =============================
# 3. KMEANS
# =============================
elif menu == "Pemodelan K-Means":
    if "X_std" not in st.session_state:
        st.warning("Preprocessing dulu")
    else:
        X = st.session_state.X_std

        st.title("⚙️ K-Means")

        inertia = []
        K_range = range(2,10)

        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(X)
            inertia.append(model.inertia_)

        st.subheader("Elbow")
        fig, ax = plt.subplots()
        ax.plot(K_range, inertia, marker='o')
        st.pyplot(fig)

        sil_scores = []
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            sil_scores.append(silhouette_score(X, labels))

        st.subheader("Silhouette")
        fig, ax = plt.subplots()
        ax.plot(K_range, sil_scores, marker='o')
        st.pyplot(fig)

        k = st.slider("Jumlah klaster", 2, 6, 2)

        if st.button("Jalankan KMeans"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            st.session_state.labels = labels

            st.success("Clustering selesai")

    col1, col2 = st.columns([1,1])
    with col1:
        st.button("⬅️ Back", on_click=prev_menu)
    with col2:
        st.button("Next ➡️", on_click=next_menu)

# =============================
# 4. EVALUASI
# =============================
elif menu == "Evaluasi Model":
    if "labels" not in st.session_state:
        st.warning("Belum clustering")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels

        df['Cluster'] = labels

        X = st.session_state.X_std

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

        st.title("🧪 Evaluasi")
        st.write(f"Silhouette: {sil:.3f}")
        st.write(f"DBI: {dbi:.3f}")

        st.subheader("Wilayah per Cluster")
        st.dataframe(df.groupby('Cluster')['wilayah'].apply(list))

    col1, col2 = st.columns([1,1])
    with col1:
        st.button("⬅️ Back", on_click=prev_menu)
    with col2:
        st.button("Next ➡️", on_click=next_menu)

# =============================
# 5. INTERPRETASI
# =============================
elif menu == "Interpretasi":
    if "labels" not in st.session_state:
        st.warning("Belum clustering")
    else:
        df = st.session_state.df.copy()
        df['Cluster'] = st.session_state.labels

        st.title("🧠 Interpretasi JAKSTRANAS")

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

    col1, col2 = st.columns([1,1])
    with col1:
        st.button("⬅️ Back", on_click=prev_menu)
