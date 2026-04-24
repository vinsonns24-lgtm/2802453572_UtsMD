import os
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go # Untuk visualisasi data

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # Mengarah ke folder artifacts sesuai struktur yang kita bahas sebelumnya
    clf_path = os.path.join(BASE_DIR, "artifacts", "classification_model.pkl")
    reg_path = os.path.join(BASE_DIR, "artifacts", "regression_model.pkl")
    
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    return clf_model, reg_model

# Proteksi jika model belum ada
try:
    clf_model, reg_model = load_models()
except FileNotFoundError:
    st.error("Model tidak ditemukan di folder 'artifacts/'. Pastikan Anda sudah menjalankan pipeline.py!")
    st.stop()

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🎓 Student Placement & Salary Predictor")
st.markdown(
    "Rancang bangun sistem prediksi **status penempatan kerja** dan "
    "**estimasi gaji** mahasiswa menggunakan arsitektur Monolithic."
)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/id/a/ad/Logo_Binus_University.png", width=150) # Opsional: Logo
    st.header("ℹ️ Informasi Proyek")
    st.info(
        "**Mata Kuliah:** Model Deployment\n\n"
        "**Dataset:** Dataset B (Genap)\n\n"
        "**Algoritma:** Random Forest"
    )
    st.markdown("---")
    

# ─── Form Input ───────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📋 Input Profil Mahasiswa")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("#### 🏫 Akademik")
        gender = st.selectbox("Gender", ["Male", "Female"])
        ssc = st.slider("SSC % (Sekolah Menengah)", 0, 100, 75)
        hsc = st.slider("HSC % (SMA)", 0, 100, 70)
        degree = st.slider("Degree % (Kuliah)", 0, 100, 72)
        cgpa = st.number_input("IPK (CGPA)", 0.0, 10.0, 7.5, 0.01)
        attendance = st.slider("Kehadiran (%)", 0, 100, 85)
        backlogs = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 20, 0)

    with col2:
        st.write("#### 🛠️ Skill & Ujian")
        entrance_score = st.slider("Skor Ujian Masuk", 0, 100, 70)
        tech_skill = st.slider("Technical Skill Score", 0, 100, 80)
        soft_skill = st.slider("Soft Skill Score", 0, 100, 75)

    with col3:
        st.write("#### 💼 Pengalaman")
        internship = st.number_input("Jumlah Magang", 0, 10, 1)
        projects = st.number_input("Live Projects", 0, 20, 2)
        work_exp = st.number_input("Pengalaman Kerja (Bulan)", 0, 100, 6)
        certs = st.number_input("Jumlah Sertifikasi", 0, 20, 3)
        extra = st.selectbox("Aktif Ekstrakurikuler?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Jalankan Prediksi", use_container_width=True)

# ─── Logika Prediksi & Visualisasi ─────────────────────────────────────────────
if submitted:
    # Persiapkan Data
    input_data = pd.DataFrame([{
        "gender": gender, "ssc_percentage": ssc, "hsc_percentage": hsc,
        "degree_percentage": degree, "cgpa": cgpa, "entrance_exam_score": entrance_score,
        "technical_skill_score": tech_skill, "soft_skill_score": soft_skill,
        "internship_count": internship, "live_projects": projects,
        "work_experience_months": work_exp, "certifications": certs,
        "attendance_percentage": attendance, "backlogs": backlogs,
        "extracurricular_activities": extra
    }])

    # Hitung Prediksi
    placement = clf_model.predict(input_data)[0]
    prob = clf_model.predict_proba(input_data)[0][1]
    salary = reg_model.predict(input_data)[0] if placement == 1 else 0.0

    st.subheader("📊 Hasil Analisis Model")
    
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        # Tampilan Status
        if placement == 1:
            st.success("### STATUS: PLACED 🚀")
            st.metric("Estimasi Gaji", f"{salary:.2f} LPA")
        else:
            st.error("### STATUS: NOT PLACED 🛑")
            st.metric("Estimasi Gaji", "0.00 LPA")
        
        # Gauge Chart untuk Probabilitas (Data Visualization)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Confidence Score (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': "#2ecc71" if placement == 1 else "#e74c3c"}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        # Radar Chart untuk Profil Mahasiswa (Data Visualization)
        st.write("#### 🕸️ Profil Kompetensi")
        categories = ['Technical', 'Soft Skills', 'Academic (CGPA)', 'Entrance', 'Attendance']
        # Normalisasi IPK ke skala 100 untuk radar
        values = [tech_skill, soft_skill, (cgpa/10)*100, entrance_score, attendance]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name='Profil Anda'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=350, margin=dict(l=50, r=50, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tabel Detail
    with st.expander("🔍 Lihat Detail Data Input"):
        st.table(input_data.T.rename(columns={0: 'Value'}))