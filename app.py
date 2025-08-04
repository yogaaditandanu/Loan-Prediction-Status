import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

import os
import subprocess

# Debug: cek package terinstall
st.text("Packages installed:")
st.text(subprocess.check_output(["pip", "freeze"]).decode("utf-8"))


# === Load Model & Tools ===
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- FUNGSI BARU UNTUK SKOR KREDIT OTOMATIS ---
def hitung_skor_kredit_otomatis(income, percent_income, default, home):
    """
    Menghitung estimasi skor kredit berdasarkan aturan sederhana.
    """
    # Skor dasar dimulai dari rata-rata
    score = 650

    # 1. Penyesuaian berdasarkan Riwayat Gagal Bayar (Faktor Paling Signifikan)
    if default == "Yes":
        score -= 150
    else:
        score += 50

    # 2. Penyesuaian berdasarkan Rasio Pinjaman (Loan Percent Income)
    if percent_income <= 0.20: # Sangat sehat
        score += 40
    elif percent_income <= 0.40: # Cukup sehat
        score += 15
    elif percent_income > 0.50: # Berisiko
        score -= 70

    # 3. Penyesuaian berdasarkan Kepemilikan Rumah
    if home == "OWN" or home == "MORTGAGE":
        score += 30
    else: # RENT
        score -= 20

    # 4. Penyesuaian berdasarkan Pendapatan
    if income > 150_000_000: # Pendapatan tinggi
        score += 30
    elif income < 30_000_000: # Pendapatan rendah
        score -= 25

    # Pastikan skor tetap dalam rentang 300 - 850
    if score > 850:
        score = 850
    if score < 300:
        score = 300

    return score
# --- AKHIR DARI FUNGSI BARU ---


# === Streamlit Page Setup ===
st.set_page_config(page_title="Smart Loan Approval Checker", layout="wide")

# === Sidebar ===
st.sidebar.image("loan.jpg", width=180)
st.sidebar.title("Smart Loan Approval Checker 🔮")
page = st.sidebar.radio("📂 Menu", [
    "🏡 Overview", 
    "📘 User Guide", 
    "🔎 Single Check", 
    "🗃️ Batch Check", 
    "🗣️ Feedback"
])

# === 1. Overview Page ===
if page == "🏡 Overview":
    st.markdown("<h2 style='color:#6C63FF;'>🏦 Smart Loan Approval Checker</h2>", unsafe_allow_html=True)
    st.markdown("""
Selamat datang di **Smart Loan Approval Checker**! Aplikasi ini akan membantu memprediksi apakah pengajuan pinjaman seseorang berpeluang **disetujui** atau **ditolak**, berdasarkan data finansial dan pribadi.

---

### 🚀 Fitur Utama
- 🔍 **Single Check** – Cek satu pengajuan pinjaman secara instan  
- 📁 **Batch Check** – Prediksi banyak nasabah via CSV  
- 📘 **Panduan Penggunaan** – Pelajari arti setiap input  
- 🗣️ **Feedback** – Beri saran atau pendapat kamu!

---

💡 Cocok untuk analis, staf keuangan, hingga pengguna umum.

🧑‍💻 *Dibuat oleh oleh Yoga Adi Tandanu*
""")

# === 2. Panduan Penggunaan ===
elif page == "📘 User Guide":
    st.markdown("<h2 style='color:#6C63FF;'>📘 Panduan Penggunaan</h2>", unsafe_allow_html=True)

    st.markdown("### 🔍 Single Check")
    st.markdown("""
Isi form dengan informasi berikut:

| Label | Penjelasan |
|-------|-------------|
| 🎂 Usia | Umur nasabah (18–100 tahun) |
| 🚻 Jenis Kelamin | Pilih *male* atau *female* |
| 🎓 Pendidikan | SMA, Sarjana, Magister, dll |
| 🏘️ Status Rumah | RENT, MORTGAGE, OWN |
| ❗ Riwayat Gagal Bayar | Pernah gagal bayar? Yes/No |
| 💼 Pendapatan Tahunan | Total penghasilan setahun |
| 💳 Jumlah Pinjaman | Jumlah pinjaman yang diajukan |
| 📊 Bunga Tahunan (%) | Persentase bunga pinjaman |
| 🧮 Rasio Pinjaman | Pinjaman ÷ Pendapatan |
| 📉 Skor Kredit | Nilai antara 300 - 850 |
| 🎯 Tujuan Pinjaman | VENTURE, MEDICAL, dll |
""")

    st.markdown("### 🗃️ Batch Check")
    st.markdown("Unggah file CSV dengan kolom berikut:")
    st.code("person_age, person_gender, person_education, person_income,\nperson_home_ownership, previous_loan_defaults_on_file,\nloan_amnt, loan_int_rate, loan_percent_income,\ncredit_score, loan_intent")

    st.markdown("### 🗣️ Feedback")
    st.markdown("Kirimkan kritik, saran, atau ulasan kamu di menu **Feedback**.")

# === 3. Prediksi Tunggal ===
elif page == "🔎 Single Check":
    st.markdown("<h2 style='color:#6C63FF;'>🔎 Cek Pinjaman Individu</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("#### 📋 Formulir Data Nasabah")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Usia", 18, 100, 30)
            gender = st.selectbox("🚻 Jenis Kelamin", ["male", "female"])
            education = st.selectbox("🎓 Pendidikan", ["High School", "Bachelor", "Master", "Associate"])
            home = st.selectbox("🏘️ Kepemilikan Rumah", ["RENT", "MORTGAGE", "OWN"])
            default = st.selectbox("❗ Pernah Gagal Bayar?", ["Yes", "No"])

        # --- MODIFIKASI BAGIAN INI ---
        with col2:
            income = st.number_input("💼 Pendapatan Tahunan (IDR)", 1000, 1_000_000_000, 50_000_000, step=1_000_000)
            loan_amt = st.number_input("💳 Jumlah Pinjaman (IDR)", 1000, 100_000_000, 10_000_000, step=1_000_000)
            interest = st.slider("📊 Bunga Tahunan (%)", 5.0, 30.0, 15.0)
            
            # Hitung rasio pinjaman secara dinamis
            ratio_value = round(loan_amt / income, 2) if income > 0 else 0.0
            percent_income = st.slider("🧮 Rasio Pinjaman", 0.0, 1.0, ratio_value, disabled=True)
            
            # Hitung skor kredit secara otomatis menggunakan fungsi baru
            credit_score = hitung_skor_kredit_otomatis(income, percent_income, default, home)
            
            # Tampilkan skor kredit otomatis, non-aktifkan slider agar tidak bisa diubah manual
            st.slider("📉 Skor Kredit (Otomatis)", 300, 850, credit_score, disabled=True)
            
            intent = st.selectbox("🎯 Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION"])
        # --- AKHIR DARI MODIFIKASI ---

        st.info("💡 Contoh: Pendapatan 50 juta, pinjaman 10 juta → Rasio Pinjaman = 0.2")

        if st.button("🧠 Prediksi Sekarang"):
            input_df = pd.DataFrame([{
                'person_age': age,
                'person_gender': gender,
                'person_education': education,
                'person_income': income,
                'person_home_ownership': home,
                'loan_amnt': loan_amt,
                'loan_int_rate': interest,
                'loan_percent_income': percent_income,
                'credit_score': credit_score,
                'loan_intent': intent,
                'previous_loan_defaults_on_file': default
            }])

            for col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

            features = model.get_booster().feature_names
            scaled_input = scaler.transform(input_df[features])

            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            st.markdown("### 🔮 Hasil Prediksi")
            if prediction == 1:
                st.success("✅ Pinjaman kemungkinan **DISETUJUI**")
            else:
                st.error("❌ Pinjaman kemungkinan **DITOLAK**")
            st.metric("📈 Peluang Persetujuan", f"{probability * 100:.2f} %")

# === 4. Batch Prediction ===
elif page == "🗃️ Batch Check":
    st.markdown("<h2 style='color:#6C63FF;'>🗃️ Cek Banyak Data Sekaligus</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Unggah file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])

        features = model.get_booster().feature_names
        df_scaled = scaler.transform(df[features])

        df["prediction"] = model.predict(df_scaled)
        df["approval_prob"] = model.predict_proba(df_scaled)[:, 1]

        st.markdown("### 📋 Hasil Prediksi")
        filter_val = st.slider("🔎 Tampilkan data dengan peluang > ", 0.0, 1.0, 0.5)
        st.dataframe(df[df["approval_prob"] > filter_val].head(10))

        st.download_button("📥 Unduh Hasil", data=df.to_csv(index=False), file_name="hasil_prediksi_pinjaman.csv")

# === 5. Feedback Page ===
elif page == "🗣️ Feedback":
    st.markdown("<h2 style='color:#6C63FF;'>🗣️ Berikan Pendapatmu</h2>", unsafe_allow_html=True)

    name = st.text_input("🧑 Nama Kamu (Opsional)")
    rating = st.slider("⭐ Beri Rating Aplikasi", 1, 5, 4)
    comments = st.text_area("💬 Saran, kritik, atau kesan:")

    if st.button("📨 Kirim"):
        st.success("Terima kasih atas feedback-nya!")
        st.markdown(f"**Nama**: {name if name else 'Anonim'}  \n**Rating**: {rating}/5  \n**Komentar**: {comments}")

