import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import pickle
import base64
from sklearn.ensemble import RandomForestClassifier
import sqlite3
from datetime import datetime


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
if "page" not in st.session_state:
    st.session_state.page = "home"
if "user" not in st.session_state:
    st.session_state.user = None
if "symptoms" not in st.session_state:
    st.session_state.symptoms = []
if "disease" not in st.session_state:
    st.session_state.disease = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Dashboard"

st.markdown("""
<style>

/* Remove default header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== MAIN APP BACKGROUND ===== */
[data-testid="stAppViewContainer"] {
    background: #f4f6f9;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    padding: 25px;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ===== DASHBOARD CARDS ===== */
.dashboard-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    transition: 0.3s ease-in-out;
}

.dashboard-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, #4e73df, #1cc88a);
    color: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #4e73df, #224abe);
    color: white;
    border-radius: 8px;
    height: 42px;
    width: 100%;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #224abe, #1a3b8b);
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* ===== AUTH CARD ===== */
.auth-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 40px;
}

.auth-card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    width: 420px;
    transition: 0.3s;
}

.auth-card:hover {
    box-shadow: 0 12px 35px rgba(0,0,0,0.12);
}

.auth-title {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.prediction-card {
    background: white;
    padding: 35px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
    margin-top: 30px;
}

.prediction-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 5px;
}

.prediction-subtitle {
    color: #6b7280;
    margin-bottom: 25px;
    font-size: 14px;
}

.result-card {
    margin-top: 25px;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.page-title {
    font-size: 34px;
    font-weight: 800;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.med-header {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #16a34a, #15803d);
    color: white;
    margin-bottom: 30px;
}

.med-header h1 {
    font-size: 32px;
    margin: 0;
    font-weight: 800;
}

.med-header p {
    margin: 5px 0 0 0;
    opacity: 0.9;
    font-size: 15px;
}

.med-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 15px 35px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.med-item {
    background: #f9fafb;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
    font-size: 16px;
    line-height: 1.6;
}

.med-number {
    font-weight: 700;
    color: #15803d;
    margin-right: 6px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

.workout-header {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f97316, #ea580c);
    color: white;
    margin-bottom: 30px;
}

.workout-header h1 {
    font-size: 32px;
    margin: 0;
    font-weight: 800;
}

.workout-header p {
    margin: 5px 0 0 0;
    opacity: 0.9;
    font-size: 15px;
}

.workout-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 15px 35px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.workout-item {
    background: #fff7ed;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid #fed7aa;
    margin-bottom: 15px;
    font-size: 16px;
    line-height: 1.6;
}

.workout-number {
    font-weight: 700;
    color: #ea580c;
    margin-right: 8px;
}

.disclaimer-box {
    background: #fef3c7;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #fde68a;
    font-size: 14px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


protected_pages = [
    "symptoms", "prediction", "description",
    "precautions", "diet", "medication",
    "workout", "appointment", "my_appointments"
]

if st.session_state.page in protected_pages and not st.session_state.user:
    st.session_state.page = "home"
    st.warning("Please login first.")
    st.stop()
def get_db_connection():
    conn = sqlite3.connect("ai_medical.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        phone INTEGER UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_phone INTEGER,
        doctor_name TEXT,
        hospital TEXT,
        disease TEXT,
        date TEXT,
        time TEXT,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()

create_tables()
conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(appointment)")
columns = [col[1] for col in cursor.fetchall()]

if "hospital" not in columns:
    cursor.execute("ALTER TABLE appointment ADD COLUMN hospital TEXT")
    print("✅ 'hospital' column added to appointment table.")

conn.commit()
conn.close()


# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost", user="root", password="rmzyashi", database="ai_medical_db"
#     )

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

Symptom=pd.read_csv("data/symtoms_df.csv")
precautions=pd.read_csv("data/precautions_df.csv")
workout=pd.read_csv("data/workout_df.csv")
description=pd.read_csv("data/description.csv")
medications=pd.read_csv("data/medications.csv")
diet=pd.read_csv("data/diet.csv")


rf=pickle.load(open("models/rf.pkl",'rb'))
print("DESCRIPTION:",description.columns.tolist())
print("PRECAUTIONS:",precautions.columns.tolist())
print("MEDICATIONS:",medications.columns.tolist())
print("DIETS:",diet.columns.tolist())
print("WORKOUT:",workout.columns.tolist())

def find_disease_column(data):
    for col in data.columns:
        if 'disease' in col.lower() or 'prognosis' in col.lower():
            return col
    raise ValueError("No disease column found in the DataFrame.")
def helper(dis):
    dis = dis.strip().lower()

    d_col_desc = find_disease_column(description)
    d_col_pre  = find_disease_column(precautions)
    d_col_med  = find_disease_column(medications)
    d_col_die  = find_disease_column(diet)
    d_col_wrk  = find_disease_column(workout)

    desc = description[description[d_col_desc].str.lower() == dis]['Description']
    desc = " ".join(desc.values)

    pre = precautions[precautions[d_col_pre].str.lower() == dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ].values.tolist()

    med = medications[medications[d_col_med].str.lower() == dis].iloc[:, -1].values.tolist()

    die = diet[diet[d_col_die].str.lower() == dis].iloc[:, -1].values.tolist()

    wrkout = workout[workout[d_col_wrk].str.lower() == dis].iloc[:, -1].values.tolist()

    return desc, pre, med, die, wrkout
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15:'Fungal infection', 4:'Allergy', 16:'GERD', 9:'Chronic cholestasis', 14:'Drug Reaction', 33:'Peptic ulcer diseae', 1:'AIDS', 12:'Diabetes', 17:'Gastroenteritis', 6:'Bronchial Asthma', 23:'Hypertension', 30:'Migraine', 7:'Cervical spondylosis', 32:'Paralysis(brain hemorrhage)', 28:'Jaundice', 29:'Malaria', 8:'Chicken pox', 11:'Dengue', 37:'Typhoid', 40:'hepatitis A', 19:'Hepatitis B', 20:'Hepatitis C', 21:'Hepatitis D', 22:'Hepatitis E', 3:'Alcoholic hepatitis', 36:'Tuberculosis', 10:'Common Cold', 34:'Pneumonia', 13:'Dimorphic hemmorhoids(piles)', 18:'Heart attack', 39:'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def clean_list(items):
    """
    Removes NaN, None, empty strings and returns clean list
    """
    return [str(i).strip() for i in items if pd.notna(i) and str(i).strip().lower() != "nan"]

import ast

def print_numbered(items):
    count = 1
    for item in items:
        if pd.isna(item):
            continue

        # If item is a stringified list → convert
        if isinstance(item, str) and item.startswith("["):
            try:
                item = ast.literal_eval(item)
            except:
                item = [item]

        # If item is a list → print each element
        if isinstance(item, list):
            for sub in item:
                if pd.notna(sub):
                    st.write(f"{count}. {sub}")
                    count += 1
        else:
            st.write(f"{count}. {item}")
            count += 1



# ------------------- Load Datasets -------------------
@st.cache_data
def load_data():
    training = pd.read_csv("data/Training.csv")
    description_df = pd.read_csv("data/description.csv")
    precautions_df = pd.read_csv("data/precautions_df.csv")
    diets_df = pd.read_csv("data/diet.csv")
    medications_df = pd.read_csv("data/medications.csv")
    workout_df = pd.read_csv("data/workout_df.csv")
    return training, description_df, precautions_df, diets_df, medications_df, workout_df
training, description_df, precautions_df, diets_df, medications_df, workout_df = load_data()
X = training.drop("prognosis", axis=1)
y = training["prognosis"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
symptom_list = X.columns.tolist()

# ------------------- Session State -------------------

def nav(page):
    st.session_state.page = page
    st.rerun()

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Dashboard"  # default selection

with st.sidebar:
    st.image("assets/logo.png", width=120)
    st.markdown("## AI Medical System")

    if st.session_state.user:
        st.success(f"Welcome {st.session_state.user['name']}")
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.page = "login"  # replace with your login page
            st.rerun()
    else:
        st.info("Please login")

    st.title("Navigation")

    page_map = {
        "Dashboard": "home",
        "Symptoms": "symptoms",
        "My Appointments": "my_appointments"
    }

    selected_label = st.radio(
        "Navigation",
        list(page_map.keys()),
        index=list(page_map.keys()).index(st.session_state.selected_page)
    )
    # Update only if changed
    if selected_label != st.session_state.selected_page:
        st.session_state.selected_page = selected_label
        st.session_state.page = page_map[selected_label]
        st.rerun()
    else:
        st.info("Please login to access features")

# ---------------- PAGE FLOW ----------------
if st.session_state.page == "home":
    st.markdown("## 🏥 Healthcare Dashboard")

    # ---------------- KPI CARDS ----------------
    st.markdown("""
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #4e73df, #1cc88a);
        color: white;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transition: 0.3s ease-in-out;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        font-size: 18px;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown('<div class="kpi-card">🦠 40+ Diseases Covered</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi-card">🤖 AI Powered Diagnosis</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="kpi-card">📅 Appointment Booking</div>', unsafe_allow_html=True)

    # ---------------- LOGIN / SIGNUP BUTTONS ----------------
    if not st.session_state.user:
        st.markdown("<br><br>", unsafe_allow_html=True)  # add vertical space
        col1, col2 = st.columns(2, gap="medium")
        login_clicked, signup_clicked = col1.button("Login"), col2.button("Signup")

        if login_clicked:
            st.session_state.page = "login"
            st.rerun()
        if signup_clicked:
            st.session_state.page = "signup"
            st.rerun()

#------------------------ SIGNUP PAGE -------------------
elif st.session_state.page == "signup":

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">📝 Create Account</div>', unsafe_allow_html=True)

    name = st.text_input("Full Name")
    age = st.number_input("Age", 1, 120)
    phone = st.number_input("Phone Number", min_value=1000000000, max_value=9999999999, step=1)
    pw = st.text_input("Password", type="password")

    if st.button("Create Account"):
        if not name or not pw:
            st.error("All fields are required")
        else:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE phone=?", (int(phone),))
            if cursor.fetchone():
                st.error("Phone number already registered!")
            else:
                hashed_pw = make_hashes(pw)
                cursor.execute(
                    "INSERT INTO users (name, age, phone, password) VALUES (?, ?, ?, ?)",
                    (name, age, int(phone), hashed_pw)
                )
                conn.commit()
                st.success("Account created successfully!")
                st.session_state.page = "login"
                st.rerun()
            cursor.close()
            conn.close()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# ------------------------ LOGIN PAGE -------------------
elif st.session_state.page == "login":

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div style='padding:80px 60px;'>
            <h1 style='font-size:42px; font-weight:800;'>AI Medical Platform</h1>
            <p style='font-size:18px; color:#555; margin-top:15px;'>
                Smart symptom analysis. Accurate predictions.
                Secure appointment booking.
            </p>
            <div style='margin-top:40px; font-size:14px; color:#888;'>
                🔒 Secure • 🩺 Reliable • 🤖 AI Powered
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">Sign In</div>', unsafe_allow_html=True)

        phone = st.text_input("Phone Number")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE phone=?", (phone,))
            user = cursor.fetchone()
            conn.close()

            if user:
                user_dict = {key: user[key] for key in user.keys()}
                if user_dict["password"] == make_hashes(password):
                    st.session_state.user = user_dict
                    st.session_state.page = "symptoms"
                    st.rerun()
                else:
                    st.error("Incorrect password")
            else:
                st.error("Phone not registered")

        st.markdown('</div>', unsafe_allow_html=True)
            
# ------------------- SYMPTOMS PAGE -------------------

elif st.session_state.page == "symptoms":

    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="prediction-title">🧠 Symptom Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-subtitle">Select your symptoms below to get an AI-powered disease prediction.</div>',
        unsafe_allow_html=True
    )

    selected = st.multiselect(
        "Choose Your Symptoms",
        symptom_list,
        placeholder="Type to search symptoms..."
    )

    col1, col2 = st.columns([1,3])

    with col1:
        predict_clicked = st.button("Analyze")

    if predict_clicked:
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            st.session_state.symptoms = selected

            def get_predicted_value(patients_symptoms):
                input_vector = np.zeros(len(symptoms_dict))

                for item in patients_symptoms:
                    item = item.strip().lower()
                    if item in symptoms_dict:
                        input_vector[symptoms_dict[item]] = 1

                input_df = pd.DataFrame([input_vector], columns=symptoms_dict.keys())
                predicted_label = rf.predict(input_df)[0]
                return predicted_label

            with st.spinner("Analyzing symptoms..."):
                predicted_index = get_predicted_value(selected)
                predicted_disease = diseases_list[predicted_index]

            st.session_state.disease = predicted_disease
            st.session_state.page = "prediction"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- PREDICTION PAGE -------------------
elif st.session_state.page == "prediction":

    st.markdown('<div class="page-card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="page-title">Prediction Report</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">AI-based disease prediction from selected symptoms</div>',
        unsafe_allow_html=True
    )

    disease = st.session_state.disease

    # ===== RESULT CARD =====
    st.markdown(
        f"""
        <div class="result-card">
            <p style="font-size:14px; color:#6b7280; margin-bottom:5px;">
            Predicted Condition
            </p>
            <h1 style="color:#2563eb; margin-top:0;">
            {disease}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Optional disclaimer
    st.markdown(
        """
        <p style="margin-top:20px; font-size:13px; color:#6b7280;">
        ⚠ This result is generated by a machine learning model and is not a medical diagnosis.
        Please consult a healthcare professional for confirmation.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "symptoms"
            st.rerun()

    with col2:
        if st.button("Next → Description"):
            st.session_state.page = "description"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- DESCRIPTION PAGE -------------------
elif st.session_state.page == "description":

    st.markdown("""
    <div style="
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        margin-bottom: 30px;
    ">
        <h1 style="font-size:32px; font-weight:800; margin:0;">
        Disease Overview
        </h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
        Detailed explanation of your predicted condition
        </p>
    </div>
    """, unsafe_allow_html=True)

    desc, _, _, _, _ = helper(st.session_state.disease)

    if desc and str(desc).strip().lower() != "nan":

        st.markdown(f"""
        <div style="
            background:white;
            padding:35px;
            border-radius:20px;
            border:1px solid #e5e7eb;
            box-shadow:0 15px 35px rgba(0,0,0,0.05);
            font-size:18px;
            line-height:1.8;
            color:#374151;
        ">
            {desc}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Description not available.")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "prediction"
            st.rerun()

    with col2:
        if st.button("Next → Precautions"):
            st.session_state.page = "precautions"
            st.rerun()
    

# ------------------- PRECAUTIONS PAGE -------------------
elif st.session_state.page == "precautions":

    # 🔶 Top Gradient Banner
    st.markdown("""
    <div style="
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15);
    ">
        <h1 style="font-size:40px; font-weight:800; margin:0;">
        🛡 Precautionary Measures
        </h1>
        <p style="margin-top:10px; font-size:18px; opacity:0.95;">
        Important safety steps to manage your condition
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, precautions, _, _, _ = helper(st.session_state.disease)

    if precautions:
        clean_prec = clean_list(precautions[0])

        for i, p in enumerate(clean_prec, 1):
            st.markdown(f"""
            <div style="
                background: white;
                padding: 22px;
                border-radius: 18px;
                margin-bottom: 18px;
                font-size: 18px;
                font-weight: 500;
                box-shadow: 0 6px 25px rgba(0,0,0,0.05);
                transition: 0.3s;
            ">
                <span style="
                    font-weight: 700;
                    font-size: 20px;
                    color: #ea580c;
                ">
                    {i}.
                </span> {p}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("No precautions available.")

    st.info("⚠ Follow medical advice for proper treatment and recovery.")

    # Navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "description"
            st.rerun()

    with col2:
        if st.button("Next → Diet"):
            st.session_state.page = "diet"
            st.rerun()
  
# ------------------- DIET PAGE -------------------
elif st.session_state.page == "diet":
    st.markdown("""
    <div style="
        padding: 25px;
        border-radius: 20px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    ">
        <h1 style="font-size:36px; margin:0;">🥗 Dietary Recommendations</h1>
        <p style="margin-top:6px; font-size:16px; opacity:0.9;">
            Nutrition guidance based on your predicted condition
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "disease" not in st.session_state:
        st.warning("No disease selected. Please go back to the symptoms page.")
    else:
        predicted_disease = st.session_state.disease
        disease_row = diet[diet["Disease"].str.lower() == predicted_disease.lower()]

        if disease_row.empty:
            st.info("No diet recommendations available for this disease.")
        else:
            # General Diet
            general_diet = disease_row["General_Diet_Recommendations"].values[0]
            if pd.notna(general_diet):
                st.markdown("### General Diet Recommendations")
                for i, item in enumerate(general_diet.split(";"), 1):
                    st.markdown(f"""
                    <div style="
                        background:white;
                        padding:18px;
                        border-radius:15px;
                        margin-bottom:10px;
                        font-size:16px;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                    ">
                        {i}. {item.strip()}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Veg / Non-Veg choice
            diet_choice = st.radio("Choose your preferred diet type:", ("Veg", "Non-Veg"))

            if diet_choice == "Veg":
                veg_diet = disease_row["Veg_Diet"].values[0]
                if pd.notna(veg_diet):
                    st.markdown("### Veg Diet Recommendations")
                    for i, item in enumerate(veg_diet.split(";"), 1):
                        st.markdown(f"""
                        <div style="
                            background:#d1fae5;
                            padding:18px;
                            border-radius:15px;
                            margin-bottom:10px;
                            font-size:16px;
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                        ">
                            {i}. {item.strip()}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No Veg diet recommendations available.")
            else:
                non_veg_diet = disease_row["Non_Veg_Diet"].values[0]
                if pd.notna(non_veg_diet):
                    st.markdown("### Non-Veg Diet Recommendations")
                    for i, item in enumerate(non_veg_diet.split(";"), 1):
                        st.markdown(f"""
                        <div style="
                            background:#fee2e2;
                            padding:18px;
                            border-radius:15px;
                            margin-bottom:10px;
                            font-size:16px;
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                        ">
                            {i}. {item.strip()}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("No Non-Veg diet recommendations available.")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "precautions"
            st.rerun()
    with col2:
        if st.button("Next → Medication"):
            st.session_state.page = "medication"
            st.rerun()
# ------------------- MEDICATION PAGE -------------------
elif st.session_state.page == "medication":

    # Header
    st.markdown("""
    <div class="med-header">
        <h1>Suggested Medications</h1>
        <p>Recommended pharmaceutical support for your predicted condition</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, meds, _, _ = helper(st.session_state.disease)

    if meds:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)

        for i, m in enumerate(meds, 1):
            st.markdown(
                f'<div class="med-item"><span class="med-number">{i}.</span> {m}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Medication information not available.")

    # Navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "diet"
            st.rerun()

    with col2:
        if st.button("Next → Workout"):
            st.session_state.page = "workout"
            st.rerun()

# ------------------- LIFESTYLE PAGE -------------------
elif st.session_state.page == "workout":

    # Header
    st.markdown("""
    <div class="workout-header">
        <h1>Workout & Lifestyle Advice</h1>
        <p>Supportive physical activity and healthy habit recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, _, _, workout = helper(st.session_state.disease)
    workout = clean_list(workout)

    if workout:
        st.markdown('<div class="workout-card">', unsafe_allow_html=True)

        for i, w in enumerate(workout, 1):
            st.markdown(
                f'<div class="workout-item"><span class="workout-number">{i}.</span>{w}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Workout advice not available.")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ This system provides supportive guidance only. Always consult a licensed medical professional before starting any new exercise or treatment plan.
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "medication"
            st.rerun()

    with col2:
        if st.button("Next → Doctor Appointment"):
            st.session_state.page = "appointment"
            st.rerun()

# ------------------- APPOINTMENT PAGE -------------------
elif st.session_state.page == "appointment":

    st.markdown("""
    <div style="
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        margin-bottom: 30px;
    ">
        <h1 style="font-size:32px; font-weight:800; margin:0;">
        Doctor Appointment Booking
        </h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
        Select hospital, doctor and time slot
        </p>
    </div>
    """, unsafe_allow_html=True)

    predicted_disease = st.session_state.disease

    doctors_df = pd.read_csv("data/doctors.csv")
    doctors_df["Disease_clean"] = doctors_df["Disease"].str.strip().str.lower()
    predicted_clean = predicted_disease.strip().lower()

    matched = doctors_df[doctors_df["Disease_clean"] == predicted_clean]

    if matched.empty:
        st.warning("No doctors available for this disease.")
    else:

        st.markdown('<div style="background:white;padding:30px;border-radius:20px;box-shadow:0 10px 30px rgba(0,0,0,0.05);">', unsafe_allow_html=True)

        hospital_options = matched["Hospital"].unique().tolist()
        selected_hospital = st.selectbox("🏥 Select Hospital", hospital_options)

        hospital_doctors = matched[matched["Hospital"] == selected_hospital]["Name"].tolist()
        selected_doctor = st.selectbox("👨‍⚕️ Select Doctor", hospital_doctors)

        selected_date = st.date_input("📅 Select Date", min_value=datetime.today())

        all_slots = ["09:00 AM", "10:00 AM", "11:00 AM",
                     "01:00 PM", "02:00 PM", "03:00 PM"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT time FROM appointment
            WHERE doctor_name=? AND date=? AND status='booked'
        """, (selected_doctor, str(selected_date)))

        booked_slots = [row[0] for row in cursor.fetchall()]
        conn.close()

        available_slots = [s for s in all_slots if s not in booked_slots]

        if not available_slots:
            st.error("No available slots for this date.")
        else:
            selected_time = st.selectbox("⏰ Available Time", available_slots)

            if st.button("Confirm Booking"):

                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO appointment
                    (user_phone, doctor_name, hospital, disease, date, time, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    st.session_state.user["phone"],
                    selected_doctor,
                    selected_hospital,
                    predicted_disease,
                    str(selected_date),
                    selected_time,
                    "booked"
                ))
                conn.commit()
                conn.close()

                st.markdown(f"""
                <div style="
                    margin-top:20px;
                    padding:20px;
                    border-radius:15px;
                    background:#ecfeff;
                    border:1px solid #a5f3fc;
                ">
                    <b>Appointment Confirmed</b><br><br>
                    Hospital: {selected_hospital}<br>
                    Doctor: {selected_doctor}<br>
                    Date: {selected_date}<br>
                    Time: {selected_time}
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "workout"
            st.rerun()

    with col2:
        if st.button("View My Appointments"):
            st.session_state.page = "my_appointments"
            st.rerun()


# ---------------- MY APPOINTMENTS ----------------
elif st.session_state.page == "my_appointments":

    st.markdown("""
    <div style="
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        color: white;
        margin-bottom: 30px;
    ">
        <h1 style="font-size:32px; font-weight:800; margin:0;">
        My Appointments
        </h1>
        <p style="margin:5px 0 0 0; opacity:0.9;">
        View or cancel your booked appointments
        </p>
    </div>
    """, unsafe_allow_html=True)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, hospital, doctor_name, disease, date, time
        FROM appointment
        WHERE user_phone=? AND status='booked'
    """, (st.session_state.user["phone"],))

    records = cursor.fetchall()
    conn.close()

    if not records:
        st.info("No active appointments found.")
    else:
        for appt in records:
            appt_id, hospital, doctor, disease, date, time = appt

            st.markdown(f"""
            <div style="
                background:white;
                padding:20px;
                border-radius:15px;
                margin-bottom:15px;
                box-shadow:0 5px 20px rgba(0,0,0,0.05);
            ">
                <b>Hospital:</b> {hospital}<br>
                <b>Doctor:</b> {doctor}<br>
                <b>Disease:</b> {disease}<br>
                <b>Date:</b> {date}<br>
                <b>Time:</b> {time}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Cancel Appointment {appt_id}"):
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE appointment
                    SET status='cancelled'
                    WHERE id=?
                """, (appt_id,))
                conn.commit()
                conn.close()

                st.success("Appointment cancelled successfully.")
                st.rerun()

    if st.button("⬅ Back to Booking"):
        st.session_state.page = "appointment"
        st.rerun()