from flask import Flask, render_template, request, g
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sqlite3
from datetime import datetime, timedelta
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global dictionary to store models and scalers
models = {}
scalers = {}

# Disease features with realistic input ranges
disease_features = {
    'Heart': {
        'features': ['age', 'cholesterol', 'blood_pressure', 'heart_rate'],
        'ranges': {'age': (1, 120), 'cholesterol': (100, 600), 'blood_pressure': (80, 200), 'heart_rate': (40, 200)}
    },
    'Kidney': {
        'features': ['age', 'creatinine', 'urea', 'sodium', 'potassium'],
        'ranges': {'age': (1, 120), 'creatinine': (0.1, 20), 'urea': (10, 300), 'sodium': (120, 160), 'potassium': (2.5, 7.0)}
    },
    'Liver': {
        'features': ['age', 'bilirubin', 'albumin', 'alt'],
        'ranges': {'age': (1, 120), 'bilirubin': (0.1, 5), 'albumin': (2, 6), 'alt': (5, 500)}
    },
    'Diabetes': {
        'features': ['age', 'glucose', 'bmi', 'insulin'],
        'ranges': {'age': (1, 120), 'glucose': (50, 400), 'bmi': (10, 60), 'insulin': (0, 300)}
    },
    'Lung': {
        'features': ['age', 'fev1', 'smoking_years', 'oxygen_saturation'],
        'ranges': {'age': (1, 120), 'fev1': (0.5, 6), 'smoking_years': (0, 80), 'oxygen_saturation': (50, 100)}
    },
    'Thyroid': {
        'features': ['age', 'tsh', 't3', 't4'],
        'ranges': {'age': (1, 120), 'tsh': (0.1, 20), 't3': (0.5, 5), 't4': (2, 20)}
    },
    'Stroke': {
        'features': ['age', 'blood_pressure', 'bmi', 'smoking_status'],
        'ranges': {'age': (1, 120), 'blood_pressure': (80, 200), 'bmi': (10, 60), 'smoking_status': (0, 1)}
    }
}

# Specialist doctor profiles and precautions
disease_info = {
    'Heart': {
        'specialist': 'Cardiologist',
        'name': 'Dr. Priya Sharma',
        'phone': '+91-98765-43210',
        'tests': ['ECG', 'Echocardiogram', 'Lipid Profile'],
        'precautions': {
            'eat': ['Oats', 'Fruits (e.g., berries)', 'Nuts', 'Leafy greens'],
            'avoid': ['High-fat foods', 'Excess salt', 'Trans fats', 'Smoking']
        }
    },
    'Kidney': {
        'specialist': 'Nephrologist',
        'name': 'Dr. Anil Kumar',
        'phone': '+91-87654-32109',
        'tests': ['Urine Analysis', 'Kidney Function Test', 'Ultrasound'],
        'precautions': {
            'eat': ['Low-potassium fruits (e.g., apples)', 'Lean proteins', 'Whole grains'],
            'avoid': ['High-sodium foods', 'Processed meats', 'Excess protein', 'Alcohol']
        }
    },
    'Liver': {
        'specialist': 'Hepatologist',
        'name': 'Dr. Meena Patel',
        'phone': '+91-76543-21098',
        'tests': ['ALT/AST Test', 'Bilirubin Test', 'Liver Ultrasound'],
        'precautions': {
            'eat': ['Vegetables (e.g., broccoli)', 'Fruits', 'Whole grains', 'Lean fish'],
            'avoid': ['Alcohol', 'Fatty foods', 'Sugary drinks', 'Raw shellfish']
        }
    },
    'Diabetes': {
        'specialist': 'Endocrinologist',
        'name': 'Dr. Rajesh Gupta',
        'phone': '+91-65432-10987',
        'tests': ['HbA1c', 'Fasting Glucose', 'Oral Glucose Tolerance Test'],
        'precautions': {
            'eat': ['Whole grains (e.g., brown rice)', 'Vegetables', 'Legumes', 'Nuts'],
            'avoid': ['Sugary foods/drinks', 'Refined carbs', 'High-fat meats', 'Excess sweets']
        }
    },
    'Lung': {
        'specialist': 'Pulmonologist',
        'name': 'Dr. Sanjay Verma',
        'phone': '+91-54321-09876',
        'tests': ['Spirometry', 'Chest X-ray', 'Pulse Oximetry'],
        'precautions': {
            'eat': ['Antioxidant-rich foods (e.g., berries)', 'Vegetables', 'Fish'],
            'avoid': ['Smoking', 'Polluted areas', 'Processed foods', 'High-fat dairy']
        }
    },
    'Thyroid': {
        'specialist': 'Endocrinologist',
        'name': 'Dr. Neha Singh',
        'phone': '+91-43210-98765',
        'tests': ['TSH Test', 'T3/T4 Levels', 'Thyroid Ultrasound'],
        'precautions': {
            'eat': ['Iodine-rich foods (e.g., seaweed)', 'Fruits', 'Lean proteins'],
            'avoid': ['Soy products', 'Cruciferous vegetables (in excess)', 'Processed sugars']
        }
    },
    'Stroke': {
        'specialist': 'Neurologist',
        'name': 'Dr. Vikram Rao',
        'phone': '+91-32109-87654',
        'tests': ['CT Scan', 'MRI', 'Carotid Ultrasound'],
        'precautions': {
            'eat': ['Fruits', 'Vegetables', 'Whole grains', 'Fish'],
            'avoid': ['High-sodium foods', 'Smoking', 'Excess alcohol', 'Fatty meats']
        }
    }
}

# Database connection management
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(':memory:', check_same_thread=False)
        c = g.db.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reviews
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, rating INTEGER, comment TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS appointments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_name TEXT, doctor_name TEXT, 
                      appointment_date TEXT, appointment_time TEXT, disease TEXT, created_at TEXT)''')
        g.db.commit()
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def get_reviews(db_conn):
    try:
        c = db_conn.cursor()
        c.execute("SELECT name, rating, comment FROM reviews")
        reviews = [{'name': row[0], 'rating': row[1], 'comment': row[2]} for row in c.fetchall()]
        return reviews
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        return []

def check_doctor_availability(db_conn, doctor_name, appointment_date, appointment_time):
    try:
        print(f"Raw appointment_time received: {appointment_time}")
        
        # Normalize appointment_time to HH:MM format
        time_parts = appointment_time.split(':')
        if len(time_parts) >= 2:
            appointment_time = f"{time_parts[0]:>02}:{time_parts[1]:>02}"  # Ensure HH:MM format
        else:
            raise ValueError(f"Invalid time format: {appointment_time}")
        
        print(f"Checking availability for {doctor_name} on {appointment_date} at {appointment_time}")
        
        c = db_conn.cursor()
        c.execute("SELECT * FROM appointments")
        all_appointments = c.fetchall()
        print(f"All appointments in database: {all_appointments}")
        
        # Check if the requested slot is booked
        c.execute('''SELECT appointment_time FROM appointments 
                     WHERE doctor_name = ? AND appointment_date = ?''',
                  (doctor_name, appointment_date))
        booked_times = [row[0] for row in c.fetchall()]
        booked_times = [time.strip() for time in booked_times]
        print(f"Booked times on {appointment_date} for {doctor_name}: {booked_times}")
        
        # Allow the first booking if no appointments exist
        if not all_appointments:
            print("No appointments exist, allowing first booking")
            return True, []
        
        if appointment_time in booked_times:
            print(f"Slot {appointment_time} on {appointment_date} is booked")
            available_slots = []
            base_date = datetime.strptime(appointment_date, '%Y-%m-%d')
            working_hours = ['09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00']
            
            for day_offset in range(0, 3):
                check_date = (base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                c.execute('''SELECT appointment_time FROM appointments 
                             WHERE doctor_name = ? AND appointment_date = ?''',
                          (doctor_name, check_date))
                booked_on_date = [row[0] for row in c.fetchall()]
                booked_on_date = [time.strip() for time in booked_on_date]
                
                for hour in working_hours:
                    if hour not in booked_on_date:
                        available_slots.append({'date': check_date, 'time': hour})
                
                if len(available_slots) >= 5:
                    break
            
            print(f"Suggested slots: {available_slots}")
            return False, available_slots
        
        print(f"Slot {appointment_time} on {appointment_date} is available")
        return True, []
    except Exception as e:
        print(f"Error in check_doctor_availability: {e}")
        return False, []

def validate_input(disease, input_data):
    ranges = disease_features[disease]['ranges']
    for feature, value in input_data.items():
        min_val, max_val = ranges[feature]
        if not (min_val <= value <= max_val):
            return f"{feature.replace('_', ' ').title()} must be between {min_val} and {max_val}"
    return None

def train_models():
    global models, scalers
    print("Starting model training...")
    
    # Attempt to train Kidney model with real data
    try:
        if not os.path.exists('kidney_disease.csv'):
            print("kidney_disease.csv not found. Generating synthetic data for Kidney...")
            raise FileNotFoundError("kidney_disease.csv not found")
        
        df_kidney = pd.read_csv('kidney_disease.csv')
        required_columns = ['age', 'creatinine', 'urea', 'sodium', 'potassium', 'classification']
        missing_cols = [col for col in required_columns if col not in df_kidney.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in kidney_disease.csv: {missing_cols}")
        
        df_kidney = df_kidney.dropna(subset=required_columns)
        df_kidney['classification'] = df_kidney['classification'].map({'ckd': 1, 'notckd': 0})
        df_kidney = df_kidney.dropna(subset=['classification'])
        
        if df_kidney.empty:
            raise ValueError("Kidney dataset is empty after preprocessing")
        
        X_kidney = df_kidney[['age', 'creatinine', 'urea', 'sodium', 'potassium']]
        y_kidney = df_kidney['classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.3, random_state=42)
        scaler_kidney = StandardScaler()
        X_train_scaled = scaler_kidney.fit_transform(X_train)
        
        model_kidney = LogisticRegression(max_iter=1000)
        model_kidney.fit(X_train_scaled, y_train)
        
        models['Kidney'] = model_kidney
        scalers['Kidney'] = scaler_kidney
        print("Kidney model and scaler trained successfully with real data")
    
    except Exception as e:
        print(f"Error loading/training kidney dataset: {e}. Falling back to synthetic data for Kidney.")
        # Fallback to synthetic data for Kidney
        try:
            n_samples = 1000
            data = {}
            ranges = disease_features['Kidney']['ranges']
            for feature in disease_features['Kidney']['features']:
                min_val, max_val = ranges[feature]
                if feature == 'age':
                    data[feature] = np.random.randint(min_val, max_val + 1, n_samples)
                else:
                    data[feature] = np.random.uniform(min_val, max_val, n_samples)
            
            data['target'] = np.where((data['creatinine'] > 1.5) & (data['urea'] > 50), 1, 0)
            
            df_kidney = pd.DataFrame(data)
            df_kidney['target'] = df_kidney['target'].astype(int)
            
            X_kidney = df_kidney.drop('target', axis=1)
            y_kidney = df_kidney['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.3, random_state=42)
            scaler_kidney = StandardScaler()
            X_train_scaled = scaler_kidney.fit_transform(X_train)
            
            model_kidney = LogisticRegression(max_iter=1000)
            model_kidney.fit(X_train_scaled, y_train)
            
            models['Kidney'] = model_kidney
            scalers['Kidney'] = scaler_kidney
            print("Kidney model and scaler trained successfully with synthetic data")
        except Exception as e:
            print(f"Failed to train Kidney model with synthetic data: {e}")

    # Train models for other diseases with synthetic data
    np.random.seed(42)
    for disease in disease_features.keys():
        if disease == 'Kidney':
            continue
        try:
            n_samples = 1000
            data = {}
            ranges = disease_features[disease]['ranges']
            for feature in disease_features[disease]['features']:
                min_val, max_val = ranges[feature]
                if feature in ['age', 'smoking_years', 'smoking_status']:
                    data[feature] = np.random.randint(min_val, max_val + 1, n_samples)
                else:
                    data[feature] = np.random.uniform(min_val, max_val, n_samples)
            
            if disease == 'Heart':
                data['target'] = np.where((data['cholesterol'] > 200) & (data['blood_pressure'] > 140), 1, 0)
            elif disease == 'Liver':
                data['target'] = np.where((data['bilirubin'] > 1.2) & (data['alt'] > 40), 1, 0)
            elif disease == 'Diabetes':
                data['target'] = np.where((data['glucose'] > 126) & (data['bmi'] > 30), 1, 0)
            elif disease == 'Lung':
                data['target'] = np.where((data['fev1'] < 2.5) & (data['smoking_years'] > 20), 1, 0)
            elif disease == 'Thyroid':
                data['target'] = np.where((data['tsh'] > 4.5) | (data['tsh'] < 0.4), 1, 0)
            elif disease == 'Stroke':
                data['target'] = np.where((data['blood_pressure'] > 140) & (data['smoking_status'] == 1), 1, 0)
            
            df = pd.DataFrame(data)
            df['target'] = df['target'].astype(int)
            
            X = df.drop('target', axis=1)
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            models[disease] = model
            scalers[disease] = scaler
            print(f"{disease} model and scaler trained successfully")
        
        except Exception as e:
            print(f"Error training {disease} model: {e}")
    
    print(f"Training complete. Models: {list(models.keys())}, Scalers: {list(scalers.keys())}")

@app.route('/', methods=['GET', 'POST'])
def index():
    db = get_db()
    reviews = get_reviews(db)
    if request.method == 'POST':
        disease = request.form.get('disease')
        if disease in disease_features:
            return render_template('input.html', disease=disease, 
                                 features=disease_features[disease]['features'], 
                                 ranges=disease_features[disease]['ranges'])
        return render_template('index.html', error="Invalid disease selected", 
                             diseases=disease_features.keys(), reviews=reviews)
    return render_template('index.html', diseases=disease_features.keys(), reviews=reviews)

@app.route('/predict', methods=['POST'])
def predict():
    db = get_db()
    reviews = get_reviews(db)
    try:
        disease = request.form.get('disease')
        print(f"Processing disease: {disease}")
        print(f"Form data received: {request.form.to_dict()}")
        if disease not in disease_features:
            print(f"Invalid disease: {disease}")
            return render_template('error.html', message="Invalid disease selected")
        
        input_data = {}
        for feature in disease_features[disease]['features']:
            value = request.form.get(feature)
            print(f"Feature {feature}: Value {value}")
            if value is None or value.strip() == '':
                print(f"Missing or empty value for {feature}")
                return render_template('error.html', message=f"Please provide a value for {feature.replace('_', ' ').title()}")
            try:
                input_data[feature] = float(value)
            except ValueError as ve:
                print(f"Invalid value for {feature}: {value}, Error: {ve}")
                return render_template('error.html', message=f"Invalid value for {feature.replace('_', ' ').title()}: {value}")
        
        validation_error = validate_input(disease, input_data)
        if validation_error:
            print(f"Validation error: {validation_error}")
            return render_template('error.html', message=validation_error)
        
        input_df = pd.DataFrame([input_data])
        print(f"Input DataFrame: {input_df}")
        
        scaler = scalers.get(disease)
        if scaler is None:
            print(f"Scaler not found for {disease}. Available scalers: {list(scalers.keys())}")
            return render_template('error.html', message=f"Scaler not initialized for {disease} disease. Please try again later.")
        input_scaled = scaler.transform(input_df)
        print(f"Input scaled: {input_scaled}")
        
        model = models.get(disease)
        if model is None:
            print(f"Model not found for {disease}. Available models: {list(models.keys())}")
            return render_template('error.html', message=f"Model not trained for {disease} disease")
        probability = model.predict_proba(input_scaled)[0][1] * 100
        print(f"Prediction probability: {probability}%")
        
        info = disease_info[disease]
        specialist = info['specialist']
        doctor_name = info['name']
        phone_number = info['phone']
        recommended_tests = info['tests'] if probability > 50 else ["Consult a doctor for further evaluation"]
        precautions_eat = info['precautions']['eat']
        precautions_avoid = info['precautions']['avoid']
        
        print("Rendering result.html with data")
        return render_template('result.html',
                            disease=disease,
                            probability=probability,
                            specialist=specialist,
                            doctor_name=doctor_name,
                            phone_number=phone_number,
                            recommended_tests=recommended_tests,
                            precautions_eat=precautions_eat,
                            precautions_avoid=precautions_avoid,
                            reviews=reviews)
    
    except ValueError as e:
        print(f"ValueError in predict: {e}")
        return render_template('error.html', message=f"Invalid input value: {e}")
    except Exception as e:
        print(f"Unexpected error in predict: {e}")
        return render_template('error.html', message=f"Prediction error: {str(e)}")

@app.route('/submit_review', methods=['POST'])
def submit_review():
    db = get_db()
    try:
        name = request.form.get('name', 'Anonymous')
        rating = int(request.form.get('rating', 5))
        comment = request.form.get('comment', '')
        
        if not (1 <= rating <= 5):
            return render_template('index.html', diseases=disease_features.keys(), 
                                 reviews=get_reviews(db), error="Rating must be between 1 and 5")
        
        c = db.cursor()
        c.execute("INSERT INTO reviews (name, rating, comment) VALUES (?, ?, ?)", (name, rating, comment))
        db.commit()
        
        reviews = get_reviews(db)
        return render_template('index.html', diseases=disease_features.keys(), 
                             reviews=reviews, success="Review submitted successfully!")
    
    except Exception as e:
        print(f"Error in submit_review: {e}")
        return render_template('index.html', diseases=disease_features.keys(), 
                             reviews=get_reviews(db), error=f"Error submitting review: {str(e)}")

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    db = get_db()
    try:
        patient_name = request.form.get('patient_name')
        doctor_name = request.form.get('doctor_name')
        disease = request.form.get('disease')
        appointment_date = request.form.get('appointment_date')
        appointment_time = request.form.get('appointment_time')
        
        print(f"Booking appointment: {patient_name}, {doctor_name}, {disease}, {appointment_date}, {appointment_time}")
        
        if not all([patient_name, doctor_name, disease, appointment_date, appointment_time]):
            return render_template('error.html', message="All appointment fields are required")
        
        try:
            datetime.strptime(appointment_date, '%Y-%m-%d')
        except ValueError:
            return render_template('error.html', message="Invalid date format. Use YYYY-MM-DD")
        
        time_parts = appointment_time.split(':')
        if len(time_parts) >= 2:
            appointment_time = f"{time_parts[0]:>02}:{time_parts[1]:>02}"
        else:
            return render_template('error.html', message="Invalid time format. Use HH:MM (24-hour)")
        
        is_available, available_slots = check_doctor_availability(db, doctor_name, appointment_date, appointment_time)
        if not is_available:
            print(f"Doctor {doctor_name} not available at {appointment_time} on {appointment_date}. Suggested slots: {available_slots}")
            return render_template('appointment.html',
                                 patient_name=patient_name,
                                 doctor_name=doctor_name,
                                 appointment_date=appointment_date,
                                 appointment_time=appointment_time,
                                 disease=disease,
                                 error="This slot is not available. Please choose one of the following slots or try another date/time.",
                                 available_slots=available_slots)
        
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c = db.cursor()
        c.execute('''INSERT INTO appointments (patient_name, doctor_name, appointment_date, 
                    appointment_time, disease, created_at) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (patient_name, doctor_name, appointment_date, appointment_time, disease, created_at))
        db.commit()
        print(f"Appointment inserted. Last row ID: {c.lastrowid}")
        
        print("Rendering appointment.html with success")
        return render_template('appointment.html',
                             patient_name=patient_name,
                             doctor_name=doctor_name,
                             appointment_date=appointment_date,
                             appointment_time=appointment_time,
                             disease=disease,
                             success="Appointment booked successfully!")
    
    except Exception as e:
        print(f"Error in book_appointment: {e}")
        return render_template('error.html', message=f"Error booking appointment: {str(e)}")

# Initialize the app
try:
    train_models()  # Train models at startup
except Exception as e:
    print(f"Failed to initialize app: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)