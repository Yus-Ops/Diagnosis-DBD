from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
import pytz
from zoneinfo import ZoneInfo


app = Flask(__name__)
CORS(app)

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'App', 'model', 'dbd_model.joblib')
ENCODER_PATH = os.path.join(PROJECT_ROOT, 'App', 'model', 'target_encoder.pkl')
DATABASE_PATH = os.path.join(PROJECT_ROOT, 'data', 'diagnosis_history.db')

model = None
target_encoder = None

# Load model and encoder
def load_model():
    global model, target_encoder
    try:
        model = joblib.load(MODEL_PATH)
        target_encoder = joblib.load(ENCODER_PATH)
        print("Model dan encoder berhasil dimuat!")
        print("Mapping kelas:", list(target_encoder.classes_))
        return True
    except Exception as e:
        print(f"Error loading model/encoder: {e}")
        return False

# Initialize database
def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            patient_id TEXT,
            wbc_count REAL,
            platelet_count REAL,
            neutrophils REAL,
            lymphocytes REAL,
            mpv REAL,
            pdw REAL,
            hemoglobin REAL,
            hct REAL,
            prediction TEXT,
            confidence REAL,
            risk_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Save diagnosis to DB
def save_diagnosis(data, prediction, confidence, risk_level):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO diagnosis_history 
            (timestamp, patient_id, wbc_count, platelet_count, neutrophils, lymphocytes, 
             mpv, pdw, hemoglobin, hct, prediction, confidence, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('timestamp', datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()),
            data.get('patient_id', 'Unknown'),
            data['wbc_count'],
            data['platelet_count'],
            data['neutrophils'],
            data['lymphocytes'],
            data['mpv'],
            data['pdw'],
            data['hemoglobin'],
            data['hct'],
            prediction,
            confidence,
            risk_level
        ))
        conn.commit()
        conn.close()
        print("Diagnosis berhasil disimpan ke database.")
        return True
    except Exception as e:
        print(f"Error saving diagnosis: {e}")
        print(f"[DB ERROR] Gagal simpan diagnosis: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    try:
        data = request.json
        required_fields = [
            'wbc_count', 'platelet_count', 'neutrophils', 'lymphocytes',
            'mpv', 'pdw', 'hemoglobin', 'hct'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        input_data = pd.DataFrame({
            'Total WBC count(/cumm)': [data['wbc_count']],
            'Total Platelet Count(/cumm)': [data['platelet_count']],
            'Neutrophils(%)': [data['neutrophils']],
            'Lymphocytes(%)': [data['lymphocytes']],
            'MPV(fl)': [data['mpv']],
            'PDW(%)': [data['pdw']],
            'Hemoglobin(g/dl)': [data['hemoglobin']],
            'HCT(%)': [data['hct']]
        })

        prediction = model.predict(input_data)
        predicted_label = target_encoder.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_labels = list(target_encoder.classes_)

        result = {
            'prediction': predicted_label,
            'confidence': float(max(probabilities)),
            'probabilities': {
                class_labels[0]: float(probabilities[0]),
                class_labels[1]: float(probabilities[1])
            },
            'risk_level': get_risk_level(max(probabilities)),
            'timestamp': datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()

        }

        save_diagnosis(data, result['prediction'], result['confidence'], result['risk_level'])
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            print("[DEBUG] Tidak ada file")
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        print(f"[DEBUG] Nama file: {file.filename}")
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df = pd.read_csv(file)
        print("[DEBUG] Kolom asli:", df.columns.tolist())
        print("[DEBUG] 3 baris pertama:\n", df.head())
        column_mapping = {
            'wbc_count': 'Total WBC count(/cumm)',
            'platelet_count': 'Total Platelet Count(/cumm)',
            'neutrophils': 'Neutrophils(%)',
            'lymphocytes': 'Lymphocytes(%)',
            'mpv': 'MPV(fl)',
            'pdw': 'PDW(%)',
            'hemoglobin': 'Hemoglobin(g/dl)',
            'hct': 'HCT(%)'
        }
        df_renamed = df.rename(columns=column_mapping)
        print("[DEBUG] Kolom setelah rename:", df_renamed.columns.tolist())
        predictions = model.predict(df_renamed)
        predicted_labels = target_encoder.inverse_transform(predictions)
        probabilities = model.predict_proba(df_renamed)
        print("[DEBUG] Prediksi:", predictions)
        print("[DEBUG] Probabilitas:", probabilities)
        class_labels = list(target_encoder.classes_)

        results = []
        for i, (label, prob) in enumerate(zip(predicted_labels, probabilities)):
            results.append({
                'row': i + 1,
                'prediction': label,
                'confidence': float(max(prob)),
                'probabilities': {
                    class_labels[0]: float(prob[0]),
                    class_labels[1]: float(prob[1])
                },
                'risk_level': get_risk_level(max(prob))
            })

        positive_label = 'Positive' if 'Positive' in class_labels else class_labels[1]
        positive_count = sum(1 for r in results if r['prediction'] == positive_label)
        summary = {
            'total_samples': len(results),
            'positive_cases': positive_count,
            'negative_cases': len(results) - positive_count,
            'positive_rate': positive_count / len(results) * 100
        }

        return jsonify({
            'results': results,
            'summary': summary,
            'timestamp': datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM diagnosis_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        columns = [description[0] for description in cursor.description]
        history = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'history': history, 'total': len(history)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM diagnosis_history')
        total = cursor.fetchone()[0]

        cursor.execute('''
            SELECT prediction, COUNT(*) 
            FROM diagnosis_history 
            GROUP BY prediction
        ''')
        prediction_counts = dict(cursor.fetchall())

        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM diagnosis_history 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''')
        daily_activity = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        conn.close()

        return jsonify({
            'total_diagnoses': total,
            'prediction_distribution': prediction_counts,
            'daily_activity': daily_activity
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_risk_level(confidence):
    if confidence >= 0.8:
        return 'High'
    elif confidence >= 0.6:
        return 'Medium'
    else:
        return 'Low'
    
    
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)  # Ganti sesuai nama file DB kamu
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/history/clear', methods=['POST'])
def clear_history():
    conn = get_db_connection()
    conn.execute("DELETE FROM diagnosis_history")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared", "message": "Semua riwayat berhasil dihapus"})

if __name__ == '__main__':
    if not load_model():
        print("Error: Model tidak dapat dimuat!")
        exit(1)
    init_database()
    app.run(host='0.0.0.0', port=5000, debug=True)
