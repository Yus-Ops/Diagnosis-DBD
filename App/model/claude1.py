# Random Forest Pipeline - Template Lengkap
# Tinggal ganti bagian yang ditandai dengan # GANTI INI

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BAGIAN YANG PERLU DIGANTI
# =============================================================================

# GANTI INI: Nama file dataset
FILE_NAME = "Dengue.csv"  # Ganti dengan nama file Anda

# GANTI INI: Nama kolom target/dependent variable
TARGET_COLUMN = "Result"  # Ganti dengan nama kolom target Anda

# GANTI INI: Daftar fitur/independent variables yang akan digunakan
# Kosongkan list jika ingin menggunakan semua kolom kecuali target
FEATURES = [
    "Total WBC count(/cumm)", 
    "Total Platelet Count(/cumm)", 
    "Neutrophils(%)", 
    "Lymphocytes(%)", 
    "MPV(fl)", 
    "PDW(%)", 
    "Hemoglobin(g/dl)", 
    "HCT(%)"
]
  # Contoh: ["feature1", "feature2", "feature3"] atau kosongkan []

# GANTI INI: Jenis problem - "classification" atau "regression"
PROBLEM_TYPE = "classification"  # atau "regression"

# GANTI INI: Parameter untuk tuning (opsional)
TUNE_HYPERPARAMETERS = True  # True jika ingin melakukan hyperparameter tuning

# GANTI INI: Handling imbalanced dataset
HANDLE_IMBALANCED = True  # True jika dataset tidak seimbang
BALANCING_METHOD = "oversample"  # "class_weight", "oversample", "undersample"

# GANTI INI: Custom threshold untuk classification (default 0.5)
CUSTOM_THRESHOLD = 0.5  # None untuk otomatis, atau nilai seperti 0.3

# =============================================================================
# FUNGSI UTILITY
# =============================================================================

def load_and_explore_data(file_name):
    """Memuat dan mengeksplorasi dataset"""
    print("="*50)
    print("MEMUAT DAN MENGEKSPLORASI DATA")
    print("="*50)
    
    # Memuat data
    df = pd.read_csv(file_name)
    print(f"Shape dataset: {df.shape}")
    print(f"Kolom tersedia: {list(df.columns)}")
    
    # Info dasar
    print("\nInfo Dataset:")
    print(df.info())
    
    # Statistik deskriptif
    print("\nStatistik Deskriptif:")
    print(df.describe())
    
    # Missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("Tidak ada missing values")
    
    return df

def handle_imbalanced_data(X, y, method="class_weight"):
    """Handle imbalanced dataset"""
    print(f"\nDistribusi kelas sebelum balancing:")
    print(Counter(y))
    
    if method == "oversample":
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"Distribusi kelas setelah SMOTE oversampling:")
        print(Counter(y_balanced))
        return X_balanced, y_balanced, None
        
    elif method == "undersample":
        undersampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        print(f"Distribusi kelas setelah undersampling:")
        print(Counter(y_balanced))
        return X_balanced, y_balanced, None
        
    elif method == "class_weight":
        print("Menggunakan class_weight='balanced'")
        return X, y, "balanced"
    
    return X, y, None

def find_optimal_threshold(model, X_val, y_val):
    """Mencari threshold optimal untuk classification"""
    # Prediksi probabilitas
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Hitung precision-recall untuk berbagai threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    
    # F1-score untuk setiap threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Threshold dengan F1-score tertinggi
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"Threshold optimal: {optimal_threshold:.3f}")
    print(f"F1-score optimal: {f1_scores[optimal_idx]:.3f}")
    
    return optimal_threshold

def preprocess_data(df, target_col, features_list, problem_type):
     """Preprocessing data"""
     print("\n" + "="*50)
     print("PREPROCESSING DATA")
     print("="*50)

    # Pilih fitur
     if not features_list:
        features_list = [col for col in df.columns if col != target_col]

     print(f"Fitur yang digunakan: {features_list}")
     print(f"Target variable: {target_col}")

     # Pisahkan fitur dan target
     X = df[features_list].copy()
     y = df[target_col].copy()

     # Handle missing values
     if X.isnull().sum().sum() > 0:
        print("Mengisi missing values dengan median/mode...")
        for col in X.columns:
            if X[col].dtype in ['object']:
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)

    # Encode categorical features
     label_encoders = {}
     categorical_cols = X.select_dtypes(include=['object']).columns

     if len(categorical_cols) > 0:
        print(f"Encoding categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Encode target jika classification dan berbentuk string
     target_encoder = None
     if problem_type == "classification" and y.dtype == 'object':
        print("Encoding target variable...")
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        print("Target Classes:", list(target_encoder.classes_))

     print(f"Shape setelah preprocessing - X: {X.shape}, y: {y.shape}")
     return X, y, label_encoders, target_encoder

     """Preprocessing data"""
     print("\n" + "="*50)
     print("PREPROCESSING DATA")
     print("="*50)
    
    # Pilih fitur
     if not features_list:  # Jika list kosong, gunakan semua kolom kecuali target
        features_list = [col for col in df.columns if col != target_col]
    
     print(f"Fitur yang digunakan: {features_list}")
     print(f"Target variable: {target_col}")
    
    # Pisahkan fitur dan target
     X = df[features_list].copy()
     y = df[target_col].copy()
    
    # Handle missing values (simple imputation)
     if X.isnull().sum().sum() > 0:
        print("Mengisi missing values dengan median/mode...")
        for col in X.columns:
            if X[col].dtype in ['object']:
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)
    
    # Encode categorical variables
     label_encoders = {}
     categorical_cols = X.select_dtypes(include=['object']).columns
    
     if len(categorical_cols) > 0:
        print(f"Encoding categorical variables: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Encode target jika classification dan berbentuk string
     target_encoder = None
     if problem_type == "classification" and y.dtype == 'object':
        print("Encoding target variable...")
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    
     print(f"Shape setelah preprocessing - X: {X.shape}, y: {y.shape}")
    
     return X, y, label_encoders, target_encoder

def train_random_forest(X, y, problem_type, tune_params=True, handle_imbalanced=True, balance_method="class_weight"):
    """Training Random Forest dengan atau tanpa hyperparameter tuning"""
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST")
    print("="*50)
    
    # Handle imbalanced data jika classification
    class_weight_param = None
    if problem_type == "classification" and handle_imbalanced:
        X, y, class_weight_param = handle_imbalanced_data(X, y, balance_method)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if problem_type == "classification" else None
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Pilih model berdasarkan jenis problem
    if problem_type == "classification":
        base_model = RandomForestClassifier(
            random_state=42, 
            class_weight=class_weight_param
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        scoring = 'f1' if len(np.unique(y)) == 2 else 'f1_macro'
    else:
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        scoring = 'r2'
    
    if tune_params:
        print("Melakukan hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        print("Training dengan parameter default...")
        best_model = base_model
        best_model.fit(X_train, y_train)
    
    return best_model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, problem_type, feature_names, custom_threshold=None):
    """Evaluasi model"""
    print("\n" + "="*50)
    print("EVALUASI MODEL")
    print("="*50)
    
    if problem_type == "classification":
        # Cari threshold optimal jika tidak ditentukan
        if custom_threshold is None:
            # Split validation set untuk threshold tuning
            X_val = X_test[:len(X_test)//2]
            y_val = y_test[:len(y_test)//2]
            X_test_final = X_test[len(X_test)//2:]
            y_test_final = y_test[len(y_test)//2:]
            
            optimal_threshold = find_optimal_threshold(model, X_val, y_val)
        else:
            optimal_threshold = custom_threshold
            X_test_final = X_test
            y_test_final = y_test
        
        # Prediksi dengan threshold optimal
        train_pred = model.predict(X_train)
        test_proba = model.predict_proba(X_test_final)[:, 1]
        test_pred = (test_proba >= optimal_threshold).astype(int)
        
        # Basic metrics
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test_final, test_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Threshold yang digunakan: {optimal_threshold:.3f}")
        
        print("\nClassification Report (dengan threshold optimal):")
        print(classification_report(y_test_final, test_pred))
        
        # ROC AUC Score
        if len(np.unique(y_test_final)) == 2:  # Binary classification
            roc_auc = roc_auc_score(y_test_final, test_proba)
            print(f"ROC AUC Score: {roc_auc:.4f}")
            
            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test_final, test_proba)
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            # Plot Precision-Recall Curve
            precision, recall, thresholds = precision_recall_curve(y_test_final, test_proba)
            plt.subplot(1, 3, 2)
            plt.plot(recall, precision)
            plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_final, test_pred)
        if len(np.unique(y_test_final)) == 2:
            plt.subplot(1, 3, 3)
        else:
            plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if len(np.unique(y_test_final)) == 2:
            plt.tight_layout()
        plt.show()
        
        # Analisis per kelas
        print("\n" + "="*30)
        print("ANALISIS DETAIL PER KELAS")
        print("="*30)
        
        # Class distribution
        print("Distribusi kelas di test set:")
        unique, counts = np.unique(y_test_final, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = count / len(y_test_final) * 100
            print(f"Kelas {cls}: {count} samples ({percentage:.1f}%)")
        
        # Performance per class
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(y_test_final, test_pred)
        
        print("\nPerformance per kelas:")
        for i, cls in enumerate(unique):
            print(f"Kelas {cls}:")
            print(f"  Precision: {precision[i]:.3f}")
            print(f"  Recall: {recall[i]:.3f}")
            print(f"  F1-Score: {f1[i]:.3f}")
            print(f"  Support: {support[i]}")
            
    else:
        # Prediksi untuk regression
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics untuk regression
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Scatter plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
    
    return feature_importance

def cross_validation_score(model, X, y, problem_type):
    """Cross validation"""
    print("\n" + "="*50)
    print("CROSS VALIDATION")
    print("="*50)
    
    scoring = 'accuracy' if problem_type == "classification" else 'r2'
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    
    print(f"Cross Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Fungsi utama"""
    try:
        # 1. Load dan explore data
        df = load_and_explore_data(FILE_NAME)
        
        # 2. Preprocessing
        X, y, label_encoders, target_encoder = preprocess_data(
            df, TARGET_COLUMN, FEATURES, PROBLEM_TYPE
        )
        
        # 3. Training
        model, X_train, X_test, y_train, y_test = train_random_forest(
            X, y, PROBLEM_TYPE, TUNE_HYPERPARAMETERS, HANDLE_IMBALANCED, BALANCING_METHOD
        )
        
        # 4. Evaluasi
        feature_importance = evaluate_model(
            model, X_train, X_test, y_train, y_test, PROBLEM_TYPE, X.columns, CUSTOM_THRESHOLD
        )
        
        # 5. Cross validation
        cross_validation_score(model, X, y, PROBLEM_TYPE)
        
        print("\n" + "="*50)
        print("TRAINING SELESAI!")
        print("="*50)
        print("Model berhasil dilatih dan dievaluasi.")
                # Simpan model ke file .joblib
        import joblib
        joblib.dump(target_encoder, 'app/model/target_encoder.pkl')
        joblib.dump(model, 'app/model/dbd_model.joblib')
        print("Model berhasil disimpan di: app/model/dbd_model.joblib")

        print("Anda dapat menggunakan 'model' untuk prediksi baru.")
        
        # Return objects yang mungkin dibutuhkan
        return {
            'model': model,
            'feature_importance': feature_importance,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Pastikan:")
        print("1. File dataset tersedia")
        print("2. Nama kolom target dan fitur sudah benar")
        print("3. Problem type sudah sesuai")

# Jalankan program
if __name__ == "__main__":
    results = main()

# =============================================================================
# CONTOH PENGGUNAAN UNTUK PREDIKSI BARU
# =============================================================================

def predict_new_data(model, new_data, label_encoders=None):
    """
    Fungsi untuk memprediksi data baru
    
    Parameters:
    model: trained model
    new_data: DataFrame dengan data baru
    label_encoders: dictionary label encoders untuk categorical variables
    """
    
    # Encode categorical variables jika ada
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in new_data.columns:
                new_data[col] = encoder.transform(new_data[col])
    
    # Prediksi
    predictions = model.predict(new_data)
    probabilities = None
    
    if hasattr(model, 'predict_proba'):  # Untuk classification
        probabilities = model.predict_proba(new_data)
    
    return predictions, probabilities


# Contoh penggunaan prediksi:
# new_sample = pd.DataFrame({'feature1': [value1], 'feature2': [value2], ...})
# pred, prob = predict_new_data(results['model'], new_sample, results['label_encoders'])
