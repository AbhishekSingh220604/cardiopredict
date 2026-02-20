import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CARDIOPREDICT - HEART ATTACK PREDICTION MODEL")
print("="*60)
print()

# ========================================
# 1. LOAD AND EXPLORE DATA
# ========================================
print("📊 Step 1: Loading Kaggle heart.csv dataset...")
print("-" * 60)

try:
    df = pd.read_csv('heart.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape[0]} samples, {df.shape[1]} features")
    print()
except FileNotFoundError:
    print("❌ ERROR: heart.csv not found!")
    print("   Please download from: https://www.kaggle.com/datasets/arezaei81/heartcsv")
    print("   Place it in the same directory as this script.")
    exit(1)

print("Dataset Preview:")
print(df.head())
print()

print("Dataset Info:")
print(df.info())
print()

print("Statistical Summary:")
print(df.describe())
print()

# Check for missing values
print("Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values!")
print()

# Check target distribution
print("Target Distribution:")
print(df['target'].value_counts())
print(f"  Class 0 (No Disease): {(df['target'] == 0).sum()} ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  Class 1 (Disease):    {(df['target'] == 1).sum()} ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
print()

# ========================================
# 2. DATA PREPROCESSING
# ========================================
print("🔧 Step 2: Preprocessing data...")
print("-" * 60)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features: {list(X.columns)}")
print(f"Target: {y.name}")
print()

# Split data - stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Data split complete!")
print(f"  Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

# Feature scaling
print("⚖️  Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling complete!")
print()

# ========================================
# 3. MODEL TRAINING & COMPARISON
# ========================================
print("🤖 Step 3: Training multiple models...")
print("-" * 60)
print()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"✓ {name}")
    print(f"  Test Accuracy:  {accuracy:.4f}")
    print(f"  CV Accuracy:    {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  ROC AUC:        {roc_auc:.4f}" if roc_auc else "  ROC AUC:        N/A")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print()

# ========================================
# 4. SELECT BEST MODEL
# ========================================
print("🏆 Step 4: Selecting best model...")
print("-" * 60)

# Select based on F1 score (balanced metric for medical diagnosis)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
best_metrics = results[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"  Test Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"  CV Accuracy:    {best_metrics['cv_mean']:.4f} (±{best_metrics['cv_std']:.4f})")
print(f"  ROC AUC:        {best_metrics['roc_auc']:.4f}" if best_metrics['roc_auc'] else "")
print(f"  Precision:      {best_metrics['precision']:.4f}")
print(f"  Recall:         {best_metrics['recall']:.4f}")
print(f"  F1 Score:       {best_metrics['f1']:.4f}")
print()

# ========================================
# 5. DETAILED EVALUATION
# ========================================
print("📊 Step 5: Detailed evaluation of best model...")
print("-" * 60)
print()

y_pred = best_metrics['predictions']

print("Classification Report:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=['No Heart Disease', 'Heart Disease'],
    digits=4
))
print()

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()
print(f"True Negatives:  {cm[0][0]:3d} (Correctly predicted no disease)")
print(f"False Positives: {cm[0][1]:3d} (Incorrectly predicted disease)")
print(f"False Negatives: {cm[1][0]:3d} (Missed disease cases) ⚠️")
print(f"True Positives:  {cm[1][1]:3d} (Correctly predicted disease)")
print()

# Calculate additional metrics
specificity = cm[0][0] / (cm[0][0] + cm[0][1])
sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
print(f"Specificity: {specificity:.4f} (True Negative Rate)")
print(f"Sensitivity: {sensitivity:.4f} (True Positive Rate / Recall)")
print()

# ========================================
# 6. FEATURE IMPORTANCE (if available)
# ========================================
if hasattr(best_model, 'feature_importances_'):
    print("📈 Feature Importance:")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    print()

# ========================================
# 7. SAVE MODEL
# ========================================
print("💾 Step 6: Saving model...")
print("-" * 60)

model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': list(X.columns),
    'model_name': best_model_name,
    'metrics': {
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'f1': best_metrics['f1'],
        'roc_auc': best_metrics['roc_auc']
    }
}

with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved as 'heart_model.pkl'")
print()

# ========================================
# 8. TEST PREDICTION EXAMPLE
# ========================================
print("🧪 Step 7: Testing prediction function...")
print("-" * 60)

# Sample test case (high risk patient)
test_sample = {
    'age': 55,
    'sex': 1,
    'cp': 1,
    'trestbps': 140,
    'chol': 230,
    'fbs': 1,
    'restecg': 1,
    'thalach': 150,
    'exang': 1,
    'oldpeak': 2.3,
    'slope': 1,
    'ca': 1,
    'thal': 2
}

test_features = [test_sample[col] for col in X.columns]
test_scaled = scaler.transform([test_features])
prediction = best_model.predict(test_scaled)[0]
probability = best_model.predict_proba(test_scaled)[0][1] * 100

print("Test Sample:")
for key, value in test_sample.items():
    print(f"  {key:10s}: {value}")
print()
print(f"Prediction: {'HEART DISEASE' if prediction == 1 else 'NO DISEASE'}")
print(f"Probability: {probability:.2f}%")
print()

# ========================================
# SUMMARY
# ========================================
print("="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print()
print("Summary:")
print(f"  ✓ Dataset: {df.shape[0]} samples with {df.shape[1]-1} features")
print(f"  ✓ Best Model: {best_model_name}")
print(f"  ✓ Test Accuracy: {best_metrics['accuracy']:.4f}")
print(f"  ✓ F1 Score: {best_metrics['f1']:.4f}")
print(f"  ✓ Model saved: heart_model.pkl")
print()
print("Next Steps:")
print("  1. Test the API locally: python app.py")
print("  2. Deploy to Render")
print("  3. Update your CardioPredict frontend with the Render URL")
print()
print("🚀 Your AI model is ready for deployment!")
print("="*60)
