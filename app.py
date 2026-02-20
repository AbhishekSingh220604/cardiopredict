from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for CardioPredict frontend (GitHub Pages)
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, specify your GitHub Pages URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ========================================
# LOAD MODEL
# ========================================
try:
    logger.info("Loading model...")
    with open('heart_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    model_name = model_data.get('model_name', 'Unknown')
    metrics = model_data.get('metrics', {})
    
    logger.info(f"✓ Model loaded successfully!")
    logger.info(f"  Model Type: {model_name}")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
    logger.info(f"  Features: {len(feature_names)}")
    
except FileNotFoundError:
    logger.error("❌ Model file 'heart_model.pkl' not found!")
    logger.error("   Please run 'python train_model.py' first.")
    model = None
    scaler = None
    feature_names = None
    model_name = None
    
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None
    scaler = None
    feature_names = None
    model_name = None

# Expected feature names in correct order
EXPECTED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# ========================================
# ROUTES
# ========================================

@app.route('/')
def home():
    """Root endpoint - API information"""
    return jsonify({
        'name': 'CardioPredict API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'model_type': model_name if model_name else 'Not loaded',
        'accuracy': f"{metrics.get('accuracy', 0):.4f}" if metrics else 'N/A',
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST)',
            'model_info': '/model-info (GET)'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'model_name': model_name if model_name else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Model information endpoint"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_name': model_name,
        'metrics': metrics,
        'features': feature_names,
        'num_features': len(feature_names)
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Prediction endpoint for CardioPredict Frontend
    
    Expected JSON format:
    {
        "age": 55,
        "sex": 1,
        "cp": 1,
        "trestbps": 140,
        "chol": 230,
        "fbs": 1,
        "restecg": 1,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 2.3,
        "slope": 1,
        "ca": 1,
        "thal": 2
    }
    
    Returns:
    {
        "prediction": 0 or 1,
        "probability": 0-100,
        "risk_level": "LOW" / "MODERATE" / "HIGH" / "VERY HIGH",
        "message": "Risk assessment message",
        "confidence": 0-100,
        "timestamp": ISO datetime
    }
    """
    
    # Handle OPTIONS request (CORS preflight)
    if request.method == 'OPTIONS':
        return '', 204
    
    # Check if model is loaded
    if model is None or scaler is None:
        logger.error("Prediction requested but model not loaded")
        return jsonify({
            'error': 'Model not loaded. Please contact administrator.',
            'status': 'error'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            logger.warning("No data provided in request")
            return jsonify({
                'error': 'No data provided. Please send JSON data.',
                'status': 'error'
            }), 400
        
        logger.info(f"Received prediction request: {data}")
        
        # Validate all required features are present
        missing_features = [f for f in EXPECTED_FEATURES if f not in data]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            return jsonify({
                'error': f'Missing required features: {missing_features}',
                'status': 'error'
            }), 400
        
        # Extract and validate features in correct order
        try:
            features = []
            for feature in EXPECTED_FEATURES:
                value = data[feature]
                
                # Convert to appropriate type
                if feature == 'oldpeak':
                    features.append(float(value))
                else:
                    features.append(int(value))
            
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type: {str(e)}")
            return jsonify({
                'error': 'Invalid data format. All features must be numeric.',
                'status': 'error'
            }), 400
        
        # Validate feature ranges
        validations = [
            (0 <= data['age'] <= 120, 'Age must be between 0 and 120'),
            (data['sex'] in [0, 1], 'Sex must be 0 (Female) or 1 (Male)'),
            (data['cp'] in [0, 1, 2, 3], 'Chest pain type must be 0-3'),
            (50 <= data['trestbps'] <= 300, 'Resting BP must be 50-300'),
            (100 <= data['chol'] <= 600, 'Cholesterol must be 100-600'),
            (data['fbs'] in [0, 1], 'Fasting blood sugar must be 0 or 1'),
            (data['restecg'] in [0, 1, 2], 'Resting ECG must be 0-2'),
            (60 <= data['thalach'] <= 220, 'Max heart rate must be 60-220'),
            (data['exang'] in [0, 1], 'Exercise angina must be 0 or 1'),
            (0 <= data['oldpeak'] <= 10, 'Oldpeak must be 0-10'),
            (data['slope'] in [0, 1, 2], 'Slope must be 0-2'),
            (data['ca'] in [0, 1, 2, 3, 4], 'CA must be 0-4'),
            (data['thal'] in [0, 1, 2], 'Thal must be 0-2')
        ]
        
        for is_valid, error_msg in validations:
            if not is_valid:
                logger.warning(f"Validation failed: {error_msg}")
                return jsonify({
                    'error': error_msg,
                    'status': 'error'
                }), 400
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = int(model.predict(features_scaled)[0])
        
        # Get probability
        probability_raw = model.predict_proba(features_scaled)[0][1]
        probability = float(probability_raw * 100)
        
        # Determine risk level and message
        if probability >= 75:
            risk_level = "VERY HIGH"
            message = "⚠️ Critical risk detected. Immediate medical consultation strongly recommended."
            color = "danger"
        elif probability >= 50:
            risk_level = "HIGH"
            message = "⚠️ High risk detected. Please consult a cardiologist soon."
            color = "warning"
        elif probability >= 25:
            risk_level = "MODERATE"
            message = "⚠️ Moderate risk detected. Regular health monitoring advised."
            color = "info"
        else:
            risk_level = "LOW"
            message = "✓ Low risk. Continue maintaining a healthy lifestyle."
            color = "success"
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability_raw - 0.5) * 200  # 0-100 scale
        
        # Log prediction
        logger.info(f"Prediction: {prediction} | Probability: {probability:.2f}% | Risk: {risk_level}")
        
        # Prepare response (format matches CardioPredict frontend expectations)
        response = {
            'prediction': prediction,
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'message': message,
            'confidence': round(confidence, 2),
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error during prediction.',
            'details': str(e),
            'status': 'error'
        }), 500

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error',
        'available_endpoints': ['/predict', '/health', '/model-info', '/']
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'status': 'error',
        'hint': 'Use POST for /predict endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == '__main__':
    port = 5000
    
    print("="*60)
    print("CARDIOPREDICT API")
    print("="*60)
    print()
    print(f"🚀 Starting Flask server...")
    print(f"📡 API available at: http://localhost:{port}")
    print(f"📝 Prediction endpoint: http://localhost:{port}/predict")
    print(f"💚 Health check: http://localhost:{port}/health")
    print()
    
    if model is None:
        print("⚠️  WARNING: Model not loaded!")
        print("   Please run: python train_model.py")
        print()
    else:
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Accuracy: {metrics.get('accuracy', 'N/A')}")
        print()
    
    print("Press CTRL+C to stop")
    print("="*60)
    print()
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )
