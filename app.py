from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import sqlite3
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import requests
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for ML model
ml_model = None
disease_info = None
model_loaded = False

def load_model():
    """Load the disease detection model"""
    global ml_model, disease_info, model_loaded
    
    try:
        if os.path.exists('models/plant_disease_model.h5'):
            ml_model = tf.keras.models.load_model('models/plant_disease_model.h5')
            print("✅ Disease detection model loaded successfully")
        else:
            print("⚠️ Model file not found. Please run train_model.py first")
            return False
            
        if os.path.exists('models/disease_info.pkl'):
            with open('models/disease_info.pkl', 'rb') as f:
                disease_info = pickle.load(f)
            print("✅ Disease info loaded successfully")
        else:
            print("⚠️ Disease info file not found")
            return False
            
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()

# Enhanced pesticide database for chatbot
PESTICIDE_DATABASE = {
    'tomato': {
        'diseases': ['Early Blight', 'Late Blight', 'Bacterial Spot', 'Septoria Leaf Spot'],
        'pesticides': [
            {
                'name': 'Chlorothalonil',
                'type': 'Fungicide',
                'dosage': '2g/liter',
                'frequency': 'Every 7-10 days',
                'price_range': '₹150-200/100g',
                'effectiveness': '85%'
            },
            {
                'name': 'Copper Oxychloride',
                'type': 'Bactericide',
                'dosage': '3g/liter',
                'frequency': 'Every 15 days',
                'price_range': '₹80-120/100g',
                'effectiveness': '78%'
            },
            {
                'name': 'Neem Oil',
                'type': 'Organic',
                'dosage': '5ml/liter',
                'frequency': 'Weekly',
                'price_range': '₹60-100/100ml',
                'effectiveness': '70%'
            }
        ],
        'organic_alternatives': [
            {
                'name': 'Neem Oil Spray',
                'ingredients': 'Neem oil, liquid soap, water',
                'preparation': 'Mix 5ml neem oil + 2ml soap in 1L water',
                'effectiveness': '70%'
            },
            {
                'name': 'Garlic-Chili Spray',
                'ingredients': 'Garlic, green chili, soap, water',
                'preparation': 'Blend 100g garlic + 50g chili, strain, add soap',
                'effectiveness': '65%'
            },
            {
                'name': 'Baking Soda Solution',
                'ingredients': 'Baking soda, liquid soap, water',
                'preparation': '1 tsp baking soda + 2ml soap in 1L water',
                'effectiveness': '60%'
            }
        ]
    },
    'rice': {
        'diseases': ['Blast', 'Bacterial Blight', 'Sheath Blight', 'Brown Spot'],
        'pesticides': [
            {
                'name': 'Tricyclazole',
                'type': 'Fungicide',
                'dosage': '1g/liter',
                'frequency': 'At tillering stage',
                'price_range': '₹200-300/100g',
                'effectiveness': '90%'
            },
            {
                'name': 'Kasugamycin',
                'type': 'Bactericide',
                'dosage': '2ml/liter',
                'frequency': 'At symptom appearance',
                'price_range': '₹180-250/100ml',
                'effectiveness': '85%'
            },
            {
                'name': 'Validamycin',
                'type': 'Fungicide',
                'dosage': '2ml/liter',
                'frequency': 'Early infection stage',
                'price_range': '₹160-220/100ml',
                'effectiveness': '82%'
            }
        ],
        'organic_alternatives': [
            {
                'name': 'Panchagavya',
                'ingredients': 'Cow dung, urine, milk, curd, ghee',
                'preparation': 'Ferment for 15 days, dilute 1:10',
                'effectiveness': '75%'
            },
            {
                'name': 'Jeevamrutha',
                'ingredients': 'Cow dung, urine, jaggery, gram flour',
                'preparation': 'Mix and ferment for 7 days',
                'effectiveness': '70%'
            }
        ]
    },
    'cotton': {
        'diseases': ['Boll Rot', 'Leaf Curl Virus', 'Fusarium Wilt', 'Bacterial Blight'],
        'pesticides': [
            {
                'name': 'Carbendazim',
                'type': 'Fungicide',
                'dosage': '1g/liter',
                'frequency': 'Pre-flowering stage',
                'price_range': '₹120-180/100g',
                'effectiveness': '88%'
            },
            {
                'name': 'Imidacloprid',
                'type': 'Insecticide',
                'dosage': '0.5ml/liter',
                'frequency': 'At sowing',
                'price_range': '₹200-280/100ml',
                'effectiveness': '85%'
            }
        ],
        'organic_alternatives': [
            {
                'name': 'Neem Seed Extract',
                'ingredients': 'Neem seeds, water',
                'preparation': 'Soak 100g seeds overnight, grind and strain',
                'effectiveness': '68%'
            }
        ]
    },
    'wheat': {
        'diseases': ['Rust', 'Smut', 'Powdery Mildew', 'Leaf Blight'],
        'pesticides': [
            {
                'name': 'Propiconazole',
                'type': 'Fungicide',
                'dosage': '1ml/liter',
                'frequency': 'At flowering',
                'price_range': '₹180-250/100ml',
                'effectiveness': '92%'
            },
            {
                'name': 'Tebuconazole',
                'type': 'Fungicide',
                'dosage': '0.5ml/liter',
                'frequency': 'Early growth stage',
                'price_range': '₹220-300/100ml',
                'effectiveness': '89%'
            }
        ],
        'organic_alternatives': [
            {
                'name': 'Cow Urine Spray',
                'ingredients': 'Fresh cow urine, water',
                'preparation': 'Dilute 1:10 with water',
                'effectiveness': '65%'
            }
        ]
    }
}

# Enhanced shop database with real-like data
PESTICIDE_SHOPS = [
    {
        'name': 'Green Earth Agri Solutions',
        'location': 'Pune',
        'address': 'Shop 15, Agricultural Market, Pune-411001',
        'phone': '+91-9876543210',
        'specialties': ['Organic Pesticides', 'Bio-fertilizers'],
        'crops_served': ['Tomato', 'Cotton', 'Sugarcane'],
        'distance': '2.5 km',
        'rating': 4.5,
        'open_hours': '8:00 AM - 7:00 PM',
        'delivery': True
    },
    {
        'name': 'Krishak Seeds & Pesticides',
        'location': 'Nashik',
        'address': 'Main Road, Nashik-422001',
        'phone': '+91-9876543211',
        'specialties': ['Chemical Pesticides', 'Seeds'],
        'crops_served': ['Rice', 'Wheat', 'Cotton'],
        'distance': '5.1 km',
        'rating': 4.2,
        'open_hours': '7:30 AM - 8:00 PM',
        'delivery': False
    },
    {
        'name': 'Organic Farm Store',
        'location': 'Mumbai',
        'address': 'Andheri East, Mumbai-400069',
        'phone': '+91-9876543212',
        'specialties': ['100% Organic', 'Neem Products'],
        'crops_served': ['All Crops'],
        'distance': '8.3 km',
        'rating': 4.8,
        'open_hours': '9:00 AM - 6:00 PM',
        'delivery': True
    },
    {
        'name': 'Government Agriculture Store',
        'location': 'Nagpur',
        'address': 'Civil Lines, Nagpur-440001',
        'phone': '+91-9876543213',
        'specialties': ['Subsidized Pesticides', 'Government Schemes'],
        'crops_served': ['Rice', 'Wheat', 'Soybean'],
        'distance': '12.7 km',
        'rating': 4.0,
        'open_hours': '10:00 AM - 5:00 PM',
        'delivery': False
    },
    {
        'name': 'Farm Fresh Solutions',
        'location': 'Kolhapur',
        'address': 'Market Yard, Kolhapur-416001',
        'phone': '+91-9876543214',
        'specialties': ['Integrated Pest Management', 'Consultation'],
        'crops_served': ['Sugarcane', 'Grapes', 'Cotton'],
        'distance': '15.2 km',
        'rating': 4.3,
        'open_hours': '8:30 AM - 7:30 PM',
        'delivery': True
    }
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/location')
def location():
    return render_template('location.html')

@app.route('/location/manual')
def location_manual():
    return render_template('location_manual.html')

@app.route('/crops')
def crops():
    return render_template('crops.html')

@app.route('/water')
def water():
    return render_template('water.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html')

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')

@app.route('/crop-analysis')
def crop_analysis():
    return render_template('profit.html')

@app.route('/profit')
def profit():
    return render_template('profit.html')

@app.route('/profit_prediction')
def profit_prediction():
    return redirect('/profit')

# Disease detection API
@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not model_loaded:
            return jsonify({'error': 'Disease detection model not loaded'}), 500
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image for prediction
        image = Image.open(filepath)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = ml_model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        disease_name = disease_info['classes'][predicted_class_index]
        
        # Generate treatment recommendations
        treatments = generate_treatment_recommendations(disease_name)
        
        return jsonify({
            'success': True,
            'disease_name': disease_name.replace('_', ' ').title(),
            'confidence': f"{confidence:.1f}",
            'treatments': treatments,
            'image_path': f"/static/uploads/{filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_treatment_recommendations(disease_name):
    """Generate treatment recommendations based on disease"""
    treatments = {
        'Tomato_Early_Blight': [
            'Remove and destroy infected leaves immediately',
            'Apply copper-based fungicide every 7-10 days',
            'Improve air circulation around plants',
            'Avoid overhead watering',
            'Use mulch to prevent soil splash'
        ],
        'Tomato_Late_Blight': [
            'Apply preventive fungicide before symptoms appear',
            'Remove infected plant parts immediately',
            'Ensure good drainage and air circulation',
            'Avoid working with wet plants',
            'Use resistant varieties in future plantings'
        ],
        'Powdery_Mildew': [
            'Apply sulfur-based fungicide',
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove infected leaves',
            'Use baking soda spray (1 tsp per liter)'
        ],
        'Healthy_Plant': [
            'Continue current care practices',
            'Monitor regularly for early disease signs',
            'Maintain proper watering schedule',
            'Ensure adequate nutrition',
            'Practice crop rotation'
        ]
    }
    
    return treatments.get(disease_name, [
        'Consult with local agricultural expert',
        'Remove affected plant parts',
        'Apply appropriate fungicide',
        'Improve plant care practices',
        'Monitor plant health regularly'
    ])

# Chatbot API endpoints
@app.route('/api/chatbot/pesticide-info', methods=['POST'])
def get_pesticide_info():
    """Get pesticide information for a specific crop"""
    try:
        data = request.get_json()
        crop = data.get('crop', '').lower()
        
        if crop in PESTICIDE_DATABASE:
            crop_data = PESTICIDE_DATABASE[crop]
            return jsonify({
                'success': True,
                'crop': crop.title(),
                'diseases': crop_data['diseases'],
                'pesticides': crop_data['pesticides'],
                'organic_alternatives': crop_data['organic_alternatives']
            })
        else:
            # Return general pesticide info
            return jsonify({
                'success': True,
                'crop': crop.title(),
                'message': f'Specific data for {crop} not available. Here are general recommendations:',
                'pesticides': [
                    {
                        'name': 'Neem Oil',
                        'type': 'Organic',
                        'dosage': '5ml/liter',
                        'frequency': 'Weekly',
                        'price_range': '₹60-100/100ml',
                        'effectiveness': '70%'
                    },
                    {
                        'name': 'Copper Fungicide',
                        'type': 'General',
                        'dosage': '3g/liter',
                        'frequency': 'Every 15 days',
                        'price_range': '₹80-120/100g',
                        'effectiveness': '75%'
                    }
                ],
                'organic_alternatives': [
                    {
                        'name': 'Garlic-Soap Spray',
                        'ingredients': 'Garlic, liquid soap, water',
                        'preparation': 'Blend 100g garlic in 1L water, add 2ml soap',
                        'effectiveness': '65%'
                    }
                ]
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chatbot/find-shops', methods=['POST'])
def find_pesticide_shops():
    """Find pesticide shops near a location"""
    try:
        data = request.get_json()
        location = data.get('location', '').lower()
        crop = data.get('crop', '').lower()
        
        # Filter shops by location
        matching_shops = []
        for shop in PESTICIDE_SHOPS:
            if (location in shop['location'].lower() or 
                shop['location'].lower() in location):
                matching_shops.append(shop)
        
        # If no exact matches, return all shops with distance simulation
        if not matching_shops:
            matching_shops = PESTICIDE_SHOPS[:3]  # Return top 3 shops
            for shop in matching_shops:
                shop['distance'] = f"{random.uniform(5, 25):.1f} km"
        
        # Filter by crop if specified
        if crop:
            filtered_shops = []
            for shop in matching_shops:
                if (crop in [c.lower() for c in shop['crops_served']] or 
                    'all crops' in [c.lower() for c in shop['crops_served']]):
                    filtered_shops.append(shop)
            if filtered_shops:
                matching_shops = filtered_shops
        
        return jsonify({
            'success': True,
            'location': location.title(),
            'shops': matching_shops[:5],  # Return max 5 shops
            'total_found': len(matching_shops)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chatbot/organic-alternatives', methods=['POST'])
def get_organic_alternatives():
    """Get organic pesticide alternatives"""
    try:
        data = request.get_json()
        crop = data.get('crop', '').lower()
        disease = data.get('disease', '').lower()
        
        organic_solutions = []
        
        if crop in PESTICIDE_DATABASE:
            organic_solutions = PESTICIDE_DATABASE[crop]['organic_alternatives']
        
        # Add general organic solutions
        general_organic = [
            {
                'name': 'Neem Oil Spray',
                'ingredients': 'Neem oil, liquid soap, water',
                'preparation': 'Mix 5ml neem oil + 2ml soap in 1L water',
                'effectiveness': '70%',
                'benefits': ['Broad spectrum', 'Safe for beneficial insects', 'No residue']
            },
            {
                'name': 'Soap Solution',
                'ingredients': 'Liquid soap, water',
                'preparation': '5ml liquid soap in 1L water',
                'effectiveness': '60%',
                'benefits': ['Cheap', 'Readily available', 'Safe']
            },
            {
                'name': 'Chili-Garlic Spray',
                'ingredients': 'Green chili, garlic, soap, water',
                'preparation': 'Blend 50g chili + 100g garlic, strain, add soap',
                'effectiveness': '65%',
                'benefits': ['Natural repellent', 'Homemade', 'Cost effective']
            }
        ]
        
        # Combine crop-specific and general solutions
        all_solutions = organic_solutions + general_organic
        
        return jsonify({
            'success': True,
            'crop': crop.title(),
            'organic_solutions': all_solutions[:6],  # Return max 6 solutions
            'tips': [
                'Apply in early morning or evening',
                'Test on small area first',
                'Reapply after rain',
                'Combine with good cultural practices'
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chatbot/emergency-contact', methods=['POST'])
def get_emergency_contact():
    """Get emergency agricultural contacts"""
    try:
        data = request.get_json()
        location = data.get('location', '').lower()
        
        emergency_contacts = [
            {
                'type': 'Agricultural Officer',
                'name': 'Dr. Rajesh Kumar',
                'phone': '+91-9876543220',
                'location': 'District Agriculture Office',
                'availability': '24/7 Emergency Line',
                'specialization': 'Crop Diseases & Pest Management'
            },
            {
                'type': 'Krishi Vigyan Kendra',
                'name': 'KVK Extension Officer',
                'phone': '+91-9876543221',
                'location': 'Local KVK Center',
                'availability': '9 AM - 6 PM',
                'specialization': 'Integrated Pest Management'
            },
            {
                'type': 'Pesticide Dealer',
                'name': 'Licensed Pesticide Consultant',
                'phone': '+91-9876543222',
                'location': 'Registered Dealer Network',
                'availability': '8 AM - 8 PM',
                'specialization': 'Pesticide Recommendations'
            }
        ]
        
        return jsonify({
            'success': True,
            'location': location.title(),
            'emergency_contacts': emergency_contacts,
            'helpline': {
                'name': 'Kisan Call Center',
                'phone': '1800-180-1551',
                'availability': '24/7 Toll Free',
                'languages': ['Hindi', 'English', 'Regional Languages']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting FarmAI Assistant...")
    print("📱 Disease Detection: http://localhost:5000/diseases")
    print("🤖 AI Chatbot: Available on diseases page")
    print("📅 Calendar: http://localhost:5000/calendar")
    print("📈 Profit Prediction: http://localhost:5000/profit")
    app.run(debug=True, host='0.0.0.0', port=5000)