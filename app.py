from flask import Flask, request, redirect, render_template, url_for, flash, jsonify, session, send_from_directory
from pymongo import MongoClient
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
import time
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
# MongoDB setup
client = MongoClient('mongodb+srv://madhurapantulaabhi5:ArrX41A4230798160@@cluster0.ysos8jj.mongodb.net/')
db = client['users']

# Define collections
users_collection = db['users']
scan_history_collection = db['scan_history']

# Load the trained model
MODEL_PATH = 'resnet50_mobilenetv2_lstm_model.keras'
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# Load class names from the model
class_names = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust',
    'Apple_healthy', 'Blueberry_healthy', 'Cherry_healthy',
    'Cherry_Powdery_mildew', 'Corn_Cercospora_leaf_spot',
    'Corn_Common_rust', 'Corn_healthy', 'Corn_Northern_Leaf_Blight',
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)',
    'Grape__healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)',
    'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot',
    'Peach_healthy', 'Pepper,_bell_Bacterial_spot',
    'Pepper,bell_healthy', 'Potato__Early_blight',
    'Potato_healthy', 'Potato_Late_blight',
    'Raspberry_healthy', 'Soybean_healthy',
    'Squash_Powdery_mildew', 'Strawberry_healthy',
    'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 'Tomato_healthy',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
    'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Custom filter for formatting timestamps
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# Routes
@app.route('/')
def plant():
    if 'username' in session:
        user = users_collection.find_one({'username': session['username']})
        return render_template('plant.html', user=user)
    return render_template('plant.html', user=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_it_works')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/scan-history')
def scan_history():
    if 'username' not in session:
        return jsonify({"error": "User not logged in"}), 401  # Unauthorized

    username = session['username']
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 5))

    # Fetch paginated scan history records for this user
    history = list(scan_history_collection.find({'username': username})
                 .sort("timestamp", -1)
                 .skip(offset)
                 .limit(limit))

    # Convert MongoDB data to JSON serializable format
    for entry in history:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string

    return jsonify(history)

@app.route('/scan_history_page')
def scan_history_page():
    if 'username' not in session:
        flash("Please log in to view your history.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    
    # Fetch all scan history for this user
    history = list(scan_history_collection.find({'username': username}).sort("timestamp", -1))
    
    return render_template('scan_history.html', history=history)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        flash('Please log in first.', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Check if file is part of the request
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):  # Updated line
            # Save the file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Make prediction
            try:
                disease_name, confidence, top_predictions = predict_image(file_path)
            except Exception as e:
                flash(f"An error occurred during analysis: {e}", 'error')
                return redirect(request.url)

            # Save scan to history collection
            scan_entry = {
                'username': session['username'],
                'image_filename': unique_filename,
                'predicted_class': disease_name,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'timestamp': time.time()
            }
            scan_history_collection.insert_one(scan_entry)

            return render_template('upload.html', prediction=disease_name, confidence=confidence, image_filename=unique_filename)
    return render_template('upload.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = users_collection.find_one({'email': email, 'password': password})
        if user:
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for('upload'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    fullname = request.form.get('fullname')
    
    if username and password and email and fullname:
        # Check if username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Try another one!', 'error')
            return redirect(url_for('signup'))
        
        # Insert into MongoDB with timestamp
        users_collection.insert_one({
            'username': username,
            'password': password,
            'email': email,
            'fullname': fullname,
            'created_at': time.time()
        })
        flash('Signup successful! Please log in.', 'success')
        return redirect('/login')

    return "Invalid input, please try again."

# Signup page (GET)
@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

# Handle AJAX upload
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Please log in first'}), 401
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        
        # Predict the disease
        try:
            disease_name, confidence, top_predictions = predict_image(file_path)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'An error occurred during analysis: {e}'}), 500
        
        # Store the result in session
        session['prediction_result'] = {
            'disease': disease_name,
            'confidence': confidence,
            'image_path': file_path,
            'top_predictions': top_predictions,
            'timestamp': time.time()
        }
        
        # Save scan to history
        scan_entry = {
            'username': session['username'],
            'image_filename': unique_filename,
            'predicted_class': disease_name,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'timestamp': time.time()
        }
        scan_history_collection.insert_one(scan_entry)
        
        return jsonify({
            'status': 'success',
            'message': f'Disease detected: {disease_name}',
            'disease': disease_name,
            'confidence': confidence,
            'image_url': f'/uploads/{os.path.basename(file_path)}',
            'top_predictions': top_predictions,
            'redirect_url': url_for('upload')
        })
    
    return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

# Result display route
@app.route('/result')
def result():
    if 'username' not in session:
        flash("Please log in to view results.", "error")
        return redirect(url_for('login'))
    
    # Check if we have an ID parameter (viewing from history)
    result_id = request.args.get('id')
    if result_id:
        # Convert string ID to ObjectId for MongoDB query
        from bson.objectid import ObjectId
        scan_result = scan_history_collection.find_one({"_id": ObjectId(result_id)})
        
        if scan_result and scan_result['username'] == session['username']:
            image_url = f'/uploads/{scan_result["image_filename"]}'
            return render_template('result.html', 
                                  disease=scan_result['predicted_class'], 
                                  confidence=scan_result['confidence'], 
                                  image_url=image_url,
                                  top_predictions=scan_result['top_predictions'])
        else:
            flash("Scan result not found or unauthorized.", "error")
            return redirect(url_for('profile'))
    
    # Check for session stored result
    if 'prediction_result' not in session:
        flash("No prediction result found. Please upload an image first.", "error")
        return redirect(url_for('upload'))
    
    result = session['prediction_result']
    
    # Check if the result is recent (within 10 minutes)
    if time.time() - result['timestamp'] > 600:  # 600 seconds = 10 minutes
        flash("Your prediction result has expired. Please upload a new image.", "error")
        session.pop('prediction_result', None)
        return redirect(url_for('upload'))
    
    image_url = f'/uploads/{os.path.basename(result["image_path"])}'
    
    return render_template('result.html', 
                          disease=result['disease'], 
                          confidence=result['confidence'], 
                          image_url=image_url,
                          top_predictions=result['top_predictions'])

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('prediction_result', None)
    flash('logged out', 'success')
    return redirect(url_for('plant'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'username' not in session:
        flash("log in to access your profile.", "error")
        return redirect(url_for('login'))  

    # Get the logged-in user's username
    username = session['username']

    # Fetch user details from the users collection
    user = users_collection.find_one({'username': username})

    # Handle profile picture upload
    if request.method == 'POST' and 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{username}_{file.filename}")
            filepath = os.path.join('static/uploads/', filename)
            file.save(filepath)

            # Update user's profile picture in MongoDB
            users_collection.update_one({'username': username}, {'$set': {'profile_picture': filename}})
            flash("Profile picture updated successfully!", "success")
            return redirect(url_for('profile'))

    # Fetch the latest 5 scan history records for this user
    user_history = list(scan_history_collection.find({'username': username}).sort("timestamp", -1).limit(5))

    # Pass both `user` and `user_history` to the template
    return render_template('profile.html', user=user, history=user_history)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'username' not in session:
        flash("Please log in to update your profile.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    fullname = request.form.get('fullname')
    email = request.form.get('email')
    location = request.form.get('location')
    bio = request.form.get('bio')
    
    # Basic validation
    if not fullname or not email:
        flash("Full name and email are required.", "error")
        return redirect(url_for('profile'))
    
    # Check if email already exists with a different user
    existing_user = users_collection.find_one({'email': email, 'username': {'$ne': username}})
    if existing_user:
        flash("Email is already in use by another account.", "error")
        return redirect(url_for('profile'))
    
    # Update user in MongoDB
    users_collection.update_one(
        {'username': username},
        {'$set': {
            'fullname': fullname, 
            'email': email,
            'location': location,
            'bio': bio
        }}
    )
    
    flash("Profile updated successfully!", "success")
    return redirect(url_for('profile'))

@app.route('/update_profile_picture', methods=['POST'])
def update_profile_picture():
    if 'username' not in session:
        flash("Please log in to update your profile picture.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    
    if 'profile_picture' not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for('profile'))
    
    file = request.files['profile_picture']
    
    if file.filename == '':
        flash("No file selected.", "error")
        return redirect(url_for('profile'))
    
    if file and allowed_file(file.filename):
        # Generate a unique filename for the profile picture
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"profile_{username}_{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Update the user's profile picture in MongoDB
        users_collection.update_one(
            {'username': username},
            {'$set': {'profile_picture': filename}}
        )
        
        flash("Profile picture updated successfully!", "success")
    else:
        flash("Invalid file type. Please upload a JPG, PNG, or GIF image.", "error")
    
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'username' not in session:
        flash("Please log in to change your password.", "error")
        return redirect(url_for('login'))
    
    username = session['username']
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Basic validation
    if not current_password or not new_password or not confirm_password:
        flash("All fields are required.", "error")
        return redirect(url_for('profile'))
    
    if new_password != confirm_password:
        flash("New password and confirmation don't match.", "error")
        return redirect(url_for('profile'))
    
    # Verify current password
    user = users_collection.find_one({'username': username, 'password': current_password})
    if not user:
        flash("Current password is incorrect.", "error")
        return redirect(url_for('profile'))
    
    # Update password in MongoDB
    users_collection.update_one(
        {'username': username},
        {'$set': {'password': new_password}}
    )
    
    flash("Password changed successfully!", "success")
    return redirect(url_for('profile'))

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

# Prediction function
def predict_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    top_predictions = []
    for i in range(len(predictions[0])):
        top_predictions.append((class_names[i], predictions[0][i] * 100))
    top_predictions.sort(key=lambda x: x[1], reverse=True)
    return predicted_class_name, confidence, top_predictions

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
