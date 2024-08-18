from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app1 = Flask(__name__)
app1.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///C:/Users/ashok/OneDrive/Desktop/Mini project reference/user.db"  # Change this to your SQL database URI
app1.config['SECRET_KEY'] = '1@3$5'  # Change this to a secure secret key
db = SQLAlchemy(app1)


# Define the User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Load the trained Random Forest model
rf_classifier = joblib.load('C:/Users/ashok/OneDrive/Desktop/Mini project reference/random_forest_model.pkl')

# Load the feature scaler
scaler = joblib.load('C:/Users/ashok/OneDrive/Desktop/Mini project reference/scaler.pkl')

# Define feature names
feature_names = ['anxiety', 'self_esteem', 'mental_health_history', 'depression', 'headache',
                 'blood_pressure', 'sleep_quality', 'breathing_problem', 'noise_level',
                 'living_conditions','safety', 'basic_needs', 'academic_performance', 'study_load',
                 'teacher_student_relationship', 'future_career_concerns', 'social_support',
                 'peer_pressure', 'extracurricular_activities', 'bullying']    

# Route for the signup page
@app1.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if the email is already registered
        if User.query.filter_by(email=email).first():
            flash('Email already exists!')
            return redirect(url_for('signup'))

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect(url_for('signup'))

        # Hash the password before saving to the database
        hashed_password = generate_password_hash(password)

        # Create a new user object and add it to the database
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('You have signed up successfully!')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Route for the login page
@app1.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()

        if user:
            # Check if the password is correct
            if check_password_hash(user.password, password):
                # Set the user's email in the session for authentication
                session['email'] = email
                flash('Logged in successfully!')
                return redirect(url_for('form'))
            else:
                flash('Incorrect password!')
        else:
            flash('User does not exist!')

    return render_template('login.html')


@app1.route('/form')
def form():
    if 'email' not in session:
        # If not authenticated, redirect to the login page
        return redirect(url_for('login'))
    # If authenticated, render the form.html page
    return render_template('form.html')

@app1.route('/submit', methods=['POST'])
def submit():
    # Get the form data
    form_data = request.form.to_dict()
    # Convert form data to a list of integers
    input_data = [int(form_data[feature]) for feature in feature_names]

    # Create a numpy array with the form data
    input_data_array = np.array([input_data])

    # Preprocess the input data
    scaled_input_data = scaler.transform(input_data_array)
    
    # Predict the stress level
    stress_level = rf_classifier.predict(scaled_input_data)[0]

    y=np.array([stress_level]*scaled_input_data.shape[0])
    # Calculate permutation importances
    result = permutation_importance(rf_classifier, scaled_input_data,y, n_repeats=10, random_state=42)

    # Get feature importances
    feature_importances = result.importances[1]

    # Find the feature with the highest permutation importance
    most_influential_feature_index = np.argmax(feature_importances)
    most_influential_feature = feature_names[most_influential_feature_index]

   
    return render_template('result.html', stress_level=stress_level, most_influential_feature=most_influential_feature)

if __name__ == '__main__':
    with app1.app_context():
        db.create_all()  # Create the database tables
    app1.run(debug=True)
