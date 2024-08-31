import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import bcrypt
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from textblob import TextBlob
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Set Streamlit page config (only once, at the beginning)
st.set_page_config(page_title="Calories Burn Prediction", page_icon="🔥", layout="centered")

# Load model
rfr = pickle.load(open('rfr.pkl', 'rb'))
x_train = pd.read_csv('X_train.csv')

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Function to verify hashed password
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

# Function to generate OTP
def generate_otp():
    return random.randint(1000, 9999)

# Function to send OTP via email
def send_otp_email(email, otp):
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = email
    msg['Subject'] = "OTP for Signup Confirmation"

    body = f"Your OTP for signup is: {otp}"
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Error sending OTP: {str(e)}")
        return False


# Function to send OTP via SMS using Twilio
def send_otp_sms(phone_number, otp):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    # Ensure phone number includes +91 prefix if not already included
    if not phone_number.startswith('+91'):
        phone_number = '+91' + phone_number.lstrip('0')  # Remove leading '0' if present

    try:
        message = client.messages.create(
            body=f"Your OTP for signup is: {otp}",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return message.sid
    except TwilioRestException as e:
        st.error(f"Error sending OTP: {str(e)}")
        return None


# Function to predict calories
def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
    prediction = rfr.predict(features).reshape(1, -1)
    return prediction[0]

# Function to load feedback data
def load_feedback_data(username, feedback_collection):
    return list(feedback_collection.find({"username": username}, {"_id": 0, "username": 0}))

# Function to save feedback data
def save_feedback_data(username, feedback_text, feedback_collection):
    feedback_collection.insert_one({"username": username, "feedback": feedback_text, "timestamp": datetime.now()})

# Function to clear feedback data (for admin)
def clear_feedback_data(feedback_collection):
    feedback_collection.delete_many({})

# Function to check if user exists
def user_exists(username, users_collection):
    return users_collection.find_one({"username": username})

# Function to check if user is admin
def is_admin(username, users_collection):
    user = users_collection.find_one({"username": username})
    return user and user.get("role") == "admin"

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Function to setup MongoDB
def setup_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["calorie_tracker"]
    users_collection = db["users"]
    calories_collection = db["calories"]
    feedback_collection = db["feedback"]

    # Check if admin exists; if not, prompt to create admin credentials
    admin = users_collection.find_one({"username": "admin"})
    if not admin:
        st.subheader("Admin Setup")
        admin_username = st.text_input("Admin Username", value="admin")
        admin_password = st.text_input("Admin Password", type="password")

        if st.button("Setup Admin"):
            hashed_password = hash_password(admin_password)
            users_collection.insert_one({"username": admin_username, "password": hashed_password, "role": "admin"})
            st.success("Admin credentials set up successfully!")

    return client, db, users_collection, calories_collection, feedback_collection

# Function to load calorie data for admin plot
def load_calorie_data(calories_collection):
    data = list(calories_collection.find({}, {"_id": 0, "username": 1, "calories": 1}))
    df = pd.DataFrame(data)
    return df

# Function to display admin app page with plot
def admin_app_page(username, x_train, calories_collection, feedback_collection, users_collection):
    st.title("Admin Module")
    st.write("Welcome to the Admin Module. Here you can manage feedback, analyze sentiment, and more.")
    st.sidebar.title("Admin Menu")

    if st.sidebar.button("View Feedback"):
        feedback_data = list(feedback_collection.find({}, {"_id": 0}))

        if feedback_data:
            feedback_df = pd.DataFrame(feedback_data)
            feedback_df['sentiment'] = feedback_df['feedback'].apply(analyze_sentiment)

            st.subheader("Feedback Analysis")
            st.write("Total Feedback Count:", len(feedback_df))
            st.write("Positive Feedback Count:", len(feedback_df[feedback_df['sentiment'] == 'positive']))
            st.write("Neutral Feedback Count:", len(feedback_df[feedback_df['sentiment'] == 'neutral']))
            st.write("Negative Feedback Count:", len(feedback_df[feedback_df['sentiment'] == 'negative']))

            st.write("Feedback Details:")
            st.dataframe(feedback_df[['username', 'feedback', 'timestamp', 'sentiment']])
        else:
            st.write("No feedback available yet.")

    if st.sidebar.button("Clear Feedback"):
        clear_feedback_data(feedback_collection)
        st.success("All feedback cleared successfully!")

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state.pop('username', None)
        st.experimental_rerun()

    # Plotting username vs calorie burnt on button click
    if st.sidebar.button("Plot for Admin"):
        st.subheader("Username vs Calorie Burnt")
        calorie_data = load_calorie_data(calories_collection)

        if not calorie_data.empty:
            fig = px.bar(calorie_data, x='username', y='calories', title='Username vs Calorie Burnt')
            st.plotly_chart(fig)
        else:
            st.write("No calorie data available yet.")

# Admin login page
def admin_login_page(client, db, users_collection):
    st.subheader("Admin Login")
    admin_username = st.text_input("Admin Username", key="admin_username")
    admin_password = st.text_input("Admin Password", type="password", key="admin_password")

    if st.button("Admin Login"):
        admin = users_collection.find_one({"username": admin_username})
        if admin and verify_password(admin['password'], admin_password) and admin['role'] == 'admin':
            st.session_state['username'] = admin_username
            st.experimental_rerun()  # Rerun the app to switch to main app page
        else:
            st.error("Invalid admin credentials")

# User login page
def user_login_page(client, db, users_collection):
    st.subheader("User Login")
    username_input = st.text_input("Username", key="user_username")
    password_input = st.text_input("Password", type="password", key="user_password")

    if st.button("User Login"):
        user = user_exists(username_input, users_collection)
        if user and verify_password(user['password'], password_input) and user['role'] == 'user':
            st.session_state['username'] = username_input
            st.experimental_rerun()  # Rerun the app to switch to main app page
        else:
            st.error("Invalid username or password")

# Function to validate email format
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@(gmail|outlook|yahoo)\.(com|org|net)$'
    return re.match(pattern, email.lower()) is not None

# Function to validate password format
def validate_password(password):
    if len(password) < 4:
        return False

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in '@$#!%*?&' for c in password)

    return has_upper and has_lower and has_digit and has_special

# User signup page with email OTP functionality
def user_signup_email_page(client, db, users_collection):
        st.subheader("User Signup")
        email = st.text_input("Email", key="new_user_email")
        new_username = st.text_input("New Username", key="new_user_username")
        new_password = st.text_input("New Password", type="password", key="new_user_password")
        otp_input = st.text_input("Enter OTP", key="otp_input")

        otp_sent = st.session_state.get('otp_sent', False)
        if st.button("Send OTP") and validate_email(email):
            otp = generate_otp()
            if send_otp_email(email, otp):
                st.session_state['otp_sent'] = True
                st.session_state['otp'] = otp  # Store OTP in session state
                st.success("OTP sent successfully to your email.")
            else:
                st.error("Failed to send OTP. Please try again.")

        if st.button("Signup"):
            if not validate_email(email):
                st.error("Please enter a valid email ending with @gmail.com, @outlook.com, or @yahoo.com.")
                return
            if not otp_sent or otp_input != str(st.session_state.get('otp', '')):
                st.error("Please enter the OTP sent to your email.")
                return
            if user_exists(new_username, users_collection):
                st.error("Username already exists")
            elif not validate_password(new_password):
                st.error(
                    "Password must be at least 4 characters long and contain at least one uppercase letter, one lowercase letter, one digit, and one special character (@$!%*?&).")
            else:
                hashed_password = hash_password(new_password)
                users_collection.insert_one(
                    {"username": new_username, "password": hashed_password, "role": "user", "email": email})
                st.success("Signup successful! Please login.")


def user_signup_phone_page(client, db, users_collection):
    st.subheader("User Signup with Phone OTP")
    new_username = st.text_input("Username", key="phone_signup_username")
    new_password = st.text_input("Password", type="password", key="phone_signup_password")
    phone_number = st.text_input("Phone Number", key="phone_number")
    otp_input = st.text_input("Enter OTP", key="phone_otp_input")

    otp_sent = st.session_state.get('phone_otp_sent', False)

    # Send OTP button logic
    if st.button("Send OTP"):
        if not new_username or not new_password or not phone_number:
            st.error("Please fill in all fields.")
        elif not re.match(r'^\d{10}$', phone_number):
            st.error("Invalid phone number format. Please enter a 10-digit phone number.")
        elif not validate_password(new_password):
            st.error("Password must have at least 4 characters, including uppercase, lowercase, digits, and special characters.")
        else:
            hashed_password = hash_password(new_password)
            user_data = {
                "username": new_username,
                "password": hashed_password,
                "phone_number": phone_number,
                "role": "user"
            }
            if user_exists(new_username, users_collection):
                st.error("Username already exists. Please choose a different username.")
            else:
                users_collection.insert_one(user_data)
                st.success("OTP to be sent")

                # Send OTP
                otp = generate_otp()
                sent = send_otp_sms(phone_number, otp)
                if sent:
                    st.success("OTP sent to your phone number. Please check and verify.")
                    st.session_state['phone_otp_sent'] = True
                    st.session_state['phone_otp'] = otp
                    st.session_state['phone_signup_user'] = new_username
                else:
                    st.error("Failed to send OTP. Please try again.")

    # Signup button logic
    if st.button("Signup"):
        if not otp_sent:
            st.error("Please send OTP first.")
        elif otp_input != str(st.session_state.get('phone_otp', '')):
            st.error("Incorrect OTP. Please try again.")
        else:
            # Register user in database after OTP verification
            hashed_password = hash_password(new_password)
            users_collection.insert_one({"username": new_username, "password": hashed_password, "role": "user", "phone_number": phone_number})
            st.success("Signup successful! Please login.")

# Function to handle user signup based on selection (email or phone)
def user_signup_handler(client, db, users_collection):
    st.subheader("User Signup")
    signup_option = st.radio("Select Signup Method:", ("Email", "Phone Number"))

    if signup_option == "Email":
        user_signup_email_page(client, db, users_collection)
    elif signup_option == "Phone Number":
        user_signup_phone_page(client, db, users_collection)

from datetime import datetime

def user_app_page(username, x_train, calories_collection, feedback_collection, users_collection):
    st.title("User Module")

    Gender = st.selectbox('Gender', ['male', 'female'])
    Age = st.selectbox('Age', sorted(x_train['Age'].unique()))
    Height = st.selectbox('Height', sorted(x_train['Height'].unique()))
    Weight = st.selectbox('Weight', sorted(x_train['Weight'].unique()))
    Duration = st.slider('Duration (minutes)', 0, 120, 30)
    Heart_rate = st.slider('Heart Rate (bpm)', 60, 180, 120)
    Body_temp = st.slider('Body Temperature (°F)', 95, 105, 98)

    # Convert selected gender label to numeric value
    Gender_numeric = 1 if Gender == 'female' else 0

    if st.button('Predict'):
        result = pred(Gender_numeric, Age, Height, Weight, Duration, Heart_rate, Body_temp)
        result_float = float(result[0])  # Convert to float from NumPy array

        # Save daily calorie burn to MongoDB
        today = datetime.now().strftime('%Y-%m-%d')
        calories_collection.insert_one({"username": username, "date": today, "calories": result_float})

        st.write("You have burnt this many calories:", result_float)

        # Plot predicted calorie dynamically
        st.subheader("Predicted Calorie Burnt")
        fig = px.bar(x=[f"Calories Burned"], y=[result_float], labels={'x': '', 'y': 'Calories Burned'})
        fig.update_yaxes(range=[0, 400])  # Set the y-axis range to 0-400 calories
        st.plotly_chart(fig)

    # Feedback section (unchanged)
    st.sidebar.title("Feedback")
    feedback = st.sidebar.text_area("Comments", placeholder="Enter your comments here...")

    if st.sidebar.button("Submit Feedback"):
        if feedback.strip():  # Check for non-empty feedback
            save_feedback_data(username, feedback, feedback_collection)
            st.sidebar.write("Thank you for your feedback!")
        else:
            st.sidebar.write("Please enter some feedback before submitting.")

    if st.sidebar.button("View Feedback"):
        feedback_data = load_feedback_data(username, feedback_collection)
        if feedback_data:
            st.sidebar.subheader("Your Feedback")
            feedback_table = []
            for entry in feedback_data:
                feedback_table.append({'Feedback': entry['feedback'], 'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})
            st.sidebar.table(feedback_table)
        else:
            st.sidebar.write("No feedback submitted yet.")

    # Button to toggle historical calorie burn data display
    if st.sidebar.button("View Historical Calorie Data"):
        st.sidebar.subheader("Historical Calorie Burn Data")
        user_calories_data = list(calories_collection.find({"username": username}, {"_id": 0, "username": 0}))

        if user_calories_data:
            df_calories = pd.DataFrame(user_calories_data)
            st.sidebar.dataframe(df_calories)

            st.sidebar.subheader("Summary")
            st.sidebar.write("Total Entries:", len(df_calories))
            st.sidebar.write("Total Calories Burned:", df_calories['calories'].sum())
        else:
            st.sidebar.write("No historical calorie data available yet.")

    if st.sidebar.button("Logout"):
        st.session_state.pop('username', None)
        st.experimental_rerun()

# Main application logic
def run_app():
    client, db, users_collection, calories_collection, feedback_collection = setup_mongodb()

    try:
        # Check if admin or user login is needed
        if 'username' not in st.session_state:
            st.title("Calorie Burnt Prediction🔥")

            # Sidebar navigation
            page = st.sidebar.radio(
                "Navigate",
                ["Home", "Signup using email","Signup using Phone number", "Login", "Admin Login"]
            )

            # Display pages based on selection
            if page == "Home":
                st.write("""
                    Welcome to your personalized Calorie Burnt Prediction app! 🚀

                    Whether you're striving for fitness goals or simply curious about your daily calorie burn, 
                    our app provides you with accurate predictions based on your activities and inputs.

                    - **Signup**: New to our app? Sign up to start tracking and predicting your calorie burn.
                    - **Login**: Already a member? Log in to continue your journey towards a healthier lifestyle.
                    - **Admin Login**: Admins can manage user feedback and enhance app functionalities.

                    Start exploring and make every step count! Let's achieve your fitness goals together.
                    """)

            elif page == "Signup using email":
                user_signup_email_page(client, db, users_collection)

            elif page == "Signup using Phone number":
                user_signup_phone_page(client, db, users_collection)

            elif page == "Login":
                user_login_page(client, db, users_collection)

            elif page == "Admin Login":
                admin_login_page(client, db, users_collection)

        elif is_admin(st.session_state['username'], users_collection):
            admin_app_page(st.session_state['username'], x_train, calories_collection, feedback_collection,
                           users_collection)
        else:
            user_app_page(st.session_state['username'], x_train, calories_collection, feedback_collection,
                          users_collection)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


# Run the app
if __name__ == "__main__":
    run_app()