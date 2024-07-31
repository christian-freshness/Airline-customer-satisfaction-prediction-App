import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title='Customer Satisfaction Prediction', layout='wide')

# Function to inject CSS for background image
#def inject_css():
css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: #DCC7AA;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: scroll;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

div.st-emotion-cache-uzeiqp.e1nzilvr4 {{
    color: 	#ffffff;
}}

div.st-emotion-cache-asc41u.h3 {{
    color: rgb(0 0 0);
}}

div.object-key-val{{
    background-color: #DCC7AA;
}}

@media (max-width: 768px) {{
    [data-testid="stAppViewContainer"] {{
        background-size: cover;
        background-position: center;
    }}
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Call the CSS injection function
#inject_css()

# Load the trained model
model = joblib.load('gradboost_model.joblib')

# Label encoders for categorical features
customer_type_encoder = LabelEncoder()
customer_type_encoder.classes_ = np.array(['Disloyal Customer', 'Loyal Customer'])

travel_type_encoder = LabelEncoder()
travel_type_encoder.classes_ = np.array(['Personal Travel', 'Business travel'])

class_encoder = LabelEncoder()
class_encoder.classes_ = np.array(['Economy', 'Business', 'Economy Plus'])

delayed_encoder = LabelEncoder()
delayed_encoder.classes_ = np.array([ 'No', 'Yes'])

# Function to make predictions
def predict_satisfaction(input_data):
    prediction = model.predict([input_data])
    return 'SATISFIED' if prediction[0] == 1 else 'DISSATISFIED'

# Streamlit app
#st.set_page_config(page_title='Customer Satisfaction Prediction', layout='wide')
st.title('Airline Customer Satisfaction Prediction')
st.markdown("""
    This application predicts whether an Airline customer is **SATISFIED** or **DISSATISFIED** based on various input parameters.
""")

# Input form
st.sidebar.header('Enter Customer Details/Ratings')

with st.sidebar.form(key='customer_form'):
    customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'], help='Select if the customer is loyal or disloyal')
    age = st.number_input('Age', min_value=0,  max_value=130, help='Enter the age of the customer')
    type_of_travel = st.selectbox('Type of Travel', ['Personal Travel', 'Business travel'], help='Select the type of travel')
    travel_class = st.selectbox('Class', ['Economy', 'Business', 'Economy Plus'], help='Select the travel class')
    flight_distance = st.number_input('Flight Distance (in km)', min_value=0, help='Enter the flight distance in kilometers')
    seat_comfort = st.slider('Seat comfort', 0, 5, help='Rate the seat comfort')
    departure_arrival_convenience = st.slider('Departure/Arrival time convenient', 0, 5, help='Rate the convenience of departure/arrival time')
    food_drink = st.slider('Food and drink', 0, 5, help='Rate the food and drink service')
    gate_location = st.slider('Gate location', 0, 5, help='Rate the gate location convenience')
    inflight_wifi = st.slider('Inflight wifi service', 0, 5, help='Rate the inflight wifi service')
    inflight_entertainment = st.slider('Inflight entertainment', 0, 5, help='Rate the inflight entertainment')
    online_support = st.slider('Online support', 0, 5, help='Rate the online support')
    ease_online_booking = st.slider('Ease of Online booking', 0, 5, help='Rate the ease of online booking')
    onboard_service = st.slider('On-board service', 0, 5, help='Rate the on-board service')
    leg_room_service = st.slider('Leg room service', 0, 5, help='Rate the leg room service')
    baggage_handling = st.slider('Baggage handling', 0, 5, help='Rate the baggage handling service')
    checkin_service = st.slider('Check-in service', 0, 5, help='Rate the checkin service')
    cleanliness = st.slider('Cleanliness', 0, 5, help='Rate the cleanliness')
    online_boarding = st.slider('Online boarding', 0, 5, help='Rate the online boarding process')
    departure_delay = st.number_input('Departure Delay in Minutes', min_value=0, help='Enter the departure delay in minutes')
    arrival_delay = st.number_input('Arrival Delay in Minutes', min_value=0, help='Enter the arrival delay in minutes')
    delayed = st.selectbox('Delayed', ['Yes', 'No'], help='Select if the flight was delayed')
    
    submit_button = st.form_submit_button(label='Predict Satisfaction')

# Preprocess categorical features
if submit_button:
    customer_type_encoded = customer_type_encoder.transform([customer_type])[0]
    type_of_travel_encoded = travel_type_encoder.transform([type_of_travel])[0]
    class_encoded = class_encoder.transform([travel_class])[0]
    delayed_encoded = delayed_encoder.transform([delayed])[0]
    # Collect input data
    input_data = [customer_type_encoded, age, type_of_travel_encoded, class_encoded, flight_distance, seat_comfort, 
                  departure_arrival_convenience, food_drink, gate_location, inflight_wifi, inflight_entertainment, 
                  online_support, ease_online_booking, onboard_service, leg_room_service, baggage_handling, 
                  checkin_service, cleanliness, online_boarding, departure_delay, arrival_delay, delayed_encoded]

    # Predict and display the result
    result = predict_satisfaction(input_data)
    st.subheader('Prediction Result')
    st.write(f'**THE CUSTOMER IS {result}**')

    # Display input data for verification
    st.subheader('Customer Data')
    st.write({
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': type_of_travel,
        'Class': travel_class,
        'Flight Distance': flight_distance,
        'Seat comfort': seat_comfort,
        'Departure/Arrival time convenient': departure_arrival_convenience,
        'Food and drink': food_drink,
        'Gate location': gate_location,
        'Inflight wifi service': inflight_wifi,
        'Inflight entertainment': inflight_entertainment,
        'Online support': online_support,
        'Ease of Online booking': ease_online_booking,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Check-in service': checkin_service,
        'Cleanliness': cleanliness,
        'Online boarding': online_boarding,
        'Departure Delay in Minutes': departure_delay,
        'Arrival Delay in Minutes': arrival_delay,
        'Delayed': delayed
    })
