import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load preprocessor, models, and label mapping
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('dbscan_model.pkl', 'rb') as f:
    dbscan_model = pickle.load(f)

with open('dbscan_cluster_label.pkl', 'rb') as f:
    dbscan_cluster_to_label_map = pickle.load(f)

# Define label mapping function
def get_labels(predicted_labels, label_map):
    return [label_map.get(label, 'Unknown') for label in predicted_labels]

# Load the data DataFrame to compute min and max values
data = pd.read_csv('path_to_your_data.csv')  # Adjust this path to where your data is stored

# Compute min and max values for relevant columns
min_max_values = {
    'src_bytes': (data['src_bytes'].min(), data['src_bytes'].max()),
    'dst_bytes': (data['dst_bytes'].min(), data['dst_bytes'].max()),
    'wrong_fragment': (data['wrong_fragment'].min(), data['wrong_fragment'].max()),
    'num_compromised': (data['num_compromised'].min(), data['num_compromised'].max()),
    'count': (data['count'].min(), data['count'].max()),
    'srv_count': (data['srv_count'].min(), data['srv_count'].max()),
    'dst_host_srv_count': (data['dst_host_srv_count'].min(), data['dst_host_srv_count'].max())
}

# Important features for prediction
nominal_cols = ['protocol_type', 'service', 'flag', 'attack_type']

protocol_types = data['protocol_type'].unique().tolist()
services = data['service'].unique().tolist()
flags = data['flag'].unique().tolist()
attack_types = data['attack_type'].unique().tolist()

# Streamlit app layout
st.title("Network Attack Prediction")

# User input for the features
st.subheader("Input the feature values")

# Sliders with min and max values from the data DataFrame
src_bytes = st.slider('Source Bytes', min_value=int(min_max_values['src_bytes'][0]), max_value=int(min_max_values['src_bytes'][1]), step=10)
dst_bytes = st.slider('Destination Bytes', min_value=int(min_max_values['dst_bytes'][0]), max_value=int(min_max_values['dst_bytes'][1]), step=10)
wrong_fragment = st.slider('Wrong Fragment', min_value=int(min_max_values['wrong_fragment'][0]), max_value=int(min_max_values['wrong_fragment'][1]), step=1)
num_compromised = st.slider('Number of Compromised', min_value=int(min_max_values['num_compromised'][0]), max_value=int(min_max_values['num_compromised'][1]), step=1)
count = st.slider('Count', min_value=int(min_max_values['count'][0]), max_value=int(min_max_values['count'][1]), step=1)
srv_count = st.slider('Service Count', min_value=int(min_max_values['srv_count'][0]), max_value=int(min_max_values['srv_count'][1]), step=1)
dst_host_srv_count = st.slider('Destination Host Service Count', min_value=int(min_max_values['dst_host_srv_count'][0]), max_value=int(min_max_values['dst_host_srv_count'][1]), step=1)

# User input for nominal and binary features
protocol_type = st.selectbox('Protocol Type', protocol_types)
service = st.selectbox('Service', services)
flag = st.selectbox('Flag', flags)
attack_type = st.selectbox('Attack Type', attack_types)

logged_in = st.radio('Logged In', [0, 1])
is_guest_login = st.radio('Is Guest Login', [0, 1])

# Prepare data for prediction
data = {
    'protocol_type': protocol_type,
    'service': service,
    'flag': flag,
    'src_bytes': src_bytes,
    'dst_bytes': dst_bytes,
    'wrong_fragment': wrong_fragment,
    'logged_in': logged_in,
    'num_compromised': num_compromised,
    'is_guest_login': is_guest_login,
    'count': count,
    'srv_count': srv_count,
    'dst_host_srv_count': dst_host_srv_count,
    'attack_type': attack_type
}

# Create a DataFrame with two duplicated rows
df = pd.DataFrame([data, data])

# Apply label encoding for nominal columns
for col in nominal_cols:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Apply log transformation to integer columns
for col in min_max_values.keys():
    if col in df.columns:
        df[col] = np.log(df[col] + 1)

# Apply scaling to integer columns
df[list(min_max_values.keys())] = scaler.transform(df[list(min_max_values.keys())])

# Predict clusters using the DBSCAN model
dbscan_predicted_labels = dbscan_model.fit_predict(df[list(min_max_values.keys())])

# Map cluster labels to original labels
dbscan_labels_in_string = get_labels(dbscan_predicted_labels, dbscan_cluster_to_label_map)

# Display results
if st.button('Predict'):
    # Convert list of labels to string and display only the first label
    first_label = dbscan_labels_in_string[0]
    st.write(f'Cluster Label: {first_label}')
