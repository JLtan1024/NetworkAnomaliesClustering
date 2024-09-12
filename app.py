import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load preprocessor and model
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Important features for prediction
important_features = [
    'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'wrong_fragment', 'logged_in', 'num_compromised', 'is_guest_login',
    'count', 'srv_count', 'dst_host_srv_count', 'dst_host_same_src_port_rate'
]

integer_cols = ['src_bytes', 'dst_bytes', 'wrong_fragment', 'num_compromised', 'count', 'srv_count', 'dst_host_srv_count']
nominal_cols = ['protocol_type', 'service', 'flag']
binary_cols = ['logged_in', 'is_guest_login']

# Unique values for nominal features
protocol_types = ['tcp', 'udp', 'icmp']
services = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 
            'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh', 'name', 
            'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 
            'printer', 'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat', 'supdup', 'iso_tsap', 
            'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm', 
            'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 
            'tim_i', 'red_i']
flags = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']

# Streamlit app layout
st.title("Network Attack Prediction")

# User input for the features
st.subheader("Input the feature values")

# Nominal (categorical) columns using selectbox
protocol_type = st.selectbox('Protocol Type', protocol_types)
service = st.selectbox('Service', services)
flag = st.selectbox('Flag', flags)

# Binary columns using radio buttons
logged_in = st.radio('Logged In', [0, 1])
is_guest_login = st.radio('Is Guest Login', [0, 1])

# Integer columns using sliders
src_bytes = st.slider('Source Bytes', min_value=0, max_value=5000, step=10)
dst_bytes = st.slider('Destination Bytes', min_value=0, max_value=5000, step=10)
wrong_fragment = st.slider('Wrong Fragment', min_value=0, max_value=10, step=1)
num_compromised = st.slider('Number of Compromised', min_value=0, max_value=10, step=1)
count = st.slider('Count', min_value=0, max_value=500, step=1)
srv_count = st.slider('Service Count', min_value=0, max_value=500, step=1)
dst_host_srv_count = st.slider('Destination Host Service Count', min_value=0, max_value=500, step=1)

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
}

df = pd.DataFrame([data])

# Apply label encoding for nominal columns
for col in nominal_cols:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Apply log transformation to integer columns
for col in integer_cols:
    df[col] = np.log(df[col] + 1)

# Apply scaling to integer columns
df[integer_cols] = scaler.transform(df[integer_cols])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(df)
    if prediction == 1:
        st.error('The input data is classified as an attack!')
    else:
        st.success('The input data is classified as normal.')
