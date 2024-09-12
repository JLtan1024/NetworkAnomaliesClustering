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

# Important features for prediction
integer_cols = ['src_bytes', 'dst_bytes', 'wrong_fragment', 'num_compromised', 'count', 'srv_count', 'dst_host_srv_count']
nominal_cols = ['protocol_type', 'service', 'flag', 'attack_type']
binary_cols = ['logged_in', 'is_guest_login']

protocol_types = ['tcp', 'udp', 'icmp']
services = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 
                'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh', 'name', 
                'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 
                'printer', 'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat', 'supdup', 'iso_tsap', 
                'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm', 
                'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 
                'tim_i', 'red_i']
flags = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']
attack_types = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 
                'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 
                'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']

# Streamlit app layout
st.title("Network Attack Prediction")

# User input for the features
st.subheader("Input the feature values")

protocol_type = st.selectbox('Protocol Type', protocol_types)
service = st.selectbox('Service', services)
flag = st.selectbox('Flag', flags)
attack_type = st.selectbox('Attack Type', attack_types)

logged_in = st.radio('Logged In', [0, 1])
is_guest_login = st.radio('Is Guest Login', [0, 1])

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
    'attack_type': attack_type
}

# Create a DataFrame with two duplicated rows
df = pd.DataFrame([data, data])

# Apply label encoding for nominal columns
for col in nominal_cols:
    if col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

# Apply log transformation to integer columns
for col in integer_cols:
    df[col] = np.log(df[col] + 1)

# Apply scaling to integer columns
df[integer_cols] = scaler.transform(df[integer_cols])

# Predict clusters using the DBSCAN model
dbscan_predicted_labels = dbscan_model.fit_predict(df[integer_cols])

# Map cluster labels to original labels
dbscan_labels_in_string = get_labels(dbscan_predicted_labels, dbscan_cluster_to_label_map)

# Display results
if st.button('Predict'):
    # Convert list of labels to string and display only the first label
    first_label = dbscan_labels_in_string[0]
    st.write(f'Cluster Label: {first_label}')
