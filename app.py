
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load necessary models and label mappings
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ms_model.pkl', 'rb') as f:
    meanShiftModel = pickle.load(f)

with open('ms_cluster_label.pkl', 'rb') as f:
    ms_cluster_to_label_map = pickle.load(f)

# Define label mapping function
def get_labels(predicted_labels, label_map):
    return [label_map.get(label, 'Unknown') for label in predicted_labels]

# Read min and max values from CSV file
min_max_df = pd.read_csv('min_max_values.csv', index_col='Column')
min_max_values = min_max_df.to_dict(orient='index')

# Feature lists
integer_cols = ['src_bytes', 'dst_bytes', 'wrong_fragment', 'num_compromised', 'count', 'srv_count', 'dst_host_srv_count']
nominal_cols = ['protocol_type', 'service', 'flag']

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
st.title("Network Attack Clustering")

# User input for the features
st.subheader(">>> Input the network features :")

protocol_type = st.selectbox('Protocol Type', protocol_types)
service = st.selectbox('Network Service', services)
flag = st.selectbox('Flag (Status of the Connection', flags)

logged_in = st.radio(
    'Successful Logged In ?',
    options=[(f'Successful (1)', 1), (f'Otherwise (0)', 0)],
    format_func=lambda x: x[0]
)
is_guest_login = st.radio(
    'Login i a "guest" ? ',
    options=[(f'Guest Login (1)', 1), (f'Otherwise (0)', 0)],
    format_func=lambda x: x[0]
)
src_bytes = st.slider('Source Bytes', min_value=int(min_max_values['src_bytes']['Min']), max_value=int(min_max_values['src_bytes']['Max']), step=10)
dst_bytes = st.slider('Destination Bytes', min_value=int(min_max_values['dst_bytes']['Min']), max_value=int(min_max_values['dst_bytes']['Max']), step=10)
wrong_fragment = st.slider('Wrong Fragment', min_value=int(min_max_values['wrong_fragment']['Min']), max_value=int(min_max_values['wrong_fragment']['Max']), step=1)
num_compromised = st.slider('Number of Compromised', min_value=int(min_max_values['num_compromised']['Min']), max_value=int(min_max_values['num_compromised']['Max']), step=1)
count = st.slider('Count(Number of connections to the same host)', min_value=int(min_max_values['count']['Min']), max_value=int(min_max_values['count']['Max']), step=1)
srv_count = st.slider('Service Count(Number of connections to the same service)', min_value=int(min_max_values['srv_count']['Min']), max_value=int(min_max_values['srv_count']['Max']), step=1)
dst_host_srv_count = st.slider('Destination Host Service Count', min_value=int(min_max_values['dst_host_srv_count']['Min']), max_value=int(min_max_values['dst_host_srv_count']['Max']), step=1)

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
    'dst_host_srv_count': dst_host_srv_count
}

# Create a DataFrame with the input data
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

# Duplicate the DataFrame to ensure PCA can be applied
df_duplicated = pd.concat([df] * 2, ignore_index=True)

# Perform PCA
pca = PCA(n_components=min(2, df[integer_cols].shape[1]))  # Adjust the number of components
df_pca = pca.fit_transform(df_duplicated[integer_cols])

# Predict clusters using the MeanShift model
ms_predicted_labels = meanShiftModel.predict(df_pca[:1])  # Predict only on the first sample
print(ms_predicted_labels)
ms_predicted_labels_str = get_labels(ms_predicted_labels, ms_cluster_to_label_map)
print(ms_predicted_labels_str)
# Display results
if st.button('Predict'):
    # Convert list of labels to string and display only the first label
    first_label = ms_predicted_labels_str[0]
    st.write(f'MeanShift Cluster Label: {first_label}')
