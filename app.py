%%writefile update_app.py
import streamlit as st
import pickle
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time

# Load necessary models and label mappings
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('agglo_model.pkl', 'rb') as f:
    aggloModel = pickle.load(f)

with open('agglo_cluster_label.pkl', 'rb') as f:
    agglo_cluster_to_label_map = pickle.load(f)

with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Read min and max values from CSV file
min_max_df = pd.read_csv('min_max_values.csv', index_col='Column')
min_max_values = min_max_df.to_dict(orient='index')

# Load the PCA-transformed training data saved in .npy format
X_train_pca = np.load("train_for_agglo.npy")

# Define feature lists
integer_cols = ['src_bytes', 'dst_bytes', 'wrong_fragment', 'num_compromised', 'count', 'srv_count', 'dst_host_srv_count', 'dst_host_same_src_port_rate']
nominal_cols = ['protocol_type', 'service', 'flag', 'logged_in', 'is_guest_login']

# Define options for features
protocol_types = ['tcp', 'udp', 'icmp']
services = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 
                'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link', 'remote_job', 'gopher', 'ssh', 'name', 
                'whois', 'domain', 'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 
                'printer', 'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat', 'supdup', 'iso_tsap', 
                'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm', 
                'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 
                'tim_i', 'red_i']
flags = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']

# Function to map cluster labels
def get_labels(cluster_labels, cluster_to_label_map):
    return [cluster_to_label_map.get(label, "Unknown") for label in cluster_labels]

# Function to handle unseen labels
def handle_unseen_labels(df, label_encoder, col_name):
    unseen_mask = ~df[col_name].isin(label_encoder.classes_)
    if unseen_mask.any():
        st.warning(f"Unseen labels found in column '{col_name}': {df[col_name][unseen_mask].unique()}")
        df.loc[unseen_mask, col_name] = -1
    df[col_name] = label_encoder.transform(df[col_name].astype(str))

# Function to validate numerical input
def get_valid_input(prompt, min_val, max_val):
    input_val = st.text_input(prompt, value=str(min_val))
    try:
        int_val = int(input_val)
        if int_val < min_val or int_val > max_val:
            st.error(f"Please enter a value between {min_val} and {max_val}.")
            return None
        return int_val
    except ValueError:
        st.error("Please enter a valid integer.")
        return None

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio id="themeAudio" autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Prediction", "Dashboard"))

if page == "Prediction":
    st.title("Network Attack Clustering - Prediction")

    option = st.radio("Select input method:", ('Single Data Input', 'CSV File Upload'))

    if option == 'Single Data Input':
        st.subheader(">>> Input the network features:")
        
        # User input for features
        protocol_type = st.selectbox('Protocol Type', protocol_types)
        service = st.selectbox('Network Service', services)
        flag = st.selectbox('Flag (Status of the Connection)', flags)

        logged_in = st.radio('Successful Logged In?', options=[('Successful (1)', 1), ('Otherwise (0)', 0)], format_func=lambda x: x[0])
        is_guest_login = st.radio('Login as a "guest"?', options=[('Guest Login (1)', 1), ('Otherwise (0)', 0)], format_func=lambda x: x[0])

        logged_in_value = logged_in[1]
        is_guest_login_value = is_guest_login[1]

        # Feature values
        src_bytes = get_valid_input('Source Bytes', int(min_max_values['src_bytes']['Min']), int(min_max_values['src_bytes']['Max']))
        dst_bytes = get_valid_input('Destination Bytes', int(min_max_values['dst_bytes']['Min']), int(min_max_values['dst_bytes']['Max']))
        wrong_fragment = st.slider('Wrong Fragment', min_value=int(min_max_values['wrong_fragment']['Min']), max_value=int(min_max_values['wrong_fragment']['Max']), step=1)
        num_compromised = st.slider('Number of Compromised', min_value=int(min_max_values['num_compromised']['Min']), max_value=int(min_max_values['num_compromised']['Max']), step=1)
        count = st.slider('Count (Number of connections to the same host)', min_value=int(min_max_values['count']['Min']), max_value=int(min_max_values['count']['Max']), step=1)
        srv_count = st.slider('Service Count (Number of connections to the same service)', min_value=int(min_max_values['srv_count']['Min']), max_value=int(min_max_values['srv_count']['Max']), step=1)
        dst_host_srv_count = st.slider('Destination Host Service Count', min_value=int(min_max_values['dst_host_srv_count']['Min']), max_value=int(min_max_values['dst_host_srv_count']['Max']), step=1)
        dst_host_same_src_port_rate = st.slider('Destination Host Same Source Port Rate', min_value=float(min_max_values['dst_host_same_src_port_rate']['Min']), max_value=float(min_max_values['dst_host_same_src_port_rate']['Max']), step=0.01)

        # Prepare data for prediction
        data = {
            'protocol_type': protocol_type,
            'service': service,
            'flag': flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'wrong_fragment': wrong_fragment,
            'logged_in': logged_in_value,
            'num_compromised': num_compromised,
            'is_guest_login': is_guest_login_value,
            'count': count,
            'srv_count': srv_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate
        }

        df = pd.DataFrame([data])

        # Apply label encoding for nominal columns
        for col in nominal_cols:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        # Apply log transformation to integer columns
        for col in integer_cols:
            df[col] = np.log(df[col] + 1)

        df = df.apply(pd.to_numeric, errors='coerce')

        # Perform PCA transformation
        df_pca = pca.transform(df)

        if st.button('Predict'):
            try:
                # Check for any missing (NaN) values in the DataFrame
                missing_data = df.isnull().sum()
                if missing_data.any():
                    st.write("Columns with missing values:")
                    st.write(missing_data[missing_data > 0])
                else:
                    distances = euclidean_distances(df_pca, X_train_pca)
                    nearest_train_idx = np.argmin(distances, axis=1)
                    agglo_test_predicted_labels = aggloModel.labels_[nearest_train_idx]
                    aggloPredictedTestStr = [agglo_cluster_to_label_map[cluster] for cluster in agglo_test_predicted_labels]

                    predicted_label = aggloPredictedTestStr[0]

                    if "neptune." in predicted_label.lower():
                        autoplay_audio("imperial_march.mp3")
                        time.sleep(1)
                        highlighted_label = f'<span style="color:red; font-size:24px;">{predicted_label}</span>'
                        st.markdown(f"Predicted Attack: {highlighted_label}", unsafe_allow_html=True)
                    else:
                        st.success(f"Predicted Attack: {predicted_label}")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

    else:  # CSV File Upload
        st.subheader(">>> Upload your CSV file for prediction:")
        uploaded_file = st.file_uploader("Choose a file...", type="csv")

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(df_upload.head())

                # Apply preprocessing similar to the above
                for col in nominal_cols:
                    if col in label_encoders:
                        df_upload[col] = label_encoders[col].transform(df_upload[col])

                for col in integer_cols:
                    df_upload[col] = np.log(df_upload[col] + 1)

                df_upload = df_upload.apply(pd.to_numeric, errors='coerce')

                # Perform PCA transformation
                df_upload_pca = pca.transform(df_upload)

                if st.button('Predict'):
                    distances = euclidean_distances(df_upload_pca, X_train_pca)
                    nearest_train_idx = np.argmin(distances, axis=1)
                    agglo_test_predicted_labels = aggloModel.labels_[nearest_train_idx]
                    aggloPredictedTestStr = [agglo_cluster_to_label_map[cluster] for cluster in agglo_test_predicted_labels]

                    df_upload['Predicted Label'] = aggloPredictedTestStr
                    st.write(df_upload)

            except Exception as e:
                st.error(f"Error processing the CSV file: {str(e)}")

elif page == "Dashboard":
  

    # Function to load the saved clustering model from the pickle file
    def load_model(path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model

    # Function to load the cluster label map from the pickle file
    def load_cluster_label_map(path):
        with open(path, 'rb') as file:
            label_map = pickle.load(file)
        return label_map

    # Function to perform PCA transformation (if not already done)
    def apply_pca(X, n_components=2):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca

    # Streamlit app title
    st.title("Clustering PCA Visualization Dashboard")

    # Add a select box for users to choose a clustering model
    model_choice = st.selectbox("Select the clustering model:", 
                                 ["DBSCAN", "HDBSCAN", "Agglomerative", "Mean Shift"])
    
    # Model and label map paths based on selection
    if model_choice == "DBSCAN":
        model_path = 'dbscan_model.pkl'
        cluster_label_map_path = 'dbscan_cluster_label.pkl'
    elif model_choice == "HDBSCAN":
        model_path = 'hdbscan_model.pkl'
        cluster_label_map_path = 'hdbscan_cluster_label.pkl'
    elif model_choice == "Agglomerative":
        model_path = 'agglo_model.pkl'
        cluster_label_map_path = 'agglo_cluster_label.pkl'
    elif model_choice == "Mean Shift":
        model_path = 'ms_model.pkl'
        cluster_label_map_path = 'ms_cluster_label.pkl'
    
    # Load the selected clustering model and cluster label map
    clustering_model = load_model(model_path)
    cluster_label_map = load_cluster_label_map(cluster_label_map_path)

    # Assuming you have PCA-transformed data saved or can be loaded again
    X_pca_path = 'train_pca.npy'  # Your saved PCA-transformed data in a numpy file
    X_pca = np.load(X_pca_path)

    # Predict cluster labels using the selected clustering model
    try:
        if model_choice in ["Mean Shift"]:
            predicted_labels = clustering_model.predict(X_pca)  # Use fit_predict for these models
        else:
            predicted_labels = clustering_model.fit_predict(X_pca)  # Use fit_predict for other models too
    except Exception as e:
        st.error(f"An error occurred while predicting: {e}")
        st.stop()

    # Map numeric cluster labels to descriptive labels using the corresponding label map
    descriptive_labels = [cluster_label_map.get(label, "Unknown") for label in predicted_labels]
    
    # Create a DataFrame for visualization
    df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df['Cluster'] = descriptive_labels  # Use descriptive labels instead of numeric labels
    
    # Plotly interactive scatter plot with hover showing descriptive labels
    fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster', 
                 title=f'PCA Plot - {model_choice} Clusters',
                 color_discrete_sequence=px.colors.qualitative.Set1)

    # Update the legend to show descriptive labels
    fig.for_each_trace(lambda t: t.update(name=t.name.split('=')[-1], legendgroup=t.name.split('=')[-1]))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
