import streamlit as st
import pickle
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

def autoplay_audio2(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio id="themeAudio" autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Streamlit app layout
st.title("Network Attack Clustering")

# Radio button for selecting input method
option = st.radio("Select input method:", ('Single Data Input', 'CSV File Upload'))

if option == 'Single Data Input':
    # User input for the features
    st.subheader(">>> Input the network features:")

    protocol_type = st.selectbox('Protocol Type', protocol_types)
    service = st.selectbox('Network Service', services)
    flag = st.selectbox('Flag (Status of the Connection)', flags)

    logged_in = st.radio(
        'Successful Logged In?',
        options=[('Successful (1)', 1), ('Otherwise (0)', 0)],
        format_func=lambda x: x[0]
    )

    is_guest_login = st.radio(
        'Login as a "guest"?',
        options=[('Guest Login (1)', 1), ('Otherwise (0)', 0)],
        format_func=lambda x: x[0]
    )

    logged_in_value = logged_in[1]
    is_guest_login_value = is_guest_login[1]

    # Feature values
    src_bytes = None
    while src_bytes is None:
        src_bytes = get_valid_input('Source Bytes', int(min_max_values['src_bytes']['Min']), int(min_max_values['src_bytes']['Max']))

    dst_bytes = None
    while dst_bytes is None:
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

                main_bg = "bg.jpg"
                main_bg_ext = "jpg"

                if "neptune." in predicted_label.lower():
                    autoplay_audio("steinsgateshort.mp3")
                    autoplay_audio("kiraslaugh.mp3")
                    time.sleep(1)
                    highlighted_label = f'<span style="color:red; font-size:24px;">{predicted_label}</span>'
                    with open(main_bg, "rb") as f:
                        data = f.read()
                        b64 = base64.b64encode(data).decode()
                        style = f"""
                            <style>
                            .stApp {{
                                background: url(data:image/jpg;base64,{b64});
                                background-size: contain;
                            }}
                            .css-15tx938, .stMarkdown, .css-10trblm, .st-bz, .css-1dx1gwv {{
                                color: white;
                            }}
                            body {{
                                background-color: #ADD8E6;
                            }}
                            </style>
                            """
                else:
                    highlighted_label = predicted_label
                    style = f"""
                            <style>
                            .stApp {{
                                background: none;
                                background-size: cover;
                            }}
                            .css-15tx938,[data-testid="stHeadingWithActionElements"], [data-testid="stMarkdownContainer"], .st-emotion-cache-jkfxgf .stMarkdown, .css-10trblm, .st-bz, .css-1dx1gwv{{
                                color: black;
                            }}
                            body {{
                                background-color: #FFFFFF;
                            }}
                            </style>
                            """
                st.markdown(
                    style,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'Predicted cluster label: {agglo_test_predicted_labels}',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'Predicted cluster label string: {highlighted_label}',
                    unsafe_allow_html=True
                )

                train_cluster_labels_str = [agglo_cluster_to_label_map[cluster] for cluster in aggloModel.labels_]

                # Set up the figure
                fig, ax = plt.subplots()

                # Plot the PCA-transformed training data with cluster labels
                unique_labels = np.unique(train_cluster_labels_str)
                palette = sns.color_palette("husl", len(unique_labels))

                for i, label in enumerate(unique_labels):
                    indices = [j for j, l in enumerate(train_cluster_labels_str) if l == label]
                    ax.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], label=label, color=palette[i])

                # Plot the PCA-transformed new data point with a single legend entry
                ax.scatter(df_pca[:, 0], df_pca[:, 1], color='red', marker='x', s=100, label='New Data Points')

                ax.legend()
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                ax.set_title('PCA Projection of Training Data with New Data Point')

                st.pyplot(fig)

                time.sleep(14)

                style = f"""
                            <style>
                            .stApp {{
                                background: none;
                                background-size: cover;
                            }}
                            .css-15tx938, .stMarkdown, .css-10trblm, .st-bz, .css-1dx1gwv{{
                                color: black;
                            }}
                            body {{
                                background-color: #FFFFFF;
                            }}
                            </style>
                            """

                st.markdown(
                    style,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif option == 'CSV File Upload':
    st.subheader(">>> Upload your CSV file:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Data Preview:")
        st.write(df.head())

        # Check if the required columns are in the uploaded file
        required_columns = set(integer_cols + nominal_cols)
        if not required_columns.issubset(df.columns):
            st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
        else:
            df_copy = df.copy()

            # Apply label encoding to nominal columns
            for col in nominal_cols:
                if col in label_encoders:
                    handle_unseen_labels(df_copy, label_encoders[col], col)

            # Apply log transformation to integer columns
            for col in integer_cols:
                df_copy[col] = np.log(df_copy[col] + 1)

            df_copy = df_copy.apply(pd.to_numeric, errors='coerce')

            # Perform PCA transformation
            df_pca = pca.transform(df_copy)

            if st.button('Predict Clusters'):
                try:
                    # Calculate distances between the new data points and all training points
                    distances = euclidean_distances(df_pca, X_train_pca)

                    # Find the nearest training point for each new data point
                    nearest_train_idx = np.argmin(distances, axis=1)

                    # Assign the cluster label of the nearest training point to the new data point
                    agglo_test_predicted_labels = aggloModel.labels_[nearest_train_idx]

                    # Convert cluster labels to their corresponding string labels
                    aggloPredictedTestStr = get_labels(agglo_test_predicted_labels, agglo_cluster_to_label_map)

                    df_copy['Predicted Cluster Label'] = agglo_test_predicted_labels
                    df_copy['Predicted Cluster Label String'] = aggloPredictedTestStr

                    attack_count = (df_copy['Predicted Cluster Label String'].str.contains('neptune.', case=False)).sum()
                    total_rows = len(df_copy)

                    st.write("Predicted Clusters:")
                    st.write(df_copy)

                    st.write(f"Detected {attack_count} attacks out of {total_rows} rows ({(attack_count / total_rows) * 100:.2f}% of the data).")

                    if attack_count > 0:
                        autoplay_audio("conan.mp3")
                        time.sleep(1)
                        st.markdown("""
                            <style>
                            .stApp {
                                background-color: #FFB6C1;
                                animation: blink 2s linear infinite;
                            }
                            @keyframes blink {
                                50% { background-color: #FF0000; }
                            }
                            </style>
                            """, unsafe_allow_html=True)
                        st.markdown("""
                                    <div style="font-size:30px; color:red;">
                                        ⚠️ <strong>ALERT: Potential attacks detected in the dataset!</strong>
                                    </div>
                                    """, unsafe_allow_html=True)

                    train_cluster_labels_str = get_labels(aggloModel.labels_, agglo_cluster_to_label_map)

                    # Set up the figure
                    fig, ax = plt.subplots()

                    # Plot the PCA-transformed training data with cluster labels
                    unique_labels = np.unique(train_cluster_labels_str)
                    palette = sns.color_palette("husl", len(unique_labels))

                    for i, label in enumerate(unique_labels):
                        indices = [j for j, l in enumerate(train_cluster_labels_str) if l == label]
                        ax.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], label=label, color=palette[i])

                    # Plot the PCA-transformed test data with a single legend entry
                    ax.scatter(df_pca[:, 0], df_pca[:, 1], color='red', marker='x', s=100, label='New Data Points')

                    ax.legend()
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    ax.set_title('PCA Projection of Training Data with New Data Points')

                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
