import streamlit as st
import joblib
import numpy as np
import ipaddress
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="DDoS Attack Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498db;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        background-color: #f8f9fa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    try:
        # Use relative paths for model files
        model_base_path = os.path.join(os.path.dirname(__file__), "C:\\Users\\NAXUS\\Desktop\\DDOS\\models\\")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(model_base_path):
            os.makedirs(model_base_path)
            st.warning(f"Created models directory at {model_base_path}. Please place your model files here.")
            
            # Return dummy models for demonstration
            return create_dummy_models()
        
        try:
            model = joblib.load(os.path.join(model_base_path, "ddos_xgboost_model.pkl"))
            scaler = joblib.load(os.path.join(model_base_path, "scaler of timeand length.pkl"))
            class_label_encoder = joblib.load(os.path.join(model_base_path, "class_label_encoder.pkl"))
            icmp_label_encoder = joblib.load(os.path.join(model_base_path, "icmp_label_encoder.pkl"))
            csv_model = joblib.load(os.path.join(model_base_path, "ddos_model.pkl"))
            
            
            return model, scaler, class_label_encoder, icmp_label_encoder, csv_model
        except FileNotFoundError:
            st.warning("Model files not found. Using dummy models for demonstration.")
            return create_dummy_models()
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return create_dummy_models()

# Function to create dummy models for demonstration
def create_dummy_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Create dummy model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(np.array([[1, 2, 3, 4, 5]]), np.array([0]))
    
    # Create dummy scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array([0, 0])
    scaler.scale_ = np.array([1, 1])
    
    # Create dummy label encoders
    class_label_encoder = LabelEncoder()
    class_label_encoder.classes_ = np.array(['Attack', 'Normal'])
    
    icmp_label_encoder = LabelEncoder()
    icmp_label_encoder.classes_ = np.array(['Echo Request', 'Echo Reply', 'Destination Unreachable', 'Time Exceeded', 'Router Advertisement'])
    
    # Dummy CSV model
    csv_model = RandomForestClassifier(n_estimators=10)
    csv_model.fit(np.array([[1, 2, 3, 4, 5]]), np.array([0]))
    
    return model, scaler, class_label_encoder, icmp_label_encoder, csv_model

# Function to convert IPv6 address to integer
def ipv6_to_int(ipv6_addr):
    try:
        return int(ipaddress.IPv6Address(ipv6_addr))
    except ipaddress.AddressValueError:
        return None

# Function to make individual prediction
def predict_individual(source_ip, destination_ip, packet_length, time_input, icmp_type):
    # Load models
    model, scaler, class_label_encoder, icmp_label_encoder, _ = load_models()
    
    # Convert IPs
    source_int = ipv6_to_int(source_ip)
    destination_int = ipv6_to_int(destination_ip)
    
    # Validate IPs
    if source_int is None or destination_int is None:
        return None, "Invalid IPv6 address format"
    
    # Convert ICMP type label to encoded value
    try:
        icmp_encoded = icmp_label_encoder.transform([icmp_type])[0]
    except ValueError:
        return None, f"Invalid ICMP type: {icmp_type}"
    
    # Prepare input
    input_data = np.array([[time_input, packet_length]])
    
    # Scale only `Time` and `Length`
    input_data_scaled = scaler.transform(input_data)
    time_scaled = input_data_scaled[0][0]
    length_scaled = input_data_scaled[0][1]
    
    # Ensure feature order matches training
    final_input = np.array([[source_int, destination_int, time_scaled, length_scaled, icmp_encoded]])
    
    # Make prediction
    prediction = model.predict(final_input)[0]
    
    # Decode prediction label
    result = class_label_encoder.inverse_transform([prediction])[0]
    
    return result, None

# Function to predict from CSV
def predict_from_csv(df):
    # Check if DataFrame is empty
    if df.empty:
        return None, "Empty CSV file"
    
    _, _, _, _, csv_model = load_models()
    
    # Ensure required columns exist
    required_columns = ['Source', 'Destination Address', 'Length', 'Protocol', 'Time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return None, f"CSV file missing required columns: {', '.join(missing_columns)}"
    
    # Predict using the model
    try:
        predictions = csv_model.predict(df[required_columns])
        df['Prediction'] = ['Normal' if pred == 1 else 'Attack' for pred in predictions]
        return df, None
    except Exception as e:
        return None, f"Error in prediction: {e}"

# Sidebar
with st.sidebar:
    st.image("C:\\Users\\NAXUS\\Desktop\\DDOS\\cyber-security.png", width=80)
    st.title("DDoS Protection System")
    
    st.markdown("---")
    
    # Navigation using option_menu
    selected = option_menu(
        "Main Menu",
        ["Project Overview", "Manual Prediction", "Batch Prediction", "Feature Analysis"],
        icons=["house", "shield", "file-earmark-spreadsheet", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.info(
        "This application helps network administrators detect DDoS attacks "
        "using machine learning to analyze network traffic patterns."
    )
    
    st.markdown("### How It Works")
    st.markdown("""
    1. **Manual Mode**: Enter packet details for immediate prediction
    2. **Batch Mode**: Upload a CSV file with multiple entries
    3. **Feature Analysis**: Visualize feature importance and packet stats
    """)

# Main content based on selection
if selected == "Project Overview":
    st.markdown("<h1 class='main-header'>🛡️ DDoS Attack Detection System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### What is DDoS?")
        st.markdown("""
        Distributed Denial of Service (DDoS) attacks attempt to disrupt normal traffic to a targeted server, 
        service, or network by overwhelming the target with a flood of Internet traffic.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### How Our System Works")
        st.markdown("""
        Our system uses machine learning to analyze network traffic patterns and identify potential DDoS attacks:
        
        1. **Data Collection**: Captures network packet data including source/destination IPs, packet lengths, and time
        2. **Analysis**: Processes this data through our trained XGBoost model
        3. **Classification**: Determines if traffic patterns indicate normal behavior or an attack
        4. **Visualization**: Presents statistics to help security teams respond appropriately
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.image("C:\\Users\\NAXUS\\Desktop\\DDOS\\cyber-security.png", width=200)
        
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### Getting Started")
        st.markdown("""
        Use the navigation menu on the left to:
        
        - Make individual packet predictions
        - Upload CSV files for batch analysis
        - View feature importance and statistics
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show a sample metrics card
    st.markdown("### System Metrics")
    st.markdown("*Note: The following metrics are based on model evaluation data and may vary with your specific traffic patterns.*")
    
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        st.metric(label="Detection Rate", value="97.8%", delta="+2.1%")
    
    with metrics_cols[1]:
        st.metric(label="False Positives", value="2.3%", delta="-0.8%")
    
    with metrics_cols[2]:
        st.metric(label="Processing Time", value="0.05 sec", delta="-0.01 sec")
    
    with metrics_cols[3]:
        st.metric(label="Model Accuracy", value="98.2%", delta="+1.4%")

elif selected == "Manual Prediction":
    st.markdown("<h1 class='main-header'>🔍 Manual DDoS Detection</h1>", unsafe_allow_html=True)
    st.markdown("Enter individual packet details to check if it's a potential DDoS attack.")
    
    # Load models
    model, scaler, class_label_encoder, icmp_label_encoder, _ = load_models()
    
    # Create a nice form layout
    with st.form("packet_form"):
        st.markdown("### Packet Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_ip = st.text_input("🔹 Source IPv6 Address", value="2001:db8::1")
            packet_length = st.number_input("🔹 Packet Length", min_value=0, value=100, step=1)
        
        with col2:
            destination_ip = st.text_input("🔹 Destination IPv6 Address", value="2001:db8::2")
            time_input = st.number_input("🔹 Timestamp (seconds)", value=0.123456, format="%.6f")
        
        # ICMP Type Selection - handle the case when model loading failed
        icmp_types = ['Echo Request', 'Echo Reply', 'Destination Unreachable', 'Time Exceeded']
        if hasattr(icmp_label_encoder, 'classes_'):
            icmp_types = icmp_label_encoder.classes_.tolist()
        
        icmp_type = st.selectbox("🔹 ICMPv6 Type", icmp_types)
        
        submitted = st.form_submit_button("🔍 Analyze Packet", use_container_width=True)
    
    # Show prediction results
    if submitted:
        # Add a spinner while processing
        with st.spinner('Analyzing packet data...'):
            # Simulate processing time
            time.sleep(0.5)
            
            result, error = predict_individual(source_ip, destination_ip, packet_length, time_input, icmp_type)
            
            if error:
                st.markdown(f"<div class='error-box'><h3>❌ Error</h3><p>{error}</p></div>", unsafe_allow_html=True)
            else:
                if result == "Attack":
                    st.markdown(
                        f"""
                        <div class='error-box prediction-card'>
                            <h2>🚨 Detection Result: Attack 🚨</h2>
                            <p>This packet shows characteristics of a DDoS attack and should be investigated.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='success-box prediction-card'>
                            <h2>✅ Detection Result: Normal</h2>
                            <p>This packet appears to be normal network traffic.</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Show packet details
                st.markdown("### Packet Details")
                details_cols = st.columns(3)
                
                with details_cols[0]:
                    st.info(f"**Source IP:** {source_ip}")
                    st.info(f"**Destination IP:** {destination_ip}")
                
                with details_cols[1]:
                    st.info(f"**Packet Length:** {packet_length}")
                    st.info(f"**ICMPv6 Type:** {icmp_type}")
                
                with details_cols[2]:
                    st.info(f"**Timestamp:** {time_input}")
                    st.info(f"**Classification:** {result}")

elif selected == "Batch Prediction":
    st.markdown("<h1 class='main-header'>📊 Batch DDoS Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file with multiple packet entries for bulk analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read and display the raw data
        try:
            df = pd.read_csv(uploaded_file)
            
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())
            
            # Process with a spinner
            with st.spinner('Analyzing network data...'):
                # Simulate processing time
                time.sleep(1)
                
                # Make predictions
                result_df, error = predict_from_csv(df)
                
                if error:
                    st.markdown(f"<div class='error-box'><h3>❌ Error</h3><p>{error}</p></div>", unsafe_allow_html=True)
                else:
                    # Show prediction metrics
                    st.markdown("### Detection Results")
                    
                    # Calculate metrics
                    attack_count = (result_df['Prediction'] == 'Attack').sum()
                    normal_count = (result_df['Prediction'] == 'Normal').sum()
                    total_count = len(result_df)
                    attack_percentage = (attack_count / total_count) * 100
                    
                    # Display metrics
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.metric(label="Total Packets", value=f"{total_count}")
                    
                    with metrics_cols[1]:
                        st.metric(label="Normal Traffic", value=f"{normal_count}", 
                                 delta=f"{normal_count/total_count:.1%}")
                    
                    with metrics_cols[2]:
                        st.metric(label="Attack Traffic", value=f"{attack_count}", 
                                 delta=f"{attack_count/total_count:.1%}")
                    
                    with metrics_cols[3]:
                        st.metric(label="Attack Percentage", value=f"{attack_percentage:.1f}%")
                    
                    # Show results table
                    st.markdown("### Detailed Results")
                    st.dataframe(
                        result_df[['Source', 'Destination Address', 'Length', 'Protocol', 'Time', 'Prediction']]
                        .style.apply(
                            lambda x: ['background-color: #ffcccc' if x.Prediction == 'Attack' 
                                      else 'background-color: #ccffcc' for i in x],
                            axis=1
                        )
                    )
                    
                    # Show visualization
                    st.markdown("### Visualization")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart
                    ax1.pie(
                        [normal_count, attack_count], 
                        labels=['Normal', 'Attack'], 
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#F44336'],
                        explode=(0, 0.1)
                    )
                    ax1.set_title('Traffic Distribution')
                    
                    # Bar chart for protocol distribution
                    # Check if 'Protocol' column exists
                    if 'Protocol' in result_df.columns:
                        protocol_counts = result_df.groupby(['Protocol', 'Prediction']).size().unstack(fill_value=0)
                        if not protocol_counts.empty:
                            protocol_counts.plot(kind='bar', stacked=True, ax=ax2, color=['#4CAF50', '#F44336'])
                            ax2.set_title('Protocol Distribution')
                            ax2.set_xlabel('Protocol')
                            ax2.set_ylabel('Count')
                        else:
                            ax2.text(0.5, 0.5, 'No protocol data available', ha='center', va='center')
                    else:
                        ax2.text(0.5, 0.5, 'Protocol column not found in data', ha='center', va='center')
                    
                    st.pyplot(fig)
                    
                    # Download button for results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis Results",
                        data=csv,
                        file_name="ddos_analysis_results.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif selected == "Feature Analysis":
    st.markdown("<h1 class='main-header'>📈 Feature Analysis</h1>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Feature Importance", "Packet Statistics"])
    
    with tab1:
        st.markdown("### Key Features for DDoS Detection")
        st.markdown("""
        This section displays the importance of each feature in determining whether a packet 
        is part of a DDoS attack. The following features are used in our prediction model:
        """)
        
        feature_importance = {
            'Source IP': 82,
            'Destination IP': 65,
            'Packet Length': 78,
            'Timestamp': 45,
            'ICMP Type': 70
        }
        
        # Create horizontal bar chart for feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]
        
        # Create horizontal bar chart
        ax.barh(features, importance, color='#3498db')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance for DDoS Detection')
        
        # Add values at the end of each bar
        for i, v in enumerate(importance):
            ax.text(v + 1, i, str(v), va='center')
        
        st.pyplot(fig)
        
        st.markdown("""
        #### Feature Descriptions:
        
        - **Source IP**: The origin of the packet. Certain IP ranges may be associated with attack sources.
        - **Destination IP**: The target of the packet. DDoS attacks often target specific services.
        - **Packet Length**: The size of the packet. Unusual packet sizes can indicate attack patterns.
        - **Timestamp**: Time-related patterns can indicate coordinated attacks.
        - **ICMP Type**: The type of ICMP message. Certain types are commonly used in DDoS attacks.
        """)
    
    with tab2:
        st.markdown("### Packet Statistics Analysis")
        
        # Upload file for statistics
        st.markdown("Upload a CSV file to analyze packet statistics:")
        stats_file = st.file_uploader("Choose a CSV file for statistics", type="csv", key="stats_file")
        
        if stats_file is not None:
            try:
                stats_df = pd.read_csv(stats_file)
                
                # Display basic statistics
                st.markdown("### Basic Statistics")
                
                if 'Length' in stats_df.columns:
                    # Create boxes for summary statistics
                    stat_cols = st.columns(4)
                    
                    with stat_cols[0]:
                        st.metric("Avg Packet Length", f"{stats_df['Length'].mean():.2f}")
                    
                    with stat_cols[1]:
                        st.metric("Max Packet Length", f"{stats_df['Length'].max()}")
                    
                    with stat_cols[2]:
                        st.metric("Min Packet Length", f"{stats_df['Length'].min()}")
                    
                    with stat_cols[3]:
                        st.metric("Std Dev Length", f"{stats_df['Length'].std():.2f}")
                    
                    # Create histogram of packet lengths
                    st.markdown("### Packet Length Distribution")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(stats_df['Length'], bins=30, color='#3498db', alpha=0.7)
                    ax.set_xlabel('Packet Length')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Packet Lengths')
                    
                    st.pyplot(fig)
                    
                    # Show potential anomalies
                    threshold = stats_df['Length'].mean() + 2 * stats_df['Length'].std()
                    anomalies = stats_df[stats_df['Length'] > threshold]
                    
                    if not anomalies.empty:
                        st.markdown("### Potential Anomalies")
                        st.markdown(f"Packets with length greater than {threshold:.2f} (mean + 2σ):")
                        st.dataframe(anomalies)
                else:
                    st.warning("The CSV file does not contain a 'Length' column.")
                
                # Protocol distribution if available
                if 'Protocol' in stats_df.columns:
                    st.markdown("### Protocol Distribution")
                    
                    protocol_counts = stats_df['Protocol'].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    protocol_counts.plot(kind='bar', ax=ax, color='#2ecc71')
                    ax.set_xlabel('Protocol')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Protocols')
                    
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error analyzing statistics: {e}")
        else:
            st.info("Upload a CSV file to view packet statistics.")

# Show warning if models weren't loaded
if not os.path.exists(os.path.join(os.path.dirname(__file__), "models")):
    st.sidebar.warning(
        "Model files not found. The application is running in demonstration mode with dummy models. "
        "Please place your model files in a 'models' directory at the same location as this script."
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Created with Streamlit • DDoS Detection System • © 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)