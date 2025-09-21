import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import google.generativeai as genai

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="NILM Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
CITY_CONFIG = {
    "Laayoune": {"model_name": "kmeans_laayoune.joblib"},
    "Marrakech": {"model_name": "kmeans_marrakech.joblib"},
    "Boujdour": {"model_name": "kmeans_boujdour.joblib"},
    "Foum Eloued": {"model_name": "kmeans_foum_eloued.joblib"}
}

# --- AI SUMMARY FUNCTION ---
def get_ai_summary(city_name, cluster_summary_df, hourly_dist_df):
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = f"""
        You are an expert energy analyst. Analyze the following NILM data for {city_name} and provide a concise summary with actionable insights.

        **Cluster Statistics:**
        This table shows the count and power draw (in Amperes) for each event type.
        {cluster_summary_df.to_markdown()}

        **Hourly Distribution of Events:**
        This table shows how many times each event type occurred during each hour of the day.
        {hourly_dist_df.to_markdown()}

        **Your Task:**
        1. **Interpret the Clusters:** Based on the 'mean' power_delta, suggest what kind of appliances 'Small', 'Medium', and 'Large' Events might be.
        2. **Identify Peak Hours:** Determine the peak energy consumption hours.
        3. **Provide Actionable Energy-Saving Tips:** Give 2-3 specific recommendations to save energy based on the data.

        Structure your response in clear, easy-to-understand markdown.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to the AI service: {e}. Please ensure your API key is correct in the `.streamlit/secrets.toml` file."

# --- 1. PREPROCESSING & VALIDATION FUNCTION ---
@st.cache_data
def preprocess_and_validate(uploaded_file, city_name):
    """
    This function has SEPARATE and CORRECT logic for CSV and XLSX files.
    """
    df = None
    
    try:
        # --- LOGIC FOR CSV FILES ---
        if uploaded_file.name.endswith('.csv'):
            if city_name.lower() not in uploaded_file.name.lower():
                st.error(f"File Name Mismatch: You selected '{city_name}', but you uploaded a CSV named '{uploaded_file.name}'. Please upload the correct file.")
                return None
            df = pd.read_csv(uploaded_file, parse_dates=['DateTime'])

        # --- LOGIC FOR XLSX FILES ---
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(uploaded_file, sheet_name=city_name)
            except ValueError:
                st.error(f"Sheet Not Found: You selected '{city_name}', but an Excel sheet with that exact name was not found in '{uploaded_file.name}'.")
                return None
        
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

        # --- COMMON PROCESSING STEPS ---
        if 'DateTime' not in df.columns:
            st.error("Error: The uploaded file must contain a 'DateTime' column.")
            return None
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.interpolate(method='linear', inplace=True)
        
        zone_columns = [col for col in df.columns if 'zone' in col.lower()]
        if not zone_columns:
            st.error("Data Error: No 'zone' columns found in the file.")
            return None
        
        st.success(f"File '{uploaded_file.name}' validated and processed successfully for '{city_name}'.")
        df['total_consumption'] = df[zone_columns].sum(axis=1)
        df['power_delta'] = df['total_consumption'].diff().fillna(0)
        df['hour'] = df.index.hour
        
        return df
        
    except Exception as e:
        st.error(f"A critical error occurred during file processing: {e}")
        return None

# --- UI Layout ---
st.title("💡 Non-Intrusive Load Monitoring (NILM) Analysis Tool")
st.write("Upload smart meter data to disaggregate energy consumption and identify appliance usage patterns.")

with st.sidebar:
    st.header("⚙️ Controls")
    city_list = list(CITY_CONFIG.keys())
    selected_city = st.selectbox("Step 1: Select a City", city_list)
    uploaded_file = st.file_uploader(f"Step 2: Upload Data for {selected_city}", type=['csv', 'xlsx'])

# --- Main App Logic ---
if uploaded_file is not None:
    processed_df = preprocess_and_validate(uploaded_file, selected_city)
    
    if processed_df is not None:
        # Load the model
        model_filename = CITY_CONFIG[selected_city]["model_name"]
        if not os.path.exists(model_filename):
            st.error(f"Fatal Error: Model file '{model_filename}' not found.")
            st.stop()
        model = joblib.load(model_filename)

        # Run model inference
        on_events = processed_df[processed_df['power_delta'] > 1].copy()
        
        
        # If on_events is empty, show a warning. Otherwise, run the entire analysis.
        if on_events.empty:
            st.warning("No significant 'ON' events were found in the data. Cannot display analysis.")
        else:
            labels = model.predict(on_events[['power_delta']])
            on_events['cluster'] = labels

            # --- VISUALIZATIONS ---
            st.markdown("---")
            st.header("📊 Section 1: Energy Disaggregation")
            st.subheader("Select a Date Range to Analyze")
            st.info("The full dataset is too dense to view at once. Please select a shorter period (e.g., one week) to see the details.")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", processed_df.index.min().date())
            with col2:
                end_date = st.date_input("End Date", processed_df.index.max().date())
            
            if start_date > end_date:
                st.error("Error: Start date must be before end date.")
            # This 'else' prevents crashing if the date is wrong
            else: 
                filtered_df = processed_df.loc[start_date:end_date]
                filtered_events = on_events.loc[start_date:end_date]
                
                # Check for data in the selected range
                if filtered_df.empty or filtered_events.empty:
                    st.warning("No data available for the selected date range.")
                else:
                    cluster_means = on_events.groupby('cluster')['power_delta'].mean().sort_values()
                    cluster_labels = ['Small Events', 'Medium Events', 'Large Events', 'Extra Large Events']
                    label_mapping = {cluster_id: label for cluster_id, label in zip(cluster_means.index, cluster_labels)}
                    on_events['cluster_label'] = on_events['cluster'].map(label_mapping)
                    filtered_events['cluster_label'] = filtered_events['cluster'].map(label_mapping)
                    palette = sns.color_palette("Set2", n_colors=len(label_mapping))
                    color_map = {label: palette[i] for i, label in enumerate(label_mapping.values())}
                    
                    fig1, ax1 = plt.subplots(figsize=(15, 7))
                    ax1.plot(filtered_df.index, filtered_df['total_consumption'], label='Total Consumption', color='royalblue', alpha=0.6)
                    ax1.fill_between(filtered_df.index, filtered_df['total_consumption'], color='royalblue', alpha=0.1)
                    
                    hue_order = [label for label in cluster_labels if label in filtered_events['cluster_label'].unique()]
                    
                    sns.scatterplot(
                        data=filtered_events,
                        x=filtered_events.index,
                        y=filtered_df.loc[filtered_events.index]['total_consumption'],
                        hue='cluster_label',
                        hue_order=hue_order,
                        palette=color_map,
                        ax=ax1,
                        s=80,
                        edgecolor='black',
                        linewidth=0.8,
                        zorder=5
                    )

                    ax1.set_title(f'Consumption and Events for {selected_city} ({start_date} to {end_date})', fontsize=16)
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel('Total Current (Amperes)')
                    ax1.legend(title='Appliance Event Type')
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig1, use_container_width=True)
                    plt.close(fig1)

                    st.markdown("---")
                    st.header("📈 Section 2: Cluster Insights")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Cluster Statistics")
                        cluster_summary = on_events.groupby('cluster_label')['power_delta'].agg(['count', 'mean', 'min', 'max']).sort_values('mean')
                        st.dataframe(cluster_summary, use_container_width=True)
                    with col2:
                        st.subheader("Hourly Distribution of Events")
                        fig2, ax2 = plt.subplots()
                        hourly_distribution = on_events.groupby(['hour', 'cluster_label']).size().unstack(fill_value=0)
                        sns.countplot(data=on_events.sort_values('cluster_label'), x='hour', hue='cluster_label', ax=ax2, palette=color_map)
                        ax2.get_legend().set_title("Event Type")
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    st.subheader("Contribution of Event Types")
                    event_counts = on_events['cluster_label'].value_counts()
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.pie(event_counts, labels=event_counts.index, autopct='%1.1f%%', startangle=90, colors=[color_map[label] for label in event_counts.index], wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                    ax3.set_title('Proportion of "ON" Events by Cluster', fontsize=16)
                    ax3.axis('equal')
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)
                    
                    st.markdown("---")
                    st.header("🤖 Section 3: AI-Powered Summary & Recommendations")
                    
                    if 'ai_summary' not in st.session_state:
                        st.session_state.ai_summary = None
                    
                    # Create button columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("Generate AI Summary", type="primary"):
                            with st.spinner("🧠 The AI is analyzing your data... Please wait."):
                                st.session_state.ai_summary = get_ai_summary(selected_city, cluster_summary, hourly_distribution)
                    
                    with col2:
                        if st.session_state.ai_summary and st.button("Clear Summary"):
                            st.session_state.ai_summary = None
                    
                    # Display AI summary if it exists
                    if st.session_state.ai_summary:
                        st.markdown("### Analysis Results")
                        st.markdown(st.session_state.ai_summary)

        

else:
    st.info("Please select a city and upload the corresponding data file to begin the analysis.")