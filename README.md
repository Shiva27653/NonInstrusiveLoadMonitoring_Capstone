# Non-Intrusive Load Monitoring (NILM) for Energy Disaggregation


This project presents a live, web-based application for Non-Intrusive Load Monitoring (NILM). The tool takes raw, aggregated smart meter data, processes it on the fly, and uses an unsupervised machine learning model to disaggregate the total energy signal into its constituent load patterns. This allows for detailed energy audits and the identification of consumption behaviors without needing sensors on every device.

-----

### 🚀 Live Application

**Access the deployed application here:** [**NILM Capstone Project**](https://noninstrusiveloadmonitoringcapstone-n75ksjd9ayghchpgdpi44s.streamlit.app/)

-----

### \#\# Key Features

  * **Live Data Analysis**: Users can select a city and upload raw smart meter data to receive a real-time analysis.
  * **Dynamic Preprocessing**: The app contains a robust pipeline that automatically cleans, processes, and engineers features from the uploaded data.
  * **Unsupervised Disaggregation**: Uses a pre-trained K-Means clustering model, specific to each city, to identify and classify appliance 'ON' events into **Small, Medium, and Large** load types.
  * **Interactive Visualization**: A dynamic chart allows users to see the total energy consumption overlaid with the disaggregated event clusters, which can be toggled on and off.
  * **AI-Powered Insights**: Integrates the **Google Gemini API** to automatically generate a summary of the city's unique energy patterns based on the model's findings.

-----

### \#\# Key Findings

The primary insight from the analysis was the successful disaggregation of the city-wide power signal into distinct load types. The K-Means model consistently identified a **"Large Events"** cluster that correlated strongly with typical business hours (approx. 6 AM - 6 PM), indicating it represents the city's **industrial and heavy commercial load**. The "Small" and "Medium" events were more broadly distributed, representing the residential and light commercial base load.

-----

### \#\# 🛠️ Technology Stack

  * **Language**: Python
  * **Data Science**: Pandas, Scikit-learn, Joblib
  * **Web Framework**: Streamlit
  * **Visualization**: Matplotlib, Seaborn
  * **LLM Integration**: Google Gemini API
  * **Deployment**: Streamlit Community Cloud & GitHub

-----

### \#\# Methodology & Model Selection

1.  **Data Preprocessing & Feature Engineering**: Raw data from four Moroccan cities was loaded. A robust pipeline was created to clean the data, handle missing values and duplicates, and consolidate multiple zones into a single `total_consumption` signal. The key feature, `power_delta`, was engineered to isolate energy "events."

2.  **Model Training & Evaluation**: An unsupervised K-Means clustering model was trained for each city. The model was chosen after a **comparative analysis against four other algorithms** (DBSCAN, GMM, Time Series K-Means with DTW, and K-Shape), as it yielded the highest Silhouette Score (**0.586** for the baseline city) and the most interpretable results.

3.  **Application Development**: A live analysis tool was built using Streamlit, allowing users to upload data and visualize the disaggregation results in real-time.

4.  **LLM Integration**: The Gemini API was integrated to provide automated, intelligent summaries based on the quantitative output of the K-Means model.

-----

### \#\# 📂 Running the Project Locally

To run this application on your local machine, please follow these steps:

**1. Prerequisites**

  * Python 3.9+
  * Anaconda or another Python environment manager.

**2. Clone the Repository**

# 1. Clone your repository from GitHub
git clone https://github.com/Shiva27653/NonInstrusiveLoadMonitoring_Capstone.git

# 2. Navigate into the newly created project folder
cd NonInstrusiveLoadMonitoring_Capstone

**3. Install Dependencies**
Create a `requirements.txt` file with the following content and then run `pip install -r requirements.txt`:

```
streamlit
pandas
scikit-learn==1.3.2
joblib
matplotlib
seaborn
google-generativeai
openpyxl
tabulate
```

**4. Set Up Streamlit Secrets**

  * Create a folder named `.streamlit` in the main project directory.
  * Inside this folder, create a file named `secrets.toml`.
  * Add your Google Gemini API key to this file:
    ```toml
    GOOGLE_API_KEY = "PASTE_YOUR_API_KEY_HERE"
    ```

**5. Run the App**
Open your terminal or Anaconda Prompt, navigate to the project folder, and run:

```bash
streamlit run app.py
```
