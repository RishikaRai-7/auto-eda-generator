import streamlit as st
import pandas as pd
import requests
import io

# Setup page layout and style
st.set_page_config(
    page_title="Auto EDA & Report Generator",
    page_icon="📊",
    layout="wide"
)

# Custom header styling
st.markdown("""
<div style="background: linear-gradient(135deg, #4f46e5, #06b6d4); padding: 25px; border-radius: 10px; margin-bottom: 25px; color: white;">
    <h1 style="margin: 0; font-size: 2.5em;">📊 Python Auto EDA & Report Generator</h1>
    <p style="margin: 5px 0 0 0; font-size: 1.1em; opacity: 0.9;">
        Upload your CSV, let pandas do the statistical profiling, and watch Gemini 2.5 write a professional data analysis report.
    </p>
</div>
""", unsafe_allow_html=True)

# Configuration for FastAPI Backend
BACKEND_URL = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")
st.sidebar.markdown("""
### How to Run:
1. **Start FastAPI Backend**:
   `uvicorn main:app --reload --port 8000`
2. **Start Streamlit Frontend**:
   `streamlit run app.py`
""")

# File upload container
st.subheader("1. Upload your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read a preview of the dataset to show the user
    try:
        # Cache preview reading to prevent repeated parsing
        @st.cache_data
        def load_preview(file_bytes):
            return pd.read_csv(io.BytesIO(file_bytes), nrows=10)
        
        # Read the file bytes
        file_bytes = uploaded_file.getvalue()
        df_preview = load_preview(file_bytes)
        
        st.success("File uploaded successfully!")
        
        # Display small preview of data
        with st.expander("🔍 Preview Dataset (First 10 Rows)", expanded=True):
            st.dataframe(df_preview, use_container_width=True)
            st.info(f"Dataset Preview shown above. Size of uploaded file: {len(file_bytes) / 1024:.2f} KB")
            
        # Button to generate the report
        st.subheader("2. Analyze and Generate Report")
        if st.button("🚀 Generate AI EDA Report", type="primary"):
            with st.spinner("Processing data, running calculations and generating Gemini analysis report..."):
                try:
                    # Prepare file payload for FastAPI multipart/form-data
                    files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                    
                    # Request to FastAPI Backend
                    response = requests.post(f"{BACKEND_URL}/api/analyze", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.balloons()
                        st.success("Analysis report generated successfully!")
                        
                        # Extract the HTML report
                        html_report = result.get("html_report", "")
                        
                        # Add download button for standalone HTML report
                        st.download_button(
                            label="📥 Download Standalone HTML Report",
                            data=html_report,
                            file_name=f"eda_report_{uploaded_file.name.split('.')[0]}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                        
                        # Render report inside iframe
                        st.subheader("3. HTML Analysis Report")
                        st.components.v1.html(html_report, height=900, scrolling=True)
                        
                    else:
                        error_detail = response.json().get("detail", "Unknown server error.")
                        st.error(f"Backend analysis failed (Status {response.status_code}): {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not connect to FastAPI backend at {BACKEND_URL}. "
                        "Please verify your backend server is running and the URL is correct."
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    
    except Exception as e:
        st.error(f"Failed to read CSV preview: {str(e)}")
else:
    # Prompt user to upload a file
    st.info("Please upload a CSV file above to begin the automated EDA process.")
    
    # Showcase a mock sample section
    st.markdown("""
    ---
    ### Features included in the Report:
    * 📐 **Dataset Dimension Audit**: Full row and column count profiling.
    * 🗂️ **Data Structure Integrity**: Automatic data type mapping and missing value audit.
    * 🔢 **Descriptive Summary Stats**: Detailed statistical analysis of numeric ranges and variability.
    * 🔠 **Categorical Deep-Dive**: Unique cardinalities and distribution mapping for categories.
    * 🧠 **Gemini-Generated Executive Summary**: Plain English digest of your dataset's contents.
    * 💡 **Advanced Key Insights**: Multi-dimensional actionable insights from the metrics.
    * 🔬 **Recommended Statistical Tests**: Recommended hypothesis test validations (e.g. t-tests, ANOVA, Chi-Square).
    * 🛡️ **Data Quality Warning System**: Highlights missing ratios, extreme outliers, collinearity, or cardinality issues.
    """)
