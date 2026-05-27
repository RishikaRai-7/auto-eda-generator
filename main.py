import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from eda import run_eda
from assistant import GeminiAssistant
from typing import Dict, Any

app = FastAPI(
    title="Auto EDA & Report Generator API",
    description="Backend API to run pandas EDA and Gemini-powered data analysis, producing a cohesive HTML report.",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def render_html_report(eda: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """
    Renders the data profile and Gemini insights into a clean, modern, and minimal HTML document.
    """
    # 1. Dataset overview metrics
    summary = eda.get("summary", {})
    rows = summary.get("rows", 0)
    cols = summary.get("columns", 0)
    num_cols = summary.get("numeric_columns_count", 0)
    cat_cols = summary.get("categorical_columns_count", 0)
    
    # 2. Build Data Types table
    dtypes_html = ""
    for col, dtype in eda.get("dtypes", {}).items():
        null_count = eda.get("null_info", {}).get(col, {}).get("count", 0)
        null_pct = eda.get("null_info", {}).get(col, {}).get("percentage", 0.0)
        dtypes_html += f"""
        <tr>
            <td><strong>{col}</strong></td>
            <td><code>{dtype}</code></td>
            <td>{null_count} ({null_pct}%)</td>
        </tr>
        """
        
    # 3. Build Basic Statistics for Numeric Columns
    stats_html = ""
    num_stats = eda.get("numeric_stats", {})
    if num_stats:
        stats_html += """
        <h3>Descriptive Statistics (Numeric Columns)</h3>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>50% (Median)</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
        """
        for col, stats in num_stats.items():
            stats_html += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{stats.get('count')}</td>
                <td>{stats.get('mean'):.4g}</td>
                <td>{stats.get('std'):.4g}</td>
                <td>{stats.get('min'):.4g}</td>
                <td>{stats.get('50%'):.4g}</td>
                <td>{stats.get('max'):.4g}</td>
            </tr>
            """
        stats_html += "</tbody></table>"
    else:
        stats_html += "<p><em>No numeric columns detected in the dataset.</em></p>"
        
    # 4. Build Categorical columns unique value tables
    cat_html = ""
    cat_stats = eda.get("categorical_stats", {})
    if cat_stats:
        cat_html += "<h3>Categorical Value Distribution (Top Values)</h3>"
        for col, stats in cat_stats.items():
            cat_html += f"""
            <div class="categorical-card">
                <h4>Column: {col} <span class="badge">Unique Values: {stats.get('unique_count')}</span></h4>
                <table>
                    <thead>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for val_info in stats.get("top_values", []):
                cat_html += f"""
                <tr>
                    <td>{val_info.get('value')}</td>
                    <td>{val_info.get('count')}</td>
                    <td>{val_info.get('percentage')}%</td>
                </tr>
                """
            cat_html += "</tbody></table></div>"
    
    # 5. Recommended Statistical Tests (Gemini)
    tests_html = ""
    rec_tests = analysis.get("recommended_statistical_tests", [])
    if rec_tests:
        for test in rec_tests:
            cols_involved = ", ".join([f"<code>{c}</code>" for c in test.get('columns_involved', [])])
            cols_label = f" ({cols_involved})" if cols_involved else ""
            tests_html += f"""
            <div class="insight-item">
                <h5>{test.get('test_name')}{cols_label}</h5>
                <p>{test.get('rationale')}</p>
            </div>
            """
    else:
        tests_html = "<p><em>No specific statistical tests recommended.</em></p>"
        
    # 6. Data Quality Issues (Gemini)
    issues_html = ""
    quality_issues = analysis.get("data_quality_issues", [])
    if quality_issues:
        for issue in quality_issues:
            cols_involved = ", ".join([f"<code>{c}</code>" for c in issue.get('columns_involved', [])])
            cols_label = f" ({cols_involved})" if cols_involved else ""
            issues_html += f"""
            <div class="issue-item">
                <h5><span class="warning-badge">{issue.get('issue_type')}</span>{cols_label}</h5>
                <p>{issue.get('description')}</p>
            </div>
            """
    else:
        issues_html = "<p>No critical data quality issues identified. The data appears well-formed!</p>"
        
    # 7. Key Insights (Gemini)
    insights_html = ""
    insights = analysis.get("key_insights", [])
    if insights:
        for insight in insights:
            insights_html += f"<li>{insight}</li>"
    else:
        insights_html = "<li>No specific insights generated.</li>"

    # HTML Boilerplate and Styles
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exploratory Data Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #4f46e5, #06b6d4);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.2em;
        }}
        .header p {{
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4f46e5;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            margin-top: 0;
            border-bottom: 2px solid #f3f4f6;
            padding-bottom: 10px;
            color: #1f2937;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background-color: #f9fafb;
            font-weight: 600;
            color: #4b5563;
        }}
        tr:hover {{
            background-color: #fcfcfd;
        }}
        .categorical-card {{
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .categorical-card h4 {{
            margin: 0 0 10px 0;
            color: #374151;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .badge {{
            background-color: #e0e7ff;
            color: #4338ca;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: normal;
        }}
        .warning-badge {{
            background-color: #fef3c7;
            color: #d97706;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 8px;
        }}
        .insight-item, .issue-item {{
            border-left: 4px solid #4f46e5;
            background-color: #f9fafb;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 15px;
        }}
        .issue-item {{
            border-left-color: #d97706;
        }}
        .insight-item h5, .issue-item h5 {{
            margin: 0 0 5px 0;
            font-size: 1.1em;
            color: #1f2937;
        }}
        .insight-item p, .issue-item p {{
            margin: 0;
            color: #4b5563;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 10px;
        }}
        code {{
            background-color: #f3f4f6;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Dataset EDA & Statistical Report</h1>
        <p>Generated by Python Auto EDA Backend & Gemini 2.5 Analyst</p>
    </div>

    <!-- Summary Metrics Cards -->
    <div class="grid">
        <div class="metric-card">
            <div class="metric-value">{rows}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{cols}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{num_cols}</div>
            <div class="metric-label">Numeric Columns</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{cat_cols}</div>
            <div class="metric-label">Categorical Columns</div>
        </div>
    </div>

    <!-- Gemini Dataset Analysis -->
    <div class="section">
        <h2>AI Dataset Summary & Key Insights</h2>
        <p><strong>Executive Summary:</strong> {analysis.get('summary', 'No summary available.')}</p>
        
        <h3>Key Insights</h3>
        <ul>
            {insights_html}
        </ul>
    </div>

    <!-- Data Types & Missing Values -->
    <div class="section">
        <h2>Data Structure & Completeness</h2>
        <table>
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Data Type</th>
                    <th>Missing Values (%)</th>
                </tr>
            </thead>
            <tbody>
                {dtypes_html}
            </tbody>
        </table>
    </div>

    <!-- Descriptive Statistics -->
    <div class="section">
        <h2>Descriptive Statistics</h2>
        {stats_html}
    </div>

    <!-- Categorical Value Distribution -->
    <div class="section">
        {cat_html}
    </div>

    <!-- Gemini Recommended Statistical Tests -->
    <div class="section">
        <h2>Recommended Statistical Hypothesis Tests</h2>
        <p>Based on your column data types and distribution patterns, these tests are mathematically recommended for deeper research:</p>
        {tests_html}
    </div>

    <!-- Gemini Data Quality Audit -->
    <div class="section">
        <h2>Data Quality & Preprocessing Audit</h2>
        <p>Potential anomalies, cleaning instructions, and modeling issues found in the data structure:</p>
        {issues_html}
    </div>

</body>
</html>
"""
    return html

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, runs statistical calculations, sends the metrics 
    to Gemini 2.5, and generates a pre-styled static HTML report.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a valid CSV file."
        )
        
    try:
        # Read the uploaded CSV content into a BytesIO stream
        contents = await file.read()
        file_stream = io.BytesIO(contents)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to read file contents: {str(e)}"
        )
        
    # Execute Pandas profiling
    eda_data = run_eda(file_stream)
    if "error" in eda_data:
        raise HTTPException(status_code=400, detail=eda_data["error"])
        
    # Run Gemini LLM analysis
    try:
        assistant = GeminiAssistant()
        analysis = assistant.analyze_eda(eda_data)
    except Exception as e:
        # Handle cases where GEMINI_API_KEY is not set or API error occurred
        analysis = {
            "error": f"Failed to initialize GeminiAssistant: {str(e)}",
            "summary": "AI summary is unavailable as the GEMINI_API_KEY is not set or the API was unreachable.",
            "key_insights": [
                "Could not run AI-powered insight generation.",
                "Please configure the GEMINI_API_KEY environment variable on your backend server."
            ],
            "recommended_statistical_tests": [
                {
                    "test_name": "API Key Missing",
                    "columns_involved": [],
                    "rationale": "Setup the GEMINI_API_KEY environment variable to retrieve statistics-driven recommended tests."
                }
            ],
            "data_quality_issues": [
                {
                    "issue_type": "Configuration Warning",
                    "columns_involved": [],
                    "description": "The backend server is running without a GEMINI_API_KEY. AI-based data profiling is disabled."
                }
            ]
        }
        
    # Generate the minimal, clean HTML report
    html_report = render_html_report(eda_data, analysis)
    
    return {
        "summary": eda_data.get("summary"),
        "dtypes": eda_data.get("dtypes"),
        "null_info": eda_data.get("null_info"),
        "numeric_stats": eda_data.get("numeric_stats"),
        "correlation_matrix": eda_data.get("correlation_matrix"),
        "categorical_stats": eda_data.get("categorical_stats"),
        "gemini_analysis": analysis,
        "html_report": html_report
    }
