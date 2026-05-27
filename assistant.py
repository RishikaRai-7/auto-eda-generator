import os
import json
import google.generativeai as genai
from typing import Dict, Any

class GeminiAssistant:
    """
    Assistant class that uses the Gemini 2.5 API (gemini-2.5-flash) to interpret
    Exploratory Data Analysis (EDA) metrics and generate a senior analyst report.
    """
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please configure this environment variable to enable Gemini features."
            )
        
        # Configure the genai library with the API key
        genai.configure(api_key=self.api_key)
        
        # Define the system instruction to set the persona
        self.system_instruction = (
            "You are an expert senior data analyst, statistician, and data quality engineer. "
            "Your job is to analyze the provided Exploratory Data Analysis (EDA) metrics of a dataset "
            "and produce high-value, deep, statistically-grounded insights, plain-English summaries, "
            "recommended statistical tests, and detailed data quality issue audits. "
            "You must format your response strictly as a JSON object."
        )
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=self.system_instruction
        )

    def analyze_eda(self, eda_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw pandas EDA output and calls Gemini to produce the statistical report.
        
        Args:
            eda_metrics: A dictionary containing data shape, types, stats, nulls, etc.
            
        Returns:
            A dictionary containing parsed Gemini analysis matching the requested JSON schema.
        """
        # Base JSON prompt structure
        prompt = f"""
Analyze the following Exploratory Data Analysis (EDA) summary metrics for a dataset.
The metrics are formatted as a JSON object containing:
- "summary": overall shape (rows, columns) and count of column types
- "dtypes": data type of each column
- "null_info": counts and percentages of missing values for each column
- "numeric_stats": descriptive summary statistics (mean, std, min, percentiles, max) for numeric columns
- "correlation_matrix": Pearson correlation coefficients between numeric columns
- "categorical_stats": cardinality and top frequency counts for categorical columns

EDA METRICS DATA:
{json.dumps(eda_metrics, indent=2)}

Provide your response in JSON format. The JSON MUST strictly follow this schema:
{{
  "summary": "A 2-3 sentence plain English summary explaining what this dataset represents, its general structure, and main theme.",
  "key_insights": [
    "Insight 1 (describe notable patterns, distributions, correlations, or anomalies in the dataset, citing specific values like means, percentages, or correlation coefficients from the metrics)",
    "Insight 2...",
    "Insight 3 (must have between 3 to 5 clear, high-value bullet points)"
  ],
  "recommended_statistical_tests": [
    {{
      "test_name": "Name of the statistical test (e.g. Chi-Square Test of Independence, Two-sample T-test, ANOVA, Mann-Whitney U, Pearson Correlation Test)",
      "columns_involved": ["column_1", "column_2"],
      "rationale": "A clear description of why this test is appropriate based on their data types (numeric vs categorical, distributions, cardinality) and what hypothesis it helps verify."
    }}
  ],
  "data_quality_issues": [
    {{
      "issue_type": "Type of issue (e.g. Missing Data, Extreme Outliers, Multi-collinearity, Class Imbalance, High Cardinality)",
      "columns_involved": ["column_name"],
      "description": "Detailed explanation of the issue, which columns are affected, citing specific figures (e.g. missing percentages, correlation values), and a brief recommendation on how to clean or pre-process it."
    }}
  ]
}}

Ensure that:
1. You only return the raw JSON object and no other surrounding text or formatting.
2. The insights and findings are deep and specific, quoting actual values from the EDA metrics.
3. If there are no obvious statistical tests or quality issues, suggest best practice validation rules based on standard data engineering.
"""
        try:
            # Request analysis using JSON response mode
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Parse the response text
            analysis_result = json.loads(response.text)
            return analysis_result
            
        except Exception as e:
            return {
                "error": f"Failed to get analysis from Gemini: {str(e)}",
                "summary": "Unable to generate plain English dataset summary because the Gemini API call failed.",
                "key_insights": [
                    "Error occurred during LLM processing.",
                    "Verify your GEMINI_API_KEY environment variable is set correctly and has permission to access gemini-2.5-flash."
                ],
                "recommended_statistical_tests": [
                    {
                        "test_name": "None",
                        "columns_involved": [],
                        "rationale": f"Analysis failed: {str(e)}"
                    }
                ],
                "data_quality_issues": [
                    {
                        "issue_type": "API Error",
                        "columns_involved": [],
                        "description": "The backend was unable to communicate with the Gemini API."
                    }
                ]
            }
