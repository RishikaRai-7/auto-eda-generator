import pandas as pd
import numpy as np
from typing import Dict, Any

def run_eda(file_path_or_buf) -> Dict[str, Any]:
    """
    Performs Exploratory Data Analysis (EDA) on a CSV file.
    
    Args:
        file_path_or_buf: File path or file-like object containing CSV data.
        
    Returns:
        A dictionary containing EDA summary metrics.
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path_or_buf)
    except Exception as e:
        return {"error": f"Failed to parse CSV file: {str(e)}"}
        
    if df.empty:
        return {"error": "The uploaded CSV file is empty."}
        
    # 1. Shape
    rows, cols = df.shape
    
    # 2. Column Types (convert to string for JSON serialization)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # 3. Null counts and percentages
    null_info = {}
    null_counts = df.isnull().sum()
    for col in df.columns:
        count = int(null_counts[col])
        pct = float((count / rows) * 100) if rows > 0 else 0.0
        null_info[col] = {
            "count": count,
            "percentage": round(pct, 2)
        }
        
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 4. Basic stats (mean/std/min/max, percentiles) for numeric columns
    basic_stats = {}
    if numeric_cols:
        # Use pandas describe and convert to dict, ensuring JSON serializable values
        desc = df[numeric_cols].describe()
        # Handle cases where std might be NaN (e.g. single row)
        desc = desc.fillna(0)
        
        for col in numeric_cols:
            col_desc = desc[col]
            basic_stats[col] = {
                "count": int(col_desc.get("count", 0)),
                "mean": float(col_desc.get("mean", 0.0)),
                "std": float(col_desc.get("std", 0.0)),
                "min": float(col_desc.get("min", 0.0)),
                "25%": float(col_desc.get("25%", 0.0)),
                "50%": float(col_desc.get("50%", 0.0)),
                "75%": float(col_desc.get("75%", 0.0)),
                "max": float(col_desc.get("max", 0.0))
            }
            
    # 5. Correlation matrix for numeric columns
    correlation_matrix = {}
    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].corr(method='pearson')
        # Replace NaN correlations with 0.0
        corr_df = corr_df.fillna(0.0)
        for col in corr_df.columns:
            correlation_matrix[col] = {other_col: float(val) for other_col, val in corr_df[col].items()}
            
    # 6. Top value counts for categorical/boolean columns
    categorical_stats = {}
    for col in categorical_cols:
        unique_count = int(df[col].nunique(dropna=True))
        # Get top 10 most common values
        top_counts = df[col].value_counts(dropna=False).head(10)
        value_distribution = []
        for val, count in top_counts.items():
            # Ensure val is string/JSON serializable
            val_str = str(val) if pd.notnull(val) else "Missing/Null"
            value_distribution.append({
                "value": val_str,
                "count": int(count),
                "percentage": round(float((count / rows) * 100), 2) if rows > 0 else 0.0
            })
            
        categorical_stats[col] = {
            "unique_count": unique_count,
            "top_values": value_distribution
        }
        
    return {
        "summary": {
            "rows": rows,
            "columns": cols,
            "numeric_columns_count": len(numeric_cols),
            "categorical_columns_count": len(categorical_cols)
        },
        "dtypes": dtypes,
        "null_info": null_info,
        "numeric_stats": basic_stats,
        "correlation_matrix": correlation_matrix,
        "categorical_stats": categorical_stats
    }
