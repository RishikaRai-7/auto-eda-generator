import sys
import os
import traceback

try:
    print("Adding project directory to path...")
    sys.path.append(r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator")
    
    print("Importing run_eda...")
    from eda import run_eda
    
    print("Running EDA on sample.csv...")
    csv_path = r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator\sample.csv"
    res = run_eda(csv_path)
    
    print("EDA execution finished successfully!")
    
    # Save the output
    import pprint
    out_path = r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator\test_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pprint.pformat(res))
    print(f"Successfully wrote output to {out_path}")

except Exception as e:
    err_path = r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator\verify_error.txt"
    with open(err_path, "w", encoding="utf-8") as f:
        f.write(f"Error occurred: {str(e)}\n")
        f.write(traceback.format_exc())
    print(f"Error written to {err_path}")
