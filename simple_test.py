import sys
try:
    with open(r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator\hello.txt", "w") as f:
        f.write("Python successfully executed and wrote this file!\n")
        f.write(f"Python version: {sys.version}\n")
except Exception as e:
    with open(r"C:\Users\rishi\.gemini\antigravity\scratch\auto_eda_generator\hello_err.txt", "w") as f:
        f.write(f"Error: {str(e)}\n")
