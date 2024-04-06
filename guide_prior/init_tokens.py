import subprocess

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Successfully ran {script_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {script_path}: {e}")
        print(e.stderr)

# Update the scripts list if needed
scripts = [
    '../guide_prior/get_doc_phecode.py',
    '../guide_prior/get_prior_GMM.py',
    '../guide_prior/get_token_counts.py'
]

for script in scripts:
    run_script(script)

