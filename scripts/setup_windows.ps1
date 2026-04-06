# Run from project root: KCC
if (Test-Path .venv) {
  Write-Host "Existing .venv detected. Recreating with Python 3.11..."
  Remove-Item -Recurse -Force .venv
}

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host "Created .env. Please set OPENAI_API_KEY before running experiments."
} else {
  Write-Host ".env already exists."
}

python -c "import datasets, faiss, openai, tiktoken, numpy, pandas, sklearn, tqdm, dotenv; print('environment_ok')"

Write-Host "Note: Hotpot evaluation should use official hotpot_evaluate_v1.py script (not hotpot-evaluate pip package)."
