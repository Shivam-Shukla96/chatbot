



<!-- commands to run the bakend after getting it from github -->

python -m venv .venv;

.\.venv\Scripts\Activate.ps1;

pip install -r requirements.txt;

uvicorn app.main:app --reload