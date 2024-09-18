if exist venv (
    echo venv exists, skip virtual environment initial.
) else (
    echo Create local virtual environment at: venv
    python -m venv venv --prompt venv3.11-YOLOSHOW --upgrade-deps
    venv\Scripts\pip install -r requirements.txt
)
venv\Scripts\python main.py
pause
