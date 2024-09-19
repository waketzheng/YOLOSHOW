if exist venv (
    echo venv exists, skip virtual environment initial.
) else (
    echo Create local virtual environment at: venv
    python -m venv venv --prompt venv3.11-YOLOSHOW --upgrade-deps
    venv\Scripts\pip install -r requirements.txt
    echo Compile ui and qrc file to fix: Could not create pixmap from XXX
    ::Inspired from https://blog.csdn.net/ovdoesLV/article/details/138873171
    venv\Scripts\pyside6-uic.exe ui\YOLOSHOWUI.ui -o ui\YOLOSHOWUI.py
    venv\Scripts\pyside6-rcc.exe ui\YOLOSHOWUI.qrc -o ui\YOLOSHOWUI_rc.py
    %setx YOLOSHOW_HOST http://localhost/downloads%
    echo Going to start main script
)
venv\Scripts\python main.py
pause
