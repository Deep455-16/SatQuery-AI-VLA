@echo off
echo ====================================================
echo SATQUERY AI - DEPENDENCY INSTALLER
echo ====================================================
echo.

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo [2/3] Installing Python backend dependencies...
pip install -r requirements.txt
pip install pandas pyarrow psutil peft transformers torch torchvision
pip install -e .

echo [3/3] Installing and building Node.js frontend...
cd frontend
call npm install
call npm run build
cd ..

echo.
echo ====================================================
echo Installation Complete!
echo The frontend has been built into frontend/dist.
echo You can now run start_app.bat to launch SatQuery AI.
echo ====================================================
pause
