@echo off
cd /d "%~dp0"
start "" http://localhost:8050
python -m mas.main
pause
