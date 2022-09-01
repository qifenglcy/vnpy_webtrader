@echo off

start cmd /k "cd /d C:\Users\qifeng&&python run_ui_web_rpc.py"
timeout /T 10
start cmd /k "cd /d C:\Users\qifeng\vnpy_web &&uvicorn main:app --host=0.0.0.0 --port=8000 --reload"


