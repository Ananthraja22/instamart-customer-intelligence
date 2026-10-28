@echo off
setlocal
cd /d "%~dp0"

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON=%CONDA_PREFIX%\python.exe"
)

if not defined PYTHON if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
)

if not defined PYTHON set "PYTHON=python"

call :run scripts\data_prep\audit_raw_data.py
if errorlevel 1 exit /b 1

call :run scripts\data_prep\clean_data.py
if errorlevel 1 exit /b 1

call :run scripts\analytics\run_eda.py
if errorlevel 1 exit /b 1

call :run scripts\data_prep\build_database.py
if errorlevel 1 exit /b 1

call :run scripts\data_prep\validate_data.py
if errorlevel 1 exit /b 1

call :run scripts\analytics\run_sql_analysis.py
if errorlevel 1 exit /b 1

call :run scripts\modeling\build_customer_segments.py
if errorlevel 1 exit /b 1

call :run scripts\support\triage_tickets.py
if errorlevel 1 exit /b 1

call :run scripts\agent\refund_agent.py
if errorlevel 1 exit /b 1

call :run scripts\agent\policy_assistant.py
if errorlevel 1 exit /b 1

echo.
echo Pipeline completed successfully.
exit /b 0

:run
echo.
echo ============================================================
echo Running: %*
echo ============================================================
"%PYTHON%" %*
if errorlevel 1 (
    echo.
    echo ERROR: %* failed.
    exit /b 1
)
exit /b 0
