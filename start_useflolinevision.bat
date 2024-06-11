@echo off
:: Logging the start of the script
echo Starting Object Counter Script > D:\usefloline\ultra\ultralytics.txt

:: Navigate to the script directory
cd "D:\usefloline\ultra\ultralytics"
if %errorlevel% neq 0 (
    echo Failed to navigate to the directory >> D:\usefloline\ultra\ultralytics.txt
    exit /b 1
)

@REM :: Activate the virtual environment (optional)
@REM call C:\path\to\your\venv\Scripts\activate
@REM if %errorlevel% neq 0 (
@REM     echo Failed to activate the virtual environment >> C:\Scripts\object_counter_log.txt
@REM     exit /b 1
@REM )

echo Starting USE-FLOLINE Vision Program ...
echo press 'q' to quit program

:: Run the Python script
python usefloline_cam.py >>D:\usefloline\ultra\ultralytics.txt 2>&1
if %errorlevel% neq 0 (
    echo Python script execution failed >> D:\usefloline\ultra\ultralytics.txt
    exit /b 1
)

:: Log successful execution
echo Script executed successfully >> D:\usefloline\ultra\ultralytics.txt
