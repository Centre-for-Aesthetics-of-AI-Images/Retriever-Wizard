@echo off
setlocal

set "HERE=%~dp0"
set "VENV_PY=%HERE%.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo Missing venv python at: "%VENV_PY%"
  echo Create it with: py -3.12 -m venv .venv
  exit /b 1
)

rem Pass through any extra args to streamlit (e.g. --server.port 8502)
"%VENV_PY%" -m streamlit run "%HERE%ReWiz.py" %*

endlocal
