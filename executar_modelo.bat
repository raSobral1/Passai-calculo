@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0python_portatil\python.exe" (
    "%~dp0python_portatil\python.exe" modelo_travessia.py %*
    goto fim
)

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 modelo_travessia.py %*
    goto fim
)

where python >nul 2>nul
if %errorlevel%==0 (
    python modelo_travessia.py %*
    goto fim
)

echo Python 3.10 ou superior nao foi encontrado neste computador.
echo Instale o Python e marque a opcao para adiciona-lo ao PATH.

:fim
echo.
pause


