@echo off
title Vetorizador de Quadras Fiscais - Servidor Web
cd /d "%~dp0app"
echo ========================================================
echo   Iniciando Vetorizador de Quadras Fiscais...
echo   Utilizando ambiente Python do QGIS 3.40
echo ========================================================
echo.
echo Abra o seu navegador no endereco: http://localhost:5055
echo Pressione Ctrl+C para encerrar o servidor.
echo.
"C:\Program Files\QGIS 3.40.4\bin\python-qgis-ltr.bat" server.py
pause
