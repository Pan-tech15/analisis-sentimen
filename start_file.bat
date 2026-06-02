@echo off
title Auto-Launcher Web App

set SCRIPT_DIR=%~dp0
set BACKEND_DIR=%SCRIPT_DIR%backend
set FRONTEND_DIR=%SCRIPT_DIR%frontend

echo ==============================
echo  Menjalankan Backend (Flask)
echo ==============================
start "Backend Server" cmd /k "cd /d %BACKEND_DIR% && venv\Scripts\activate && python run.py"

echo ==============================
echo  Menjalankan Frontend (HTTP Server)
echo ==============================
start "Frontend Server" cmd /k "cd /d %FRONTEND_DIR% && python -m http.server 5500"

:: ------------------------------------------------------------
:: Fungsi pengecekan port dengan PowerShell (TCP connect)
:: ------------------------------------------------------------
echo Menunggu backend siap di port 5000...
:wait_backend
powershell -Command "try { $tcp = New-Object System.Net.Sockets.TcpClient; $result = $tcp.BeginConnect('localhost', 5000, $null, $null); $wait = $result.AsyncWaitHandle.WaitOne(1000); if ($wait -and $tcp.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 1 /nobreak >nul
    goto wait_backend
)
echo Backend siap.

echo Menunggu frontend siap di port 5500...
:wait_frontend
powershell -Command "try { $tcp = New-Object System.Net.Sockets.TcpClient; $result = $tcp.BeginConnect('localhost', 5500, $null, $null); $wait = $result.AsyncWaitHandle.WaitOne(1000); if ($wait -and $tcp.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 1 /nobreak >nul
    goto wait_frontend
)
echo Frontend siap.

echo ==============================
echo  Membuka browser ke halaman login
echo ==============================
start http://localhost:5500/admin/login.html

echo Selesai. Tutup jendela CMD untuk menghentikan server.