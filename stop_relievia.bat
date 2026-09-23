@echo off
setlocal EnableDelayedExpansion

title Relievia - Smart Bed Stop Utility

echo ======================================================================
echo           RELIEVIA: Autonomous Pressure Injury Prevention
echo                     Stop Backend Utility
echo ======================================================================
echo.

powershell -NoProfile -Command ^
  "$conns = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "if ($conns) { foreach ($pidToKill in $conns) { Write-Host 'Terminating Relievia backend PID:' $pidToKill; taskkill /F /T /PID $pidToKill | Out-Null }; Write-Host '[OK] Relievia ML Backend stopped successfully.' } else { Write-Host '[INFO] No Relievia backend was found listening on port 5000.' }"

echo.
echo You may close this window.
powershell -NoProfile -Command "Start-Sleep -Seconds 2"
