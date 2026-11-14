@echo off
title Motion Mentor - Butterfly Effect
cd /d "%~dp0"
set "PYTHON=py -3"
%PYTHON% --version >nul 2>nul
if errorlevel 1 set "PYTHON=python"
%PYTHON% main.py %*
if errorlevel 1 (
  echo.
  echo Something went wrong - read the message above.
  pause
)
