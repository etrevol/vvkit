@echo off
rem run_sod.bat - Wrapper to run Athena++ via WSL
set INPUT_FILE=%~nx1
wsl -d Ubuntu -- bash -c "/home/etrevol/athena-collab/bin/athena -i %INPUT_FILE%"
wsl -d Ubuntu -- bash -c "cp Sod.block0.out1.00001.tab result.tab"
