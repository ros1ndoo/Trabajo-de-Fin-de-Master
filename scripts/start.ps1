param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    throw 'No existe .venv. Ejecute scripts/setup.ps1 una vez antes de iniciar la aplicación.'
}
& '.venv/Scripts/python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --browser.gatherUsageStats false
