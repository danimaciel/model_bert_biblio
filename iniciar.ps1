$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    Write-Host 'Crie o ambiente com: python -m venv .venv'
    Write-Host 'Instale com: .venv\Scripts\python -m pip install -r requirements.txt'
    exit 1
}
& $appPython -m streamlit run app.py --server.address=127.0.0.1 --server.port=8517
