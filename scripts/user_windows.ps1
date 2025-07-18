pip install poetry

git clone https://github.com/PySATL/pysatl-cpd.git
Set-Location pysatl-cpd

$pythonExecutable = $null
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonExecutable = "python3"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $version = & python -c "import sys; print(sys.version_info[0])"
    if ($version.Trim() -eq '3') {
        $pythonExecutable = "python"
    }
}

if (-not $pythonExecutable) {
    Write-Error "Ошибка: Требуется Python 3, но он не найден в PATH."
    exit 1
}

Write-Host "Используется '$pythonExecutable' для создания виртуального окружения."

& $pythonExecutable -m venv .venv

. .\.venv\Scripts\Activate.ps1
pip install poetry
poetry install
deactivate
