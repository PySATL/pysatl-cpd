pip install poetry

git clone https://github.com/PySATL/pysatl-cpd.git
Set-Location pysatl-cpd

py -3 -m venv .venv

. .\.venv\Scripts\Activate.ps1
pip install poetry
poetry install
deactivate

