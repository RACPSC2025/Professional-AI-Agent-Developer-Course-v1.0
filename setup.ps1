# Check if Python is installed
try {
    python --version
} catch {
    Write-Host "Python not found. Please install Python 3.10+ first."
    exit
}

# Create Virtual Environment
Write-Host "Creating virtual environment 'venv'..."
python -m venv venv

# Activate Virtual Environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install Requirements
Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host "Setup complete! To activate the environment in the future, run: .\venv\Scripts\Activate.ps1"
