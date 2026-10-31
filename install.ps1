# Creates the conda environment in .\.conda-env, installing Miniforge locally first if conda is not available.
# Windows support is experimental.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$EnvDir = "$PSScriptRoot\.conda-env"
$MiniforgeDir = "$PSScriptRoot\.miniforge"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    $Conda = "conda"
} elseif (Test-Path "$MiniforgeDir\Scripts\conda.exe") {
    $Conda = "$MiniforgeDir\Scripts\conda.exe"
} else {
    Write-Host "conda not found, installing Miniforge to $MiniforgeDir"
    $Installer = "$env:TEMP\Miniforge3-Windows-x86_64.exe"
    Invoke-WebRequest "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile $Installer
    Start-Process -Wait -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$MiniforgeDir"
    Remove-Item $Installer
    $Conda = "$MiniforgeDir\Scripts\conda.exe"
}

if (Test-Path $EnvDir) {
    & $Conda env update -y -p $EnvDir -f environment.yml --prune
} else {
    & $Conda env create -y -p $EnvDir -f environment.yml
}

Write-Host "Setup complete. Run the interface with run.bat, or activate with: conda activate $EnvDir"
