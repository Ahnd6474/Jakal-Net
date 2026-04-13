param(
    [string]$VenvPath = ".venv-cuda",
    [string]$PythonExe = "python",
    [string]$TorchVersion = "2.4.1",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu124",
    [switch]$InstallToolkit,
    [string]$ToolkitVersion = "12.6"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvFullPath = Join-Path $repoRoot $VenvPath
$baseRequirements = Join-Path $repoRoot "requirements-base.txt"

if (-not (Test-Path $venvFullPath)) {
    & $PythonExe -m venv $venvFullPath
}

$python = Join-Path $venvFullPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python was not found at $python"
}

& $python -m pip install --upgrade pip
& $python -m pip install -r $baseRequirements
& $python -m pip install "torch==$TorchVersion" --index-url $TorchIndexUrl

if ($InstallToolkit) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($null -eq $nvcc) {
        & winget install --id Nvidia.CUDA --version $ToolkitVersion -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw (
                "CUDA Toolkit installation failed with exit code $LASTEXITCODE. " +
                "Run an elevated PowerShell session and approve the NVIDIA installer prompt."
            )
        }
    }
}

Write-Host ""
Write-Host "CUDA environment ready at $venvFullPath"
Write-Host "Set PYTHONPATH before running scripts:"
Write-Host '  $env:PYTHONPATH = "src"'
Write-Host ""
Write-Host "Verification:"
Write-Host "  $python -c `"import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())`""
Write-Host "  $python $repoRoot\scripts\smoke_test.py --device cuda"
