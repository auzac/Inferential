<#
.SYNOPSIS
    Build the Windows x64 one-folder application and run a smoke test.

.DESCRIPTION
    Run from the repository root in PowerShell:

        powershell -ExecutionPolicy Bypass -File packaging\windows_build.ps1

    Steps: create/refresh .venv-build with the pinned dependencies, run PyInstaller,
    launch the built exe with --smoke-test, then zip the folder and write a SHA-256 checksum.
    The Windows installer (Inno Setup) is added in Milestone 5.

    NOTE: written and syntax-reviewed on Linux; it has NOT been executed on Windows yet.
#>
[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersion = "-3.12"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$Venv = Join-Path $Root ".venv-build"
$Out = Join-Path $Root "packaging\output"
$AppName = "InferentialModelStudio"

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

Write-Host "== Preparing build environment in $Venv"
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Invoke-Checked $Python @($PythonVersion, "-m", "venv", $Venv)
}
$Py = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $Py @("-m", "pip", "install", "-r", "requirements-lock.txt")
Invoke-Checked $Py @("-m", "pip", "install", "-e", ".", "--no-deps")

$Version = (& $Py -c "import inferential_studio; print(inferential_studio.__version__)").Trim()
Write-Host "== Building $AppName $Version"

Write-Host "== Running tests"
Invoke-Checked $Py @("-m", "pytest", "-q")

if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
Invoke-Checked $Py @(
    "-m", "PyInstaller", "--noconfirm", "--clean",
    "--distpath", (Join-Path $Out "dist"),
    "--workpath", (Join-Path $Out "build"),
    "packaging\pyinstaller.spec"
)

$Exe = Join-Path $Out "dist\$AppName\$AppName.exe"
if (-not (Test-Path $Exe)) { throw "Build did not produce $Exe" }

Write-Host "== Smoke test: $Exe --smoke-test"
$env:INFERENTIAL_STUDIO_HOME = Join-Path $Out "smoke-home"   # keep the developer's settings untouched
$Proc = Start-Process -FilePath $Exe -ArgumentList "--smoke-test" -Wait -PassThru
Remove-Item Env:\INFERENTIAL_STUDIO_HOME
if ($Proc.ExitCode -ne 0) { throw "Smoke test failed with exit code $($Proc.ExitCode). See $Out\smoke-home\logs" }

$Zip = Join-Path $Out "$AppName-$Version-win64.zip"
Write-Host "== Creating $Zip"
Compress-Archive -Path (Join-Path $Out "dist\$AppName") -DestinationPath $Zip
$Hash = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower()
"$Hash  $(Split-Path $Zip -Leaf)" | Set-Content -Encoding ascii "$Zip.sha256"

Write-Host ""
Write-Host "Build OK: $Exe"
Write-Host "Archive:  $Zip"
Write-Host "SHA-256:  $Hash"
