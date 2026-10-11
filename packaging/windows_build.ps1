<#
.SYNOPSIS
    Build the Windows x64 one-folder application and run a smoke test.

.DESCRIPTION
    Run from the repository root in PowerShell:

        powershell -ExecutionPolicy Bypass -File packaging\windows_build.ps1

    Steps: create/refresh .venv-build with the pinned dependencies, run the tests, run
    PyInstaller, launch the built exe with --smoke-test, zip the folder (portable build), then
    build the per-user installer with Inno Setup 6 (if ISCC.exe is found) and run the installer
    smoke test (silent install, launch, uninstall). SHA-256 checksums are written for both.

    NOTE: written and syntax-reviewed on Linux; it has NOT been executed on Windows yet.
#>
[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersion = "-3.12",
    [string]$Iscc = ""   # path to ISCC.exe; default: PATH, then the usual install folders
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

if (-not $Iscc) {
    $cmd = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($cmd) { $Iscc = $cmd.Source }
    foreach ($candidate in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (-not $Iscc -and (Test-Path $candidate)) { $Iscc = $candidate }
    }
}
$Setup = $null
if ($Iscc) {
    Write-Host "== Building installer with $Iscc"
    Invoke-Checked $Iscc @("/Qp", "/DAppVersion=$Version", "/DSourceDir=$(Join-Path $Out "dist\$AppName")", "packaging\installer\InferentialModelStudio.iss")
    $Setup = Join-Path $Out "$AppName-$Version-setup-win64.exe"
    if (-not (Test-Path $Setup)) { throw "Inno Setup did not produce $Setup" }
    $SetupHash = (Get-FileHash -Algorithm SHA256 $Setup).Hash.ToLower()
    "$SetupHash  $(Split-Path $Setup -Leaf)" | Set-Content -Encoding ascii "$Setup.sha256"
    Write-Host "== Installer smoke test"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "installer_smoke_test.ps1") -Installer $Setup
    if ($LASTEXITCODE -ne 0) { throw "Installer smoke test failed" }
} else {
    Write-Warning "Inno Setup 6 (ISCC.exe) not found: installer not built. Install it from https://jrsoftware.org/isinfo.php or pass -Iscc."
}

Write-Host ""
Write-Host "Build OK: $Exe"
Write-Host "Archive:  $Zip"
Write-Host "SHA-256:  $Hash"
if ($Setup) { Write-Host "Installer: $Setup" }
