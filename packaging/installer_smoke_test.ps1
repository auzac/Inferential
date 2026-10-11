<#
.SYNOPSIS
    Install, launch (smoke test) and uninstall the Windows installer silently.

.DESCRIPTION
    powershell -ExecutionPolicy Bypass -File packaging\installer_smoke_test.ps1 -Installer packaging\output\InferentialModelStudio-<version>-setup-win64.exe

    Installs per-user into a temporary folder, runs the installed exe with --smoke-test (window
    creation plus aligning and training all four model types), then runs the full workflow on
    the bundled example project (load data, align, train, write the model card), uninstalls,
    and checks that the program folder is gone. Uses a private
    settings folder so the developer's settings are untouched. Exit code 0 = pass.

    NOTE: written and reviewed on Linux; it has NOT been executed on Windows yet.
#>
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Installer)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Installer = Resolve-Path $Installer
$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("ims-installer-test-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$AppDir = Join-Path $Work "app"
$TestHome = Join-Path $Work "home"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

function Invoke-AndWait([string]$Exe, [string[]]$Arguments) {
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments -Wait -PassThru
    return $p.ExitCode
}

Write-Host "== Installing silently into $AppDir"
$code = Invoke-AndWait $Installer @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/DIR=`"$AppDir`"", "/TASKS=`"`"", "/LOG=`"$Work\install.log`"")
if ($code -ne 0) { throw "Installer failed with exit code $code (see $Work\install.log)" }
$Exe = Join-Path $AppDir "InferentialModelStudio.exe"
if (-not (Test-Path $Exe)) { throw "Installed exe not found: $Exe" }

$env:INFERENTIAL_STUDIO_HOME = $TestHome
try {
    Write-Host "== Smoke test (window + numeric self-check)"
    $code = Invoke-AndWait $Exe @("--smoke-test")
    if ($code -ne 0) { throw "Smoke test failed with exit code $code (logs: $TestHome\logs)" }

    $Example = Join-Path $AppDir "_internal\examples\synthetic_distillation\sample_project.imsproj"
    if (-not (Test-Path $Example)) { throw "Bundled example project missing: $Example" }
    Write-Host "== Workflow smoke test: example project -> align -> train -> model card"
    $code = Invoke-AndWait $Exe @("`"$Example`"", "--smoke-test")
    if ($code -ne 0) { throw "Example workflow failed with exit code $code (logs: $TestHome\logs)" }
    if (-not (Test-Path (Join-Path $TestHome "smoke-model-card.html"))) { throw "Model card was not written" }
}
finally {
    Remove-Item Env:\INFERENTIAL_STUDIO_HOME -ErrorAction SilentlyContinue
}

Write-Host "== Uninstalling"
$Uninstaller = Get-ChildItem -Path $AppDir -Filter "unins*.exe" | Select-Object -First 1
if ($null -eq $Uninstaller) { throw "Uninstaller not found in $AppDir" }
$code = Invoke-AndWait $Uninstaller.FullName @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
if ($code -ne 0) { throw "Uninstaller failed with exit code $code" }
Start-Sleep -Seconds 2   # the uninstaller deletes itself from a helper process
if (Test-Path $Exe) { throw "Program files remain after uninstall: $AppDir" }

Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
Write-Host "Installer smoke test passed."
