[CmdletBinding()]
param(
    [string]$PythonExe,
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepositoryRoot -PathType Container)) {
    throw "Repository root was not found: $RepositoryRoot"
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $PythonExe = $VenvPython
    }
    else {
        $PythonExe = "py"
    }
}

$ProbeDirectory = Join-Path $RepositoryRoot "outputs\preflight_python_file_probe"
New-Item -ItemType Directory -Force -Path $ProbeDirectory | Out-Null
$ProbePath = Join-Path $ProbeDirectory "python_file_invocation_probe.py"
$ProbeCode = @'
import sys
print("ANCHOREVAL_PYTHON_FILE_INVOKE=PASS")
print("PYTHON_VERSION=" + sys.version.split()[0])
'@
$Utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ProbePath, $ProbeCode, $Utf8WithoutBom)

if ($PythonExe -eq "py") {
    & py -3.11 $ProbePath
}
else {
    & $PythonExe $ProbePath
}
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    throw "Python file-invocation probe failed with exit code $ExitCode."
}

[pscustomobject]@{
    status      = "PASS"
    python      = $PythonExe
    probe_path  = $ProbePath
} | ConvertTo-Json -Depth 3
