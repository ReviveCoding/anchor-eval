[CmdletBinding()]
param(
    [string]$ScriptPath = (Join-Path $PSScriptRoot "run_anchor_eval_v15.ps1")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    throw "Target PowerShell script was not found: $ScriptPath"
}

$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path -LiteralPath $ScriptPath).Path,
    [ref]$tokens,
    [ref]$parseErrors
)

if ($parseErrors.Count -gt 0) {
    $details = $parseErrors | ForEach-Object {
        "$($_.Extent.StartLineNumber):$($_.Extent.StartColumnNumber) $($_.Message)"
    }
    throw "PowerShell parser failed for ${ScriptPath}:`n$($details -join [Environment]::NewLine)"
}

[pscustomobject]@{
    status      = "PASS"
    script_path = (Resolve-Path -LiteralPath $ScriptPath).Path
    token_count = $tokens.Count
} | ConvertTo-Json -Depth 3
