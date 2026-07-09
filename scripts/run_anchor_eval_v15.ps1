& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"

    # ========================================================================
    # AnchorEval v1.5 — Windows 11 / RTX 4090 local execution
    #
    # Runs:
    #   environment -> CPU quality gates -> package checks -> FinQA processing
    #   -> isolated BF16 Qwen load probe -> checkpointed FinQA scoring -> C0-C5 fitting
    #   -> checkpointed JudgeBench scoring -> immutable final-only evaluation.
    #
    # It intentionally does not run DPO. The repository only creates isolated,
    # equal-budget pools; a valid downstream DPO comparison needs its own
    # independent model-quality experiment.
    #
    # Native commands are called through Invoke-NativeChecked. This prevents a
    # failed pip/build/test/scoring command from being masked by a later command.
    # ========================================================================

    $RepoRoot = "C:\Users\bjw-0\Downloads\Apple03_anchor-eval"
    $DataRoot = "C:\Users\bjw-0\Downloads\Project_Data"
    $FinQARoot = Join-Path $DataRoot "FinQA-main"
    $JudgeBenchRoot = Join-Path $DataRoot "JudgeBench"

    $ExpectedVersion = "0.15.0"
    $JudgeModel = "Qwen/Qwen3-4B-Instruct-2507"
    # Windows default: BF16 direct loading. The 4-bit bitsandbytes path remains
    # available only as an explicit opt-in after a successful 4-bit probe.
    $JudgePrecision = "bf16"
    $MaxInputTokens = 1536
    $ScoringChunkSize = 32
    $FinQAMaxExamples = 500
    $Seed = 1729

    # Keep this unchanged while resuming an interrupted run. Change it only
    # when deliberately starting a fresh evidence directory.
    $RunId = "v15_finqa_judgebench"

    $VenvPath = Join-Path $RepoRoot ".venv"
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    $ExecutionRoot = Join-Path $RepoRoot "outputs\local_execution\$RunId"
    $PreparedDataRoot = Join-Path $ExecutionRoot "prepared_data"
    $ConfigRoot = Join-Path $ExecutionRoot "configs"
    $CheckpointRoot = Join-Path $ExecutionRoot "judge_checkpoints"
    $LogRoot = Join-Path $ExecutionRoot "logs"
    $InlinePythonRoot = Join-Path $ExecutionRoot "inline_python"

    # Keep large Hugging Face files outside the source repository.
    $HFRoot = "C:\hf-anchor-eval"
    $env:HF_HOME = $HFRoot
    $env:HF_HUB_CACHE = Join-Path $HFRoot "hub"
    $env:HF_DATASETS_CACHE = Join-Path $HFRoot "datasets"
    $env:TRANSFORMERS_CACHE = Join-Path $HFRoot "transformers"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:PYTHONUTF8 = "1"
    $env:MPLBACKEND = "Agg"
    $env:MPLCONFIGDIR = Join-Path $ExecutionRoot ".mplconfig"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PIP_NO_INPUT = "1"
    $env:TOKENIZERS_PARALLELISM = "false"

    function Assert-Path {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][string]$Label
        )
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "$Label was not found: $Path"
        }
    }

    function Invoke-NativeChecked {
        param(
            [Parameter(Mandatory = $true)][string]$Executable,
            [Parameter()][object[]]$Arguments = @(),
            [Parameter()][string]$Description = $Executable
        )
        & $Executable @Arguments
        $NativeExitCode = $LASTEXITCODE
        if ($NativeExitCode -ne 0) {
            $RenderedArguments = ($Arguments | ForEach-Object { [string]$_ }) -join " "
            throw "Native command failed: $Description (exit code $NativeExitCode): $Executable $RenderedArguments"
        }
    }

    function Invoke-Python {
        param(
            [Parameter(Mandatory = $true)][object[]]$Arguments,
            [Parameter()][string]$Description = "Python command"
        )
        Invoke-NativeChecked -Executable $PythonExe -Arguments $Arguments -Description $Description
    }

    function Write-InlinePythonScript {
        param(
            [Parameter(Mandatory = $true)][string]$Name,
            [Parameter(Mandatory = $true)][string]$Code
        )
        if ($Name -notmatch '^[A-Za-z0-9_.-]+$') {
            throw "Inline Python script name contains unsupported characters: $Name"
        }
        New-Item -ItemType Directory -Force -Path $InlinePythonRoot | Out-Null
        $ScriptPath = Join-Path $InlinePythonRoot "${Name}.py"
        $Utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($ScriptPath, $Code, $Utf8WithoutBom)
        return $ScriptPath
    }

    function Invoke-InlinePythonScript {
        param(
            [Parameter(Mandatory = $true)][string]$Name,
            [Parameter(Mandatory = $true)][string]$Code,
            [Parameter()][object[]]$ScriptArguments = @(),
            [Parameter()][string]$Description = "Inline Python script"
        )
        $ScriptPath = Write-InlinePythonScript -Name $Name -Code $Code
        Invoke-Python -Arguments (@($ScriptPath) + @($ScriptArguments)) -Description $Description
    }

    function Invoke-Checked {
        param(
            [Parameter(Mandatory = $true)][string]$Name,
            [Parameter(Mandatory = $true)][scriptblock]$Action
        )
        Write-Host ""
        Write-Host "========================================================================"
        Write-Host $Name
        Write-Host "========================================================================"
        try {
            & $Action
        }
        catch {
            throw "FAILED: $Name`n$($_.Exception.Message)"
        }
    }

    function Get-CsvRowCount {
        param([Parameter(Mandatory = $true)][string]$Path)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "CSV was not found: $Path"
        }
        return @(@(Import-Csv -LiteralPath $Path)).Count
    }

    function Assert-CsvRowCount {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [Parameter(Mandatory = $true)][int]$ExpectedRows,
            [Parameter(Mandatory = $true)][string]$Label
        )
        $ObservedRows = Get-CsvRowCount -Path $Path
        if ($ObservedRows -ne $ExpectedRows) {
            throw "${Label} row count mismatch. Expected ${ExpectedRows}, found ${ObservedRows}: ${Path}"
        }
    }

    function Get-ShortFileHash {
        param(
            [Parameter(Mandatory = $true)][string]$Path,
            [int]$Length = 16
        )
        $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Length -lt 8 -or $Length -gt $Hash.Length) {
            throw "Invalid short-hash length: $Length"
        }
        return $Hash.Substring(0, $Length)
    }

    function Test-TorchRuntime {
        $TorchRuntimeProbe = @'
import sys
try:
    import torch
except Exception as exc:
    print(f"TORCH_RUNTIME_UNAVAILABLE={type(exc).__name__}: {exc}")
    raise SystemExit(3)

ok = torch.__version__.split("+")[0] == "2.11.0" and torch.version.cuda == "12.8"
print(f"TORCH_VERSION={torch.__version__}")
print(f"TORCH_CUDA={torch.version.cuda}")
raise SystemExit(0 if ok else 4)
'@
        $TorchProbePath = Write-InlinePythonScript -Name "torch_runtime_probe" -Code $TorchRuntimeProbe
        $TorchLines = @(& $PythonExe $TorchProbePath)
        $TorchExitCode = $LASTEXITCODE
        foreach ($TorchLine in $TorchLines) {
            Write-Host $TorchLine
        }
        return ($TorchExitCode -eq 0)
    }

    function Test-CompletedLocalScore {
        param(
            [Parameter(Mandatory = $true)][string]$InputPath,
            [Parameter(Mandatory = $true)][string]$OutputPath,
            [Parameter(Mandatory = $true)][string]$CheckpointDirectory,
            [Parameter(Mandatory = $true)][int]$ExpectedRows
        )
        Invoke-Python -Arguments @(
            "-m", "evalforge", "validate-local-score",
            "--input", $InputPath,
            "--output", $OutputPath,
            "--model-id", $JudgeModel,
            "--precision", $JudgePrecision,
            "--max-input-tokens", $MaxInputTokens,
            "--chunk-size", $ScoringChunkSize,
            "--checkpoint-dir", $CheckpointDirectory
        ) -Description "Validate completed local-judge score"
        Assert-CsvRowCount -Path $OutputPath -ExpectedRows $ExpectedRows -Label "completed local-score output"
    }

    function Invoke-CheckpointedJudgeScoring {
        param(
            [Parameter(Mandatory = $true)][string]$InputPath,
            [Parameter(Mandatory = $true)][string]$OutputPath,
            [Parameter(Mandatory = $true)][string]$CheckpointDirectory,
            [Parameter(Mandatory = $true)][int]$ExpectedRows,
            [Parameter(Mandatory = $true)][string]$Label
        )

        Assert-CsvRowCount -Path $InputPath -ExpectedRows $ExpectedRows -Label "$Label input"
        if (Test-Path -LiteralPath $OutputPath) {
            Write-Host "Validating completed scored artifact before reuse: $OutputPath"
            Test-CompletedLocalScore `
                -InputPath $InputPath `
                -OutputPath $OutputPath `
                -CheckpointDirectory $CheckpointDirectory `
                -ExpectedRows $ExpectedRows
            return
        }

        $ScoringArguments = @(
            "-m", "evalforge", "score-local-judge",
            "--input", $InputPath,
            "--output", $OutputPath,
            "--model-id", $JudgeModel,
            "--precision", $JudgePrecision,
            "--max-input-tokens", $MaxInputTokens,
            "--chunk-size", $ScoringChunkSize,
            "--checkpoint-dir", $CheckpointDirectory
        )
        $CheckpointManifest = Join-Path $CheckpointDirectory "scoring_checkpoint_manifest.json"
        if (Test-Path -LiteralPath $CheckpointManifest) {
            $ScoringArguments += "--resume"
            Write-Host "Resuming from checkpoint manifest: $CheckpointManifest"
        }
        Invoke-Checked $Label {
            Invoke-Python -Arguments $ScoringArguments -Description $Label
        }
        Test-CompletedLocalScore `
            -InputPath $InputPath `
            -OutputPath $OutputPath `
            -CheckpointDirectory $CheckpointDirectory `
            -ExpectedRows $ExpectedRows
    }

    function Test-CompletedHeldoutEvaluation {
        param(
            [Parameter(Mandatory = $true)][string]$BundlePath,
            [Parameter(Mandatory = $true)][string]$InputPath,
            [Parameter(Mandatory = $true)][string]$OutputDirectory,
            [Parameter(Mandatory = $true)][string]$ProvenanceManifest
        )
        Invoke-Python -Arguments @(
            "-m", "evalforge", "validate-heldout",
            "--bundle", $BundlePath,
            "--input", $InputPath,
            "--output", $OutputDirectory,
            "--source-type", "external_benchmark",
            "--provenance-verified",
            "--provenance-manifest", $ProvenanceManifest
        ) -Description "Validate completed JudgeBench held-out evaluation"
    }

    Assert-Path -Path $RepoRoot -Label "Repository root"
    Assert-Path -Path $FinQARoot -Label "FinQA root"
    Assert-Path -Path $JudgeBenchRoot -Label "JudgeBench root"
    Assert-Path -Path (Join-Path $RepoRoot "pyproject.toml") -Label "pyproject.toml"
    Assert-Path -Path (Join-Path $RepoRoot "src\evalforge\local_judge.py") -Label "AnchorEval v1.5 source"

    if ($null -eq (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Windows Python Launcher 'py' was not found. Install Python 3.11 and make py.exe available."
    }

    $DriveRoot = [System.IO.Path]::GetPathRoot($RepoRoot)
    $DriveName = $DriveRoot.TrimEnd("\").TrimEnd(":")
    $Drive = Get-PSDrive -Name $DriveName
    if ($Drive.Free -lt 30GB) {
        throw "Less than 30 GB free on $DriveRoot. Free space before downloading the Qwen model."
    }

    New-Item -ItemType Directory -Force -Path `
        $ExecutionRoot, `
        $PreparedDataRoot, `
        $ConfigRoot, `
        $CheckpointRoot, `
        $LogRoot, `
        $InlinePythonRoot, `
        $env:MPLCONFIGDIR, `
        $HFRoot, `
        $env:HF_HUB_CACHE, `
        $env:HF_DATASETS_CACHE, `
        $env:TRANSFORMERS_CACHE | Out-Null

    $TranscriptPath = Join-Path $LogRoot "anchor_eval_$RunId.transcript.txt"
    $TranscriptStarted = $false

    try {
        try {
            Start-Transcript -LiteralPath $TranscriptPath -Append | Out-Null
            $TranscriptStarted = $true
        }
        catch {
            Write-Warning "Could not start transcript. Execution will continue without it: $($_.Exception.Message)"
        }

        Set-Location -LiteralPath $RepoRoot

        if (-not (Test-Path -LiteralPath $PythonExe)) {
            Invoke-Checked "Create Python 3.11 virtual environment" {
                Invoke-NativeChecked -Executable "py" -Arguments @("-3.11", "-m", "venv", $VenvPath) -Description "Create Python 3.11 virtual environment"
            }
        }

        Invoke-Checked "Verify project Python version" {
            $VersionCode = @'
import sys
assert sys.version_info[:2] == (3, 11), sys.version
print("PYTHON=", sys.version)
print("EXECUTABLE=", sys.executable)
'@
            Invoke-InlinePythonScript -Name "verify_python_version" -Code $VersionCode -Description "Verify Python 3.11"
        }

        Invoke-Checked "Install reproducible core and GPU dependencies" {
            Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -Description "Upgrade pip build tooling"
            if (-not (Test-TorchRuntime)) {
                Write-Host "Installing required torch==2.11.0 CUDA 12.8 wheel."
                Invoke-Python -Arguments @(
                    "-m", "pip", "install", "--upgrade", "--force-reinstall",
                    "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu128"
                ) -Description "Install CUDA 12.8 PyTorch"
            }
            else {
                Write-Host "Reusing verified torch==2.11.0 CUDA 12.8 wheel."
            }
            Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "--prefer-binary", "-e", ".[dev,gpu]") -Description "Install AnchorEval dependencies"
            Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "transformers>=4.51,<6") -Description "Install compatible Transformers"
            Invoke-Python -Arguments @("-m", "pip", "check") -Description "Verify dependency consistency"
        }

        Invoke-Checked "Confirm installed AnchorEval v1.5 source origin" {
            $OriginCode = @'
import evalforge
from pathlib import Path
expected = "0.15.0"
assert evalforge.__version__ == expected, evalforge.__version__
print("EVALFORGE_VERSION=", evalforge.__version__)
print("EVALFORGE_MODULE=", Path(evalforge.__file__).resolve())
'@
            Invoke-InlinePythonScript -Name "verify_evalforge_origin" -Code $OriginCode -Description "Verify installed AnchorEval version"
            Invoke-Python -Arguments @("-m", "evalforge", "doctor", "--json") -Description "Print runtime origin diagnostics"
        }

        Invoke-Checked "Verify CUDA 12.8, RTX GPU, and BF16 scorer dependencies" {
            $CudaCode = @'
import torch
import transformers
import accelerate
from packaging.version import Version

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch."
assert torch.version.cuda == "12.8", torch.version.cuda
assert Version(transformers.__version__) >= Version("4.51.0"), transformers.__version__
props = torch.cuda.get_device_properties(0)
free_memory, total_memory = torch.cuda.mem_get_info(0)
print("TORCH_VERSION=", torch.__version__)
print("TORCH_CUDA=", torch.version.cuda)
print("GPU_NAME=", torch.cuda.get_device_name(0))
print("GPU_MEMORY_GB=", round(props.total_memory / (1024 ** 3), 2))
print("GPU_FREE_MEMORY_GB=", round(free_memory / (1024 ** 3), 2))
assert free_memory >= 12 * 1024 ** 3, f"At least 12 GiB free GPU memory is required for BF16 Qwen scoring; found {free_memory / (1024 ** 3):.2f} GiB."
assert torch.cuda.is_bf16_supported(), "The Windows default scorer requires BF16 support."
print("BF16_SUPPORTED=", torch.cuda.is_bf16_supported())
print("TRANSFORMERS=", transformers.__version__)
print("ACCELERATE=", accelerate.__version__)
'@
            Invoke-InlinePythonScript -Name "verify_bf16_scorer_runtime" -Code $CudaCode -Description "Validate CUDA and BF16 scorer dependencies"
        }

        Invoke-Checked "Probe Qwen BF16 model load in isolated child process" {
            Invoke-Python -Arguments @(
                "-m", "evalforge", "probe-local-judge",
                "--model-id", $JudgeModel,
                "--precision", $JudgePrecision,
                "--max-input-tokens", $MaxInputTokens
            ) -Description "Probe Qwen BF16 scorer load"
        }

        $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($null -ne $NvidiaSmi) {
            Write-Host ""
            Write-Host "GPU status from nvidia-smi:"
            Invoke-NativeChecked -Executable $NvidiaSmi.Source -Arguments @(
                "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"
            ) -Description "Read nvidia-smi GPU status"
        }

        Invoke-Checked "Ruff lint" { Invoke-Python -Arguments @("-m", "ruff", "check", ".") -Description "Ruff lint" }
        Invoke-Checked "Ruff format check" { Invoke-Python -Arguments @("-m", "ruff", "format", "--check", ".") -Description "Ruff format check" }
        Invoke-Checked "Compile source and tests" { Invoke-Python -Arguments @("-m", "compileall", "-q", "src", "tests") -Description "Compile source and tests" }
        Invoke-Checked "Run unit and integration tests" {
            Invoke-Python -Arguments @("-m", "pytest", "--junitxml", (Join-Path $ExecutionRoot "junit.xml")) -Description "Run pytest"
        }
        Invoke-Checked "Validate GitHub Actions workflow contract" {
            Invoke-Python -Arguments @("scripts\validate_ci_contract.py") -Description "Validate CI workflow contract"
        }
        Invoke-Checked "Run canonical CPU smoke pipeline" {
            Invoke-Python -Arguments @("-m", "evalforge", "run", "--config", "configs\smoke.yaml") -Description "Run CPU smoke pipeline"
        }
        Invoke-Checked "Run five-seed controlled validation" {
            Invoke-Python -Arguments @(
                "-m", "evalforge", "controlled-validation", "--config", "configs\smoke.yaml",
                "--seeds", "1729,1730,1731,1732,1733", "--groups-per-domain", "24",
                "--output", (Join-Path $ExecutionRoot "controlled_validation")
            ) -Description "Run controlled validation"
        }

        foreach ($GeneratedDir in @("build", "dist", ".distcheck")) {
            $FullPath = Join-Path $RepoRoot $GeneratedDir
            if (Test-Path -LiteralPath $FullPath) {
                Remove-Item -LiteralPath $FullPath -Recurse -Force
            }
        }
        Invoke-Checked "Build wheel and source distribution" {
            Invoke-Python -Arguments @("-m", "build", "--sdist", "--wheel", "--outdir", "dist") -Description "Build wheel and sdist"
            Invoke-Python -Arguments @("-m", "twine", "check", "dist\*") -Description "Validate package metadata"
        }
        Invoke-Checked "Validate clean wheel installation outside source tree" {
            Invoke-Python -Arguments @(
                "scripts\verify_distribution_install.py", "--dist-dir", "dist", "--distribution", "wheel",
                "--packaged-smoke-config", "--work-dir", (Join-Path $ExecutionRoot "wheel_distribution_validation")
            ) -Description "Validate wheel installation"
        }
        Invoke-Checked "Validate clean source-distribution installation outside source tree" {
            Invoke-Python -Arguments @(
                "scripts\verify_distribution_install.py", "--dist-dir", "dist", "--distribution", "sdist",
                "--packaged-smoke-config", "--work-dir", (Join-Path $ExecutionRoot "sdist_distribution_validation")
            ) -Description "Validate sdist installation"
        }

        $FinQACandidates = @(
            Get-ChildItem -LiteralPath $FinQARoot -Recurse -File -Filter "train.json" |
                Where-Object { $_.FullName -match "\\dataset\\train\.json$" }
        )
        if ($FinQACandidates.Count -eq 0) {
            $FinQACandidates = @(Get-ChildItem -LiteralPath $FinQARoot -Recurse -File -Filter "train.json")
        }
        if ($FinQACandidates.Count -ne 1) {
            $CandidateList = ($FinQACandidates | ForEach-Object FullName) -join [Environment]::NewLine
            throw "Expected exactly one FinQA train.json. Found $($FinQACandidates.Count):`n$CandidateList"
        }

        $FinQATrainJson = $FinQACandidates[0].FullName
        $FinQAUnscored = Join-Path $PreparedDataRoot "finqa_derived_unscored.csv"
        Invoke-Checked "Build FinQA verifier-backed preference pairs" {
            Invoke-Python -Arguments @(
                "-m", "evalforge", "build-finqa-pairs", "--input", $FinQATrainJson,
                "--output", $FinQAUnscored, "--max-examples", $FinQAMaxExamples, "--seed", $Seed
            ) -Description "Build FinQA preference pairs"
        }

        $FinQARowCount = Get-CsvRowCount -Path $FinQAUnscored
        if ($FinQARowCount -lt 100) {
            throw "FinQA pair builder produced fewer than 100 valid rows: $FinQARowCount"
        }

        $FinQAInputHash = Get-ShortFileHash -Path $FinQAUnscored
        $FinQAProbeInput = Join-Path $PreparedDataRoot "finqa_gpu_probe_${FinQAInputHash}_unscored.csv"
        $FinQAProbeScored = Join-Path $PreparedDataRoot "finqa_gpu_probe_${FinQAInputHash}_local_scored.csv"
        $FinQAScored = Join-Path $PreparedDataRoot "finqa_derived_${FinQAInputHash}_local_scored.csv"
        $FinQAProbeCheckpoints = Join-Path $CheckpointRoot "finqa_probe_$FinQAInputHash"
        $FinQACheckpoints = Join-Path $CheckpointRoot "finqa_$FinQAInputHash"

        Invoke-Checked "Create eight-row Qwen GPU probe input" {
            $ProbeCode = @'
import pandas as pd
import sys
source, destination = sys.argv[1], sys.argv[2]
frame = pd.read_csv(source).head(8)
if len(frame) != 8:
    raise RuntimeError(f"Expected 8 rows; found {len(frame)}.")
frame.to_csv(destination, index=False)
print(destination)
'@
            Invoke-InlinePythonScript -Name "create_finqa_probe" -Code $ProbeCode -ScriptArguments @($FinQAUnscored, $FinQAProbeInput) -Description "Create eight-row Qwen probe"
        }

        Invoke-CheckpointedJudgeScoring `
            -InputPath $FinQAProbeInput `
            -OutputPath $FinQAProbeScored `
            -CheckpointDirectory $FinQAProbeCheckpoints `
            -ExpectedRows 8 `
            -Label "Checkpointed eight-row Qwen GPU probe"
        Invoke-CheckpointedJudgeScoring `
            -InputPath $FinQAUnscored `
            -OutputPath $FinQAScored `
            -CheckpointDirectory $FinQACheckpoints `
            -ExpectedRows $FinQARowCount `
            -Label "Checkpointed GPU score of FinQA derived corpus"

        $FinQAConfigPath = Join-Path $ConfigRoot "finqa_local_judge_${FinQAInputHash}.yaml"
        $FinQAScoredYamlPath = $FinQAScored.Replace("\", "/")
        $FinQARunOutputRoot = (Join-Path $ExecutionRoot "finqa_evidence_runs").Replace("\", "/")
        $FinQAConfig = @"
run:
  run_name: finqa_local_judge_$FinQAInputHash
  seed: $Seed
  output_root: "$FinQARunOutputRoot"
data:
  fixture_path: "$FinQAScoredYamlPath"
  source_type: public_derived_controlled
  provenance_verified: false
  train_split: train
  calibration_split: calibration
  policy_split: policy
  dpo_split: dpo
  final_split: final
anchors:
  static_budget: 16
  adaptive_budget: 24
routing:
  auto_accept_min_p_correct: 0.90
  max_wrong_auto_label_rate: 0.05
  max_escalation_rate: 0.20
  max_order_delta: 0.20
  max_style_delta: 0.15
  max_order_flip_rate: 0.10
  max_style_flip_rate: 0.10
  max_ece: 0.10
  min_auto_accept_coverage: 0.60
report:
  save_svg: true
  save_html: true
"@
        Set-Content -LiteralPath $FinQAConfigPath -Value $FinQAConfig -Encoding UTF8

        $FinQARunOutput = @(
            Invoke-Python -Arguments @("-m", "evalforge", "run", "--config", $FinQAConfigPath) -Description "Fit FinQA C0-C5 evaluator portfolio"
        )
        if ($FinQARunOutput.Count -eq 0) {
            throw "FinQA pipeline did not print a published evidence path."
        }
        $FinQARunPath = [string]($FinQARunOutput | Select-Object -Last 1)
        if (-not [System.IO.Path]::IsPathRooted($FinQARunPath)) {
            $FinQARunPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $FinQARunPath))
        }
        Assert-Path -Path $FinQARunPath -Label "FinQA published evidence run"
        $CandidateBundle = Join-Path $FinQARunPath "artifacts\candidate_bundle.joblib"
        Assert-Path -Path $CandidateBundle -Label "FinQA candidate bundle"
        Assert-Path -Path ($CandidateBundle + ".manifest.json") -Label "FinQA candidate bundle manifest"

        $JudgeBenchRawJsonl = Join-Path $PreparedDataRoot "judgebench_final_raw.jsonl"
        $JudgeBenchManifest = Join-Path $PreparedDataRoot "judgebench_provenance.json"
        $JudgeBenchStagerPath = Join-Path $PreparedDataRoot "stage_judgebench.py"
        $JudgeBenchStager = @'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
output = Path(sys.argv[2])
required = {"pair_id", "original_id", "source", "question", "response_A", "response_B", "label"}
records_by_id: dict[str, dict] = {}

def normalize_records(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("records"), list):
            return payload["records"]
        return [payload]
    return payload if isinstance(payload, list) else []

def add_records(records, origin: Path):
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            continue
        pair_id = str(record["pair_id"]).strip()
        if not pair_id:
            raise RuntimeError(f"Blank pair_id in {origin}")
        label = str(record["label"]).replace(" ", "")
        if label not in {"A>B", "B>A"}:
            raise RuntimeError(f"Unsupported label {label!r} in {origin}")
        compact = {
            "pair_id": pair_id,
            "original_id": record.get("original_id"),
            "source": str(record["source"]),
            "question": str(record["question"]),
            "response_model": str(record.get("response_model", "unknown")),
            "response_A": str(record["response_A"]),
            "response_B": str(record["response_B"]),
            "label": label,
        }
        previous = records_by_id.get(pair_id)
        if previous is not None and previous != compact:
            raise RuntimeError(f"Conflicting duplicate pair_id {pair_id!r} in {origin}")
        records_by_id[pair_id] = compact

for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    try:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            add_records([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], path)
        elif suffix == ".json":
            add_records(normalize_records(json.loads(path.read_text(encoding="utf-8"))), path)
        elif suffix == ".parquet":
            add_records(pd.read_parquet(path).to_dict(orient="records"), path)
        elif suffix == ".arrow":
            from datasets import Dataset
            add_records(Dataset.from_file(str(path)).to_list(), path)
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect potential JudgeBench file: {path}") from exc

if len(records_by_id) != 620:
    raise RuntimeError(f"Expected exactly 620 unique JudgeBench response pairs; found {len(records_by_id)}.")

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for pair_id in sorted(records_by_id):
        handle.write(json.dumps(records_by_id[pair_id], ensure_ascii=False) + "\n")
print(output)
print("PAIR_COUNT=620")
'@
        Set-Content -LiteralPath $JudgeBenchStagerPath -Value $JudgeBenchStager -Encoding UTF8
        Invoke-Checked "Stage local JudgeBench artifact into canonical JSONL" {
            Invoke-Python -Arguments @($JudgeBenchStagerPath, $JudgeBenchRoot, $JudgeBenchRawJsonl) -Description "Stage JudgeBench"
        }

        $JudgeBenchRawHash = (Get-FileHash -LiteralPath $JudgeBenchRawJsonl -Algorithm SHA256).Hash.ToLowerInvariant()
        $JudgeBenchInputHash = $JudgeBenchRawHash.Substring(0, 16)
        $JudgeBenchUnscored = Join-Path $PreparedDataRoot "judgebench_final_${JudgeBenchInputHash}_unscored.csv"
        $JudgeBenchScored = Join-Path $PreparedDataRoot "judgebench_final_${JudgeBenchInputHash}_local_scored.csv"
        $JudgeBenchCheckpoints = Join-Path $CheckpointRoot "judgebench_$JudgeBenchInputHash"

        Invoke-Checked "Normalize JudgeBench into final-only EvalForge schema" {
            Invoke-Python -Arguments @(
                "-m", "evalforge", "normalize-judgebench", "--input", $JudgeBenchRawJsonl, "--output", $JudgeBenchUnscored
            ) -Description "Normalize JudgeBench"
        }
        Assert-CsvRowCount -Path $JudgeBenchUnscored -ExpectedRows 620 -Label "JudgeBench normalized input"
        Invoke-CheckpointedJudgeScoring `
            -InputPath $JudgeBenchUnscored `
            -OutputPath $JudgeBenchScored `
            -CheckpointDirectory $JudgeBenchCheckpoints `
            -ExpectedRows 620 `
            -Label "Checkpointed GPU score of immutable JudgeBench final set"

        $JudgeBenchRevision = "local_materialized_raw_sha256:$JudgeBenchRawHash"
        $GitCommand = Get-Command git -ErrorAction SilentlyContinue
        if ((Test-Path -LiteralPath (Join-Path $JudgeBenchRoot ".git")) -and ($null -ne $GitCommand)) {
            try {
                $CandidateRevisionLines = @(Invoke-NativeChecked -Executable $GitCommand.Source -Arguments @(
                    "-C", $JudgeBenchRoot, "rev-parse", "HEAD"
                ) -Description "Read JudgeBench git revision")
                $CandidateRevision = [string]($CandidateRevisionLines | Select-Object -Last 1)
                if ([string]::IsNullOrWhiteSpace($CandidateRevision)) {
                    throw "git rev-parse returned a blank revision."
                }
                $JudgeBenchRevision = $CandidateRevision.Trim()
            }
            catch {
                Write-Warning "Could not read JudgeBench git revision; binding provenance to raw JSONL SHA-256 instead: $($_.Exception.Message)"
            }
        }

        [ordered]@{
            dataset_fingerprint_sha256 = (Get-FileHash -LiteralPath $JudgeBenchScored -Algorithm SHA256).Hash.ToLowerInvariant()
            source_name               = "ScalerLab/JudgeBench"
            source_uri                = "https://huggingface.co/datasets/ScalerLab/JudgeBench"
            license                   = "MIT"
            revision_or_commit        = $JudgeBenchRevision
            retrieved_at_utc          = (Get-Date).ToUniversalTime().ToString("o")
            permitted_splits          = @("final")
            evidence_class            = "external_benchmark"
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $JudgeBenchManifest -Encoding UTF8

        $CandidateBundleShortHash = Get-ShortFileHash -Path $CandidateBundle -Length 12
        $JudgeBenchScoreShortHash = Get-ShortFileHash -Path $JudgeBenchScored -Length 12
        $JudgeBenchEvaluationRoot = Join-Path $RepoRoot "outputs\external_evaluation\judgebench_${RunId}_${CandidateBundleShortHash}_${JudgeBenchScoreShortHash}"
        $ComparisonCsv = Join-Path $JudgeBenchEvaluationRoot "heldout_candidate_comparison.csv"
        if (Test-Path -LiteralPath $ComparisonCsv) {
            Write-Host "Validating completed JudgeBench held-out evaluation before reuse: $ComparisonCsv"
            Test-CompletedHeldoutEvaluation `
                -BundlePath $CandidateBundle `
                -InputPath $JudgeBenchScored `
                -OutputDirectory $JudgeBenchEvaluationRoot `
                -ProvenanceManifest $JudgeBenchManifest
        }
        else {
            if ((Test-Path -LiteralPath $JudgeBenchEvaluationRoot) -and (Get-ChildItem -LiteralPath $JudgeBenchEvaluationRoot -Force | Select-Object -First 1)) {
                throw "JudgeBench held-out directory is partial or invalid: $JudgeBenchEvaluationRoot. Inspect and remove it before rerunning."
            }
            Invoke-Checked "Evaluate C0-C5 on immutable JudgeBench final-only data" {
                Invoke-Python -Arguments @(
                    "-m", "evalforge", "evaluate-heldout", "--bundle", $CandidateBundle,
                    "--input", $JudgeBenchScored, "--output", $JudgeBenchEvaluationRoot,
                    "--source-type", "external_benchmark", "--provenance-verified",
                    "--provenance-manifest", $JudgeBenchManifest, "--trust-candidate-bundle"
                ) -Description "Evaluate JudgeBench held-out candidates"
            }
            Test-CompletedHeldoutEvaluation `
                -BundlePath $CandidateBundle `
                -InputPath $JudgeBenchScored `
                -OutputDirectory $JudgeBenchEvaluationRoot `
                -ProvenanceManifest $JudgeBenchManifest
        }

        Assert-Path -Path $ComparisonCsv -Label "JudgeBench candidate comparison"
        Write-Host ""
        Write-Host "========================================================================"
        Write-Host "ANCHOREVAL EXECUTION COMPLETED"
        Write-Host "========================================================================"
        Write-Host "Transcript:               $TranscriptPath"
        Write-Host "FinQA local evidence run: $FinQARunPath"
        Write-Host "JudgeBench evaluation:    $JudgeBenchEvaluationRoot"
        Write-Host "JudgeBench comparison:    $ComparisonCsv"
        Write-Host ""
        Import-Csv -LiteralPath $ComparisonCsv |
            Select-Object candidate_id, pairwise_agreement, macro_domain_pairwise_agreement, `
                brier_correctness, ece_correctness, worst_slice_agreement, `
                auto_accept_coverage, wrong_auto_label_rate |
            Format-Table -AutoSize

        Write-Host ""
        Write-Host "Claim boundary:"
        Write-Host "  - FinQA-derived results are controlled development evidence."
        Write-Host "  - JudgeBench is external held-out evaluator evidence."
        Write-Host "  - This script does not create a DPO-transfer or production-readiness claim."
    }
    finally {
        if ($TranscriptStarted) {
            Stop-Transcript | Out-Null
        }
    }
}
