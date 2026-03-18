param(
    [string]$ProjectRoot = "K:\Python\git\CVSigns",
    [string]$PythonVersion = "3.12",
    [string]$VenvName = ".venv-train",
    [int]$Epochs = 100,
    [int]$ImageSize = 640,
    [int]$BatchSize = 8,
    [int]$Workers = 6,
    [string]$RunName = "rtsd_yolo12n_640",
    [switch]$SkipInstalls
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-CommandPath {
    param([string]$CommandName)
    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Command not found: $CommandName"
    }
    return $cmd.Source
}

function Resolve-PythonLauncherVersion {
    param([string]$RequestedVersion)

    $pyList = & py -0p 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $pyList) {
        throw "Python launcher 'py' is unavailable or returned no interpreters."
    }

    $requestedPattern = "-V:$RequestedVersion"
    if ($pyList | Where-Object { $_ -match [regex]::Escape($requestedPattern) }) {
        return $RequestedVersion
    }

    $versions = @()
    foreach ($line in $pyList) {
        if ($line -match "-V:(\d+\.\d+)") {
            $versions += $Matches[1]
        }
    }

    $selected = $versions | Select-Object -Unique | Sort-Object {[version]$_} -Descending | Select-Object -First 1
    if (-not $selected) {
        throw "No usable Python versions were found via py -0p."
    }

    Write-Host "Requested Python $RequestedVersion not found, using Python $selected instead." -ForegroundColor Yellow
    return $selected
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$VenvPath = Join-Path $ProjectRoot $VenvName
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$TrainRoot = Join-Path $ProjectRoot "train"
$YoloRoot = Join-Path $TrainRoot "yolo_rtsd"
$LogsDir = Join-Path $TrainRoot "logs"
$ReportsDir = Join-Path $TrainRoot "reports\$RunName"
$RunsDir = Join-Path $ProjectRoot "runs\detect"
$TrainScriptPath = Join-Path $TrainRoot "run_yolo12_train.ps1"
$SummaryPath = Join-Path $YoloRoot "prepare_summary.json"
$ModelPath = Join-Path $ProjectRoot "yolo12n.pt"
$YoloExe = Join-Path $VenvPath "Scripts\yolo.exe"

Write-Step "Project root: $ProjectRoot"

if (-not (Test-Path (Join-Path $ProjectRoot "tools\convert_rtsd_to_yolo.py"))) {
    throw "Converter not found: tools\convert_rtsd_to_yolo.py"
}

if (-not (Test-Path (Join-Path $TrainRoot "train_anno.json"))) {
    throw "RTSD annotations not found in train\train_anno.json"
}

if (-not (Test-Path (Join-Path $TrainRoot "val_anno.json"))) {
    throw "RTSD annotations not found in train\val_anno.json"
}

if (-not (Test-Path (Join-Path $TrainRoot "rtsd-frames\rtsd-frames"))) {
    throw "RTSD images not found in train\rtsd-frames\rtsd-frames"
}

if ((Test-Path $VenvPath) -and -not (Test-Path $PythonExe)) {
    Write-Step "Removing incomplete virtual environment"
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    $PythonVersion = Resolve-PythonLauncherVersion -RequestedVersion $PythonVersion
    Write-Step "Creating virtual environment $VenvName with Python $PythonVersion"
    & py -$PythonVersion -m venv $VenvPath
}
else {
    Write-Step "Virtual environment already exists: $VenvPath"
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found in venv after creation: $PythonExe"
}

if (-not $SkipInstalls) {
    Write-Step "Upgrading pip"
    & $PythonExe -m pip install --upgrade pip

    Write-Step "Installing Ultralytics"
    & $PythonExe -m pip install --upgrade ultralytics

    Write-Step "Removing existing PyTorch packages"
    & $PythonExe -m pip uninstall -y torch torchvision torchaudio

    Write-Step "Installing CUDA-enabled PyTorch"
    & $PythonExe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}
else {
    Write-Step "Skipping installs by request"
}

if (-not (Test-Path $YoloExe)) {
    throw "Ultralytics CLI not found after setup: $YoloExe"
}

Write-Step "Checking CUDA availability"
& $PythonExe -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
& $PythonExe -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"

Write-Step "Checking YOLO12 weights load"
& $PythonExe -c "from ultralytics import YOLO; YOLO(r'$ModelPath'); print('YOLO12 OK:', r'$ModelPath')"

Write-Step "Converting RTSD dataset to YOLO format"
& $PythonExe (Join-Path $ProjectRoot "tools\convert_rtsd_to_yolo.py") --source $TrainRoot --output $YoloRoot --link-mode hardlink

Write-Step "Creating logs and reports directories"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null

$TrainCommand = @"
& '$YoloExe' detect train `
  model=$ModelPath `
  data=$(Join-Path $YoloRoot 'dataset.yaml') `
  epochs=$Epochs `
  imgsz=$ImageSize `
  batch=$BatchSize `
  device=0 `
  workers=$Workers `
  cache=disk `
  amp=True `
  patience=20 `
  cos_lr=True `
  project=$RunsDir `
  name=$RunName `
  2>&1 | Tee-Object $(Join-Path $LogsDir "$RunName.log")
"@

$TrainScript = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$ProjectRoot'
$TrainCommand
"@

Write-Step "Saving training command"
Set-Content -Path $TrainScriptPath -Value $TrainScript -Encoding UTF8

Write-Step "Saving training command as text"
Set-Content -Path (Join-Path $ReportsDir "train_command.txt") -Value $TrainScript -Encoding UTF8

Write-Step "Preparation complete"
Write-Host "YOLO dataset: $(Join-Path $YoloRoot 'dataset.yaml')" -ForegroundColor Green
Write-Host "Dataset summary: $SummaryPath" -ForegroundColor Green
Write-Host "Training script: $TrainScriptPath" -ForegroundColor Green
Write-Host "Run this later to start training:" -ForegroundColor Yellow
Write-Host "powershell -ExecutionPolicy Bypass -File `"$TrainScriptPath`"" -ForegroundColor Yellow
