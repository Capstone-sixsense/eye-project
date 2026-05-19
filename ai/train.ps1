param(
    [Parameter(Mandatory = $true)]
    [string]$Config,

    [switch]$DryRun,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$check = "import sys, torch; print('training_python=' + sys.executable); print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"

py -3.14 -c $check
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$trainArgs = @("-3.14", "-m", "drscreen.cli.train", "--config", $Config)
if ($DryRun) {
    $trainArgs += "--dry-run"
}
if ($ExtraArgs) {
    $trainArgs += $ExtraArgs
}

py @trainArgs
exit $LASTEXITCODE
