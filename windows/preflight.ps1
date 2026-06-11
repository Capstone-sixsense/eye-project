param(
    [Parameter(Mandatory = $true)]
    [string]$Root
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    Write-Host "ERROR: $Message"
    exit 1
}

function Clean-Scalar([string]$Value) {
    $v = ($Value -replace "\s+#.*$", "").Trim()
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
        return $v.Substring(1, $v.Length - 2)
    }
    return $v
}

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "$Label not found: $Path"
    }
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -le 0) {
        Fail "$Label is empty: $Path"
    }
    return $item
}

function Require-JsonKeys([string]$Path, [string[]]$Keys, [string]$Label) {
    try {
        $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Fail "$Label is not valid JSON: $Path"
    }
    $names = @($data.PSObject.Properties.Name)
    $metricNames = @()
    if ($names -contains "metrics") {
        $metricNames = @($data.metrics.PSObject.Properties.Name)
    }
    foreach ($key in $Keys) {
        if (($names -notcontains $key) -and ($metricNames -notcontains $key)) {
            Fail "$Label missing key '$key': $Path"
        }
    }
}

$repoRoot = (Resolve-Path -LiteralPath $Root).Path
$aiRoot = Join-Path $repoRoot "ai"
$configPath = Join-Path $aiRoot "configs\base.yaml"
Require-File $configPath "base config" | Out-Null

$section = ""
$project = @{}
$infer = @{}

foreach ($line in Get-Content -LiteralPath $configPath -Encoding UTF8) {
    if ($line -match "^([A-Za-z_][A-Za-z0-9_-]*):\s*$") {
        $section = $Matches[1]
        continue
    }
    if ($line -match "^\s{2}([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*$") {
        $key = $Matches[1]
        $value = Clean-Scalar $Matches[2]
        if ($section -eq "project") {
            $project[$key] = $value
        } elseif ($section -eq "infer") {
            $infer[$key] = $value
        }
    }
}

$version = $project["version"]
if ([string]::IsNullOrWhiteSpace($version)) {
    Fail "project.version not found in $configPath"
}

$checkpointRel = $infer["checkpoint_path"]
if ([string]::IsNullOrWhiteSpace($checkpointRel)) {
    Fail "infer.checkpoint_path not found in $configPath"
}

if ([System.IO.Path]::IsPathRooted($checkpointRel)) {
    $checkpointPath = $checkpointRel
} else {
    $checkpointPath = Join-Path $aiRoot $checkpointRel
}

$checkpoint = Require-File $checkpointPath "deployment checkpoint"
$quickqual = Require-File (Join-Path $repoRoot "backend\models\quickqual_dn121_512.pkl") "QuickQual model"
$hfBlob = Require-File (Join-Path $repoRoot "backend\.cache\huggingface\hub\models--timm--densenet121.tv_in1k\blobs\c894c6d9caa317a8ca1942986dee7a16a86c77734a4d691d2abe05389cfef358") "DenseNet121 HF cache blob"
$hfRef = Require-File (Join-Path $repoRoot "backend\.cache\huggingface\hub\models--timm--densenet121.tv_in1k\refs\main") "DenseNet121 HF cache ref"

$evalDir = Join-Path $aiRoot "artifacts\evaluations"
$classificationMetrics = Join-Path $evalDir "external_test_${version}_best_metrics.json"
Require-File $classificationMetrics "classification metrics" | Out-Null
Require-JsonKeys $classificationMetrics @("metrics", "metrics_at_optimal_threshold", "optimal_threshold") "classification metrics"

$evidenceType = $infer["evidence_type"]
if ([string]::IsNullOrWhiteSpace($evidenceType)) {
    $evidenceType = "cam_research"
}
$xaiSplit = $infer["xai_eval_split"]
if ([string]::IsNullOrWhiteSpace($xaiSplit)) {
    $xaiSplit = "test"
}

if ($evidenceType -in @("lesion_segmentation", "lesion_evidence", "segmentation")) {
    $xaiCandidates = @(
        (Join-Path $evalDir "xai_${version}_lesion_segmentation_${xaiSplit}_best_metrics.json"),
        (Join-Path $evalDir "xai_${version}_segmentation_${xaiSplit}_best_metrics.json")
    )
    $xaiPath = $xaiCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $xaiPath) {
        Fail "XAI metrics not found. Expected one of: $($xaiCandidates -join ', ')"
    }
    Require-JsonKeys $xaiPath @("xai_seg_union_iou", "xai_seg_mdice", "xai_auc_iou") "XAI metrics"
}

Write-Host "  Active model version: $version"
Write-Host "  Checkpoint: $($checkpoint.FullName) ($([math]::Round($checkpoint.Length / 1MB, 1)) MB)"
Write-Host "  Classification metrics: $classificationMetrics"
if ($xaiPath) {
    Write-Host "  XAI metrics: $xaiPath"
}
Write-Host "  QuickQual model: $($quickqual.FullName)"
Write-Host "  HF cache: $($hfBlob.FullName), $($hfRef.FullName)"
Write-Host "  Preflight OK"
