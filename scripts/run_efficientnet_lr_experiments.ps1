[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$Experiments = @(
    @{
        Config = "configs/isic2018_7class_efficientnet_b0_320.yaml"
        Run = "isic2018_7class_efficientnet_b0_320"
    },
    @{
        Config = "configs/isic2018_efficientnet_lr1_low_body.yaml"
        Run = "isic2018_effnet_lr1_low_body"
    },
    @{
        Config = "configs/isic2018_efficientnet_lr2_recommended.yaml"
        Run = "isic2018_effnet_lr2_recommended"
    },
    @{
        Config = "configs/isic2018_efficientnet_lr3_very_low_body.yaml"
        Run = "isic2018_effnet_lr3_very_low_body"
    },
    @{
        Config = "configs/isic2018_efficientnet_lr4_higher_body.yaml"
        Run = "isic2018_effnet_lr4_higher_body"
    }
)

Push-Location $ProjectRoot
try {
    foreach ($Experiment in $Experiments) {
        $Summary = "outputs/checkpoints/$($Experiment.Run)/summary.json"
        $Checkpoint = "outputs/checkpoints/$($Experiment.Run).pt"
        if ($Force -or -not ((Test-Path -LiteralPath $Summary) -and (Test-Path -LiteralPath $Checkpoint))) {
            Write-Host "Training $($Experiment.Run)..."
            & $Python -u -m skin_project.training --config $Experiment.Config
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed for $($Experiment.Run) with exit code $LASTEXITCODE"
            }
        }
        else {
            Write-Host "Reusing completed training: $($Experiment.Run)"
        }

        $RobustnessDirectory = "outputs/robustness/$($Experiment.Run)/dev"
        $RobustnessSummary = "$RobustnessDirectory/brightness_summary.json"
        if ($Force -or -not (Test-Path -LiteralPath $RobustnessSummary)) {
            Write-Host "Evaluating brightness robustness for $($Experiment.Run)..."
            & $Python -u -m skin_project.robustness `
                --config $Experiment.Config `
                --checkpoint $Checkpoint `
                --output-dir $RobustnessDirectory
            if ($LASTEXITCODE -ne 0) {
                throw "Brightness evaluation failed for $($Experiment.Run) with exit code $LASTEXITCODE"
            }
        }
        else {
            Write-Host "Reusing completed brightness evaluation: $($Experiment.Run)"
        }
    }

    & $Python scripts/compare_efficientnet_lr_results.py
    if ($LASTEXITCODE -ne 0) {
        throw "Learning-rate comparison failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
