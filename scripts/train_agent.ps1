param(
    [ValidateSet("ppo", "dqn")]
    [string]$Algo = "ppo",
    [ValidateSet("nes", "modern", "custom")]
    [string]$Env = "nes",
    [string]$EnvId = "",
    [int]$TotalSteps = 1500000,
    [string]$LogDir = "",
    [string]$ResumeFrom = "",
    [string]$Device = "",
    [int]$Seed = 123,
    [int]$NumEnvs = 8,
    [int]$PpoNSteps = 4096,
    [int]$PpoMinibatch = 2048,
    [int]$DqnBuffer = 200000,
    [int]$DqnBatch = 256,
    [string[]]$RewardWeight = @(),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if (-not $LogDir -or $LogDir -eq "") {
    $LogDir = "runs/{0}_{1}" -f $Algo, $Env
}

switch ($Algo) {
    "ppo" {
        $module = "tetris_v2.agents.ppo.train"
        $baseArgs = @(
            "-m", $module,
            "--env", $Env,
            "--total-timesteps", $TotalSteps,
            "--n-steps", $PpoNSteps,
            "--num-envs", $NumEnvs,
            "--minibatch-size", $PpoMinibatch,
            "--log-dir", $LogDir,
            "--seed", $Seed
        )
    }
    "dqn" {
        $module = "tetris_v2.agents.dqn.train"
        $baseArgs = @(
            "-m", $module,
            "--env", $Env,
            "--total-timesteps", $TotalSteps,
            "--buffer-size", $DqnBuffer,
            "--batch-size", $DqnBatch,
            "--num-envs", $NumEnvs,
            "--log-dir", $LogDir,
            "--seed", $Seed
        )
    }
    default {
        Write-Error "Unsupported --algo '$Algo'. Use 'ppo' or 'dqn'."
        exit 1
    }
}

if ($Env -eq "custom" -and [string]::IsNullOrEmpty($EnvId)) {
    Write-Error "--env-id is required when --env=custom."
    exit 1
}

if (-not [string]::IsNullOrEmpty($EnvId)) {
    $baseArgs += @("--env-id", $EnvId)
}
if (-not [string]::IsNullOrEmpty($ResumeFrom)) {
    $baseArgs += @("--resume-from", $ResumeFrom)
}
if (-not [string]::IsNullOrEmpty($Device)) {
    $baseArgs += @("--device", $Device)
}

foreach ($weight in $RewardWeight) {
    $baseArgs += @("--advanced-reward-weight", $weight)
}

if ($ExtraArgs) {
    $baseArgs += $ExtraArgs
}

Write-Host "Running: python $($baseArgs -join ' ')"
& python @baseArgs
