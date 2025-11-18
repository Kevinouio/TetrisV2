param(
    [ValidateSet("ppo", "dqn")]
    [string]$Algo = "ppo",
    [ValidateSet("nes", "modern", "custom")]
    [string]$Env = "nes",
    [string]$EnvId = "",
    [string]$Checkpoint = "",
    [int]$Episodes = 10,
    [int]$Seed = 321,
    [string]$Device = "",
    [switch]$Render,
    [string[]]$RewardWeight = @(),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

if (-not $Checkpoint -or $Checkpoint -eq "") {
    $Checkpoint = "runs/{0}_{1}/final_model.pt" -f $Algo, $Env
}

switch ($Algo) {
    "ppo" { $module = "tetris_v2.agents.ppo.eval" }
    "dqn" { $module = "tetris_v2.agents.dqn.eval" }
    default {
        Write-Error "Unsupported --algo '$Algo'. Use 'ppo' or 'dqn'."
        exit 1
    }
}

if ($Env -eq "custom" -and [string]::IsNullOrEmpty($EnvId)) {
    Write-Error "--env-id is required when --env=custom."
    exit 1
}

$argsList = @(
    "-m", $module,
    $Checkpoint,
    "--env", $Env,
    "--episodes", $Episodes,
    "--seed", $Seed
)

if (-not [string]::IsNullOrEmpty($EnvId)) {
    $argsList += @("--env-id", $EnvId)
}
if (-not [string]::IsNullOrEmpty($Device)) {
    $argsList += @("--device", $Device)
}
if ($Render.IsPresent) {
    $argsList += "--render"
}

foreach ($weight in $RewardWeight) {
    $argsList += @("--advanced-reward-weight", $weight)
}

if ($ExtraArgs) {
    $argsList += $ExtraArgs
}

Write-Host "Running: python $($argsList -join ' ')"
& python @argsList
