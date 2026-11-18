# Waits (in this visible window) until a stack reaches a number of frozen blocks, then swaps the data:
# stops that stack's run the same way the Earnest Stop button does, forks the frozen blocks onto a
# bigger TinyStories slice, and reopens the Earnest window on the fork. Close this window to cancel.
param([string]$Source = 'willow-tinystories-w128-32mb',
      [string]$Target = 'willow-tinystories-w128-64mb',
      [int]$AtBlocks = 3,
      [int]$TrainBytes = 67108864,
      [int]$Width = 128,
      [int]$Blocks = 12,
      [double]$Hours = 24)
$ErrorActionPreference = 'Stop'
$host.UI.RawUI.WindowTitle = "Eden Willow - data swap watcher ($Source -> $Target at $AtBlocks blocks)"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$sourceDir = Join-Path $PSScriptRoot "stacks\$Source"
$meta = Join-Path $sourceDir 'stack.json'
if (Test-Path (Join-Path $PSScriptRoot "stacks\$Target")) { Write-Host "Target stack $Target already exists; nothing to do." -ForegroundColor Yellow; return }

Write-Host "Watching $Source. Will swap to $([math]::Round($TrainBytes/1MB)) MB of TinyStories once it has $AtBlocks frozen blocks." -ForegroundColor Cyan
while ($true) {
    $count = @((Get-Content $meta -Raw | ConvertFrom-Json).blocks).Count
    Write-Host ("{0}  frozen blocks: {1} of {2}" -f (Get-Date -Format T), $count, $AtBlocks)
    if ($count -ge $AtBlocks) { break }
    Start-Sleep 30
}

$name = [regex]::Escape($Source)
Write-Host "Block $AtBlocks is frozen. Stopping the $Source run (same as the Stop button)..." -ForegroundColor Cyan
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$grow = $procs | Where-Object { $_.CommandLine -match 'stacked\.py\s+grow' -and $_.CommandLine -match "stacks\\$name(\s|`"|$)" }
$grow | ForEach-Object { Write-Host "  stopping trainer pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
Start-Sleep 3
$mon = $procs | Where-Object { $_.CommandLine -match 'monitor\.py' -and $_.CommandLine -match "--stack\s+`"?$name(`"|\s|$)" }
$mon | ForEach-Object { Write-Host "  closing Earnest window pid $($_.ProcessId) (its log is saved)"; Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
Start-Sleep 3

Write-Host "Forking the frozen blocks onto the bigger dataset..." -ForegroundColor Cyan
& $py -u stacked.py fork --source "stacks\$Source" --stack "stacks\$Target" --train-bytes $TrainBytes
if ($LASTEXITCODE -ne 0) { Write-Host "Fork failed; the $Source stack is untouched and can be resumed." -ForegroundColor Red; return }

$o = Get-CimInstance Win32_OperatingSystem
Write-Host ("Free RAM before starting: {0:N2} GB" -f ($o.FreePhysicalMemory/1MB))
Write-Host "Opening the Earnest window on $Target (normal 6,000-step blocks, $Hours-hour budget; closing it stops training)..." -ForegroundColor Cyan
Start-Process -FilePath $py -ArgumentList @('monitor.py', '--stack', $Target, '--start', '--hours', $Hours,
    '--width', $Width, '--train-bytes', $TrainBytes, '--blocks', $Blocks) -WorkingDirectory $PSScriptRoot -WindowStyle Normal
Write-Host "Swap done at $(Get-Date -Format T). This window can be closed." -ForegroundColor Green
