# Opens as its own window: creates the stack if it does not exist yet, then grows it block by block.
# Closing the window stops training; the stage in progress is saved and this script picks it up again.
param([string]$Name = 'ts-w128', [int]$Width = 128, [int]$TrainBytes = 8388608, [int]$Blocks = 12,
      [double]$Hours = 8, [string[]]$Extra = @(), [int]$ExtraRepeat = 1)
$host.UI.RawUI.WindowTitle = "Eden Willow - stack trainer ($Name)"
Set-Location $PSScriptRoot
# Keep the computer and screen awake while this window is training.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483651) | Out-Null
$stack = Join-Path $PSScriptRoot "stacks\$Name"
if (-not (Test-Path "$stack\stack.json")) {
    $create = @('new', '--stack', $stack, '--width', $Width, '--train-bytes', $TrainBytes, '--extra-repeat', $ExtraRepeat)
    foreach ($item in $Extra) { $create += @('--extra', $item) }
    & "$PSScriptRoot\stacked.ps1" @create
}
& "$PSScriptRoot\stacked.ps1" grow --stack $stack --blocks $Blocks --hours $Hours
& "$PSScriptRoot\stacked.ps1" info --stack $stack
[Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
Write-Host 'Training window finished. Run this script again to continue the same stack.' -ForegroundColor Green
