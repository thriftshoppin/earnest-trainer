Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\stacked.py" @args
    $trainerExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $trainerExitCode
