Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\baseline.py" @args
    $trainerExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $trainerExitCode
