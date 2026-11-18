Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\pythonw.exe" -ArgumentList (@('"' + "$PSScriptRoot\monitor.py" + '"') + $args) -WorkingDirectory $PSScriptRoot -WindowStyle Normal
