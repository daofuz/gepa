param(
  [string]$Python313Dir = ""
)

$ErrorActionPreference = "Stop"

function Test-Python313Exe {
  param([string]$Exe)
  if (-not $Exe -or -not (Test-Path $Exe)) {
    return $false
  }

  $version = & $Exe --version 2>&1
  if ($LASTEXITCODE -ne 0) {
    return $false
  }
  return (($version | Out-String) -match "Python 3\.13\.")
}

if ($Python313Dir) {
  $python313 = (Resolve-Path $Python313Dir).Path
  $python313Exe = Join-Path $python313 "python.exe"
  if (-not (Test-Python313Exe $python313Exe)) {
    throw "The provided folder does not contain Python 3.13: $Python313Dir"
  }
} else {
  $python313Exe = (& py -3.13 -c "import sys; print(sys.executable)" 2>&1 | Select-Object -First 1).ToString().Trim()
  if ($LASTEXITCODE -ne 0 -or -not (Test-Python313Exe $python313Exe)) {
    Write-Host "Could not get Python 3.13 from the Python launcher."
    Write-Host "Python launcher sees:"
    & py -0p
    throw "Run this script with -Python313Dir if Python 3.13 is installed in a custom folder."
  }
  $python313 = (Resolve-Path (Split-Path -Parent $python313Exe)).Path
}

$python313Scripts = Join-Path $python313 "Scripts"

[Environment]::SetEnvironmentVariable("PY_PYTHON", "3.13", "User")

$userPathRaw = [Environment]::GetEnvironmentVariable("Path", "User")
if ([string]::IsNullOrWhiteSpace($userPathRaw)) {
  $entries = @()
} else {
  $entries = $userPathRaw -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

$python313Norm = $python313.TrimEnd("\")
$python313ScriptsNorm = $python313Scripts.TrimEnd("\")

$filtered = foreach ($entry in $entries) {
  $expanded = [Environment]::ExpandEnvironmentVariables($entry.Trim()).TrimEnd("\")
  if ($expanded -ieq $python313Norm -or $expanded -ieq $python313ScriptsNorm) {
    continue
  }
  $entry
}

$newEntries = @($python313, $python313Scripts) + $filtered
$newPath = ($newEntries | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique) -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

$currentProcessPath = @($python313, $python313Scripts, $env:Path) -join ";"
$env:Path = $currentProcessPath

Write-Host "Done. User default Python is now set to 3.13."
Write-Host "Python 3.13 executable: $python313Exe"
Write-Host "Python 3.13 directory:   $python313"
Write-Host ""
Write-Host "In this terminal:"
Write-Host "  python --version"
Write-Host "  where.exe python"
Write-Host ""
Write-Host "For all future terminals, close and reopen PowerShell."
