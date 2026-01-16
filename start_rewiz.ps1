param(
  [int]$Port = 8501,
  [switch]$Headless
)

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $here '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPy)) {
  throw "Missing venv python at: $venvPy. Create venv first."
}

$rewiz = Join-Path $here 'ReWiz.py'

$args = @('-m','streamlit','run', $rewiz, '--server.port', $Port)
if ($Headless) {
  $args += @('--server.headless','true')
}

Write-Host "Starting Retriever Wizard with: $venvPy $($args -join ' ')"
& $venvPy @args
