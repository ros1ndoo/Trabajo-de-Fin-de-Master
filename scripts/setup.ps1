param([switch]$Recreate)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$venvDirectory = Join-Path $projectDirectory '.venv'
$venvPython = Join-Path $venvDirectory 'Scripts/python.exe'

if ($Recreate -and (Test-Path -LiteralPath $venvDirectory)) {
    $resolvedProject = [System.IO.Path]::GetFullPath($projectDirectory)
    $resolvedVenv = [System.IO.Path]::GetFullPath($venvDirectory)
    if (-not $resolvedVenv.StartsWith($resolvedProject + [System.IO.Path]::DirectorySeparatorChar)) {
        throw 'La ruta del entorno virtual no pertenece al proyecto.'
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv $venvDirectory
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno virtual.' }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'No se pudo actualizar pip.' }
& $venvPython -m pip install --require-hashes --requirement requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias bloqueadas.' }
& $venvPython -m pip install --no-deps --editable .
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar el proyecto.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'El entorno contiene dependencias incompatibles.' }
Write-Output 'Entorno reproducible preparado. Ejecute scripts/start.ps1 para iniciar la aplicación.'
