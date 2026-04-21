# Run validate_render.py with GTK3 Runtime Cairo DLLs (default install path).
# Usage (from repo root):
#   .\scripts\validate_render_with_gtk.ps1
#   .\scripts\validate_render_with_gtk.ps1 -MaxSamples 500

param(
    [string]$Jsonl = "data/processed/train.jsonl",
    [int]$MaxSamples = 200,
    [int]$Seed = 42
)

$gtkBin = "C:\Program Files\GTK3-Runtime Win64\bin"
if (Test-Path $gtkBin) {
    $env:CAIROCFFI_DLL_DIRECTORIES = $gtkBin
    Write-Host "Using CAIROCFFI_DLL_DIRECTORIES=$gtkBin"
} else {
    Write-Warning "GTK3 runtime not found at $gtkBin — install it or set CAIROCFFI_DLL_DIRECTORIES manually."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python).Source
& $python (Join-Path $scriptDir "validate_render.py") --jsonl $Jsonl --max-samples $MaxSamples --seed $Seed
