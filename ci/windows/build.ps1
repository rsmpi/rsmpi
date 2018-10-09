$env:MSMPI_BIN = "C:\Program Files\Microsoft MPI\Bin\"
$env:MSMPI_INC = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
$env:MSMPI_LIB32 = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86\"
$env:MSMPI_LIB64 = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\"

$env:PATH += ";$HOME\.cargo\bin;$env:MSMPI_BIN"

cmd.exe /c "cargo build --no-default-features --all --examples 2>&1"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}