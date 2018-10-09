$env:MSMPI_BIN = "C:\Program Files\Microsoft MPI\Bin\"
$env:MSMPI_INC = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\"
$env:MSMPI_LIB32 = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86\"
$env:MSMPI_LIB64 = "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\"

$env:PATH += ";$HOME\.cargo\bin;$env:MSMPI_BIN"

$examples = `
    Get-ChildItem "examples" `
    | Where-Object Name -Like "*.rs" `
    | ForEach-Object BaseName

Write-Host "Running $($examples.Count) examples"
Write-Host ""

$num_ok = 0
$num_failed = 0
$result = "ok"

foreach ($example in $examples) {
    $failed = $false

    Write-Host "Example $example on 2...8 processes"
    $output_file = New-Item -Force "$HOME\AppData\Local\Temp\$($example)_output.txt"

    foreach ($num_proc in 2..8) {
        cmd.exe /c "cargo mpirun --no-default-features --verbose -n $num_proc --example $example 2>&1" > $output_file
        if ($LASTEXITCODE -eq 0) {
            Write-Host "." -NoNewline
            Remove-Item -Force $output_file
        } else {
            Write-Host " failed on $num_proc processes."
            Write-Host "output:"
            Get-Content $output_file | Write-Output
            Remove-Item -Force $output_file
            $num_failed++
            $failed = $true
            $result = "failed"
            break
        }
    }
    if ($failed) { continue }
    Write-Host " ok."
    $num_ok++
}

Write-Host "example result: $result. $num_ok passed; $num_failed failed"
exit $num_failed