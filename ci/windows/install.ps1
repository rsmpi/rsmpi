Write-Output "Installing MS-MPI SDK..."

Invoke-WebRequest `
    -Uri https://download.microsoft.com/download/4/A/6/4A6AAED8-200C-457C-AB86-37505DE4C90D/msmpisdk.msi `
    -OutFile .\msmpisdk.msi


Start-Process -Wait -FilePath msiexec.exe -ArgumentList "/i msmpisdk.msi /quiet /qn /log install.log"
# For some reason we're getting a bad exit code even on success
# if ($LASTEXITCODE -ne 0) {
#     exit $LASTEXITCODE
# }

Write-Output "Installed MS-MPI SDK!"

Write-Output "Installing MS-MPI Redist..."

Invoke-WebRequest `
    -Uri https://download.microsoft.com/download/4/A/6/4A6AAED8-200C-457C-AB86-37505DE4C90D/msmpisetup.exe `
    -OutFile .\msmpisetup.exe

Start-Process -Wait -FilePath msmpisetup.exe -ArgumentList "-unattend -full"
    
Write-Output "Installed MS-MPI Redist!"

Write-Output "Installing Rust..."

Invoke-WebRequest `
    -Uri https://win.rustup.rs/ `
    -OutFile rustup-init.exe

cmd.exe /c ".\rustup-init.exe -y 2>&1"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Installed Rust!"

$env:PATH += ";$HOME\.cargo\bin"

Write-Output "Installing cargo-mpirun..."

cmd.exe /c "cargo install --force cargo-mpirun 2>&1"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Installed cargo-mpirun!"