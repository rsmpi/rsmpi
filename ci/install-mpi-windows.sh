echo "Installing MS-MPI SDK..."

curl -O "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisdk.msi"

msiexec.exe //i msmpisdk.msi //quiet //qn //log ./install.log

echo "Installed MS-MPI SDK!"

echo "Installing MS-MPI Redist..."

curl -O "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisetup.exe"

./msmpisetup.exe -unattend -full

echo "Installed MS-MPI Redist!"
