#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mpi_type>"
    echo "  mpi_type: msmpi or intelmpi"
    exit 1
fi

MPI_TYPE=$1

# Function to install Intel MPI on Windows
# reference: https://github.com/mpi4py/setup-mpi/blob/master/setup-mpi.sh
setup_win_intel_oneapi_mpi () {
    hash="edf463a6-a6ad-43d2-a588-daa2e30f8735" 
    version="2021.15.0" 
    build="496"
    baseurl="https://registrationcenter-download.intel.com"
    subpath="akdlm/IRC_NAS/$hash"
    package="intel-mpi-${version}.${build}_offline.exe"
    set -x
    curl -sO "$baseurl/$subpath/$package"
    ./"$package" -s -a --silent --eula accept
    set +x
}

setup_win_intel_oneapi_mpi_env () {
    ONEAPI_ROOT="C:\Program Files (x86)\Intel\oneAPI"
    I_MPI_ROOT="${ONEAPI_ROOT}\mpi\latest"
    I_MPI_OFI_LIBRARY_INTERNAL="1"
    I_MPI_BIN="${I_MPI_ROOT}\bin"
    I_MPI_LIBFABRIC_BIN="${I_MPI_ROOT}\opt\mpi\libfabric\bin"

    echo "ONEAPI_ROOT=${ONEAPI_ROOT}" >> $GITHUB_ENV
    echo "I_MPI_ROOT=${I_MPI_ROOT}" >> $GITHUB_ENV
    echo "I_MPI_OFI_LIBRARY_INTERNAL=${I_MPI_OFI_LIBRARY_INTERNAL}" >> $GITHUB_ENV
    echo "${I_MPI_BIN}" >> $GITHUB_PATH
    echo "${I_MPI_LIBFABRIC_BIN}" >> $GITHUB_PATH

    export PATH="$(cygpath -u "${I_MPI_BIN}"):$PATH"
    export PATH="$(cygpath -u "${I_MPI_LIBFABRIC_BIN}"):$PATH"
}

setup_win_ms_mpi_env () {
    MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
    MSMPI_LIB32="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86"
    MSMPI_LIB64="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
    MSMPI_BIN="C:\Program Files\Microsoft MPI\Bin"

    echo "MSMPI_INC=${MSMPI_INC}" >> $GITHUB_ENV
    echo "MSMPI_LIB32=${MSMPI_LIB32}" >> $GITHUB_ENV
    echo "MSMPI_LIB64=${MSMPI_LIB64}" >> $GITHUB_ENV
    echo "${MSMPI_BIN}" >> $GITHUB_PATH

    export PATH="$(cygpath -u "${MSMPI_BIN}"):$PATH"
}

case "$MPI_TYPE" in
    msmpi)
        echo "Installing MS-MPI SDK..."

        curl -O "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisdk.msi"

        msiexec.exe //i msmpisdk.msi //quiet //qn //log ./install.log

        echo "Installed MS-MPI SDK!"

        echo "Installing MS-MPI Redist..."

        curl -O "https://download.microsoft.com/download/a/5/2/a5207ca5-1203-491a-8fb8-906fd68ae623/msmpisetup.exe"

        ./msmpisetup.exe -unattend -full

        setup_win_ms_mpi_env

        echo "Installed MS-MPI Redist!"
        ;;

    intelmpi)
        echo "Installing Intel MPI..."

        setup_win_intel_oneapi_mpi
        setup_win_intel_oneapi_mpi_env
        hydra_service.exe -install

        echo "Installed Intel MPI!"
        ;;

    *)
        echo "Error: Invalid MPI type '$MPI_TYPE'"
        echo "Usage: $0 <mpi_type>"
        echo "  mpi_type: msmpi or intelmpi"
        exit 1
        ;;
esac
