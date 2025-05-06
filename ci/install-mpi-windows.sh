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
    hash=18932                                version=2021.7.0  build=9549
    hash=19011                                version=2021.7.1  build=15761
    hash=19160                                version=2021.8.0  build=25543
    hash=c11b9bf0-2527-4925-950d-186dba31fb40 version=2021.9.0  build=43421
    hash=4f7f4251-7781-446f-89ac-c777dacb766f version=2021.10.0 build=49373
    hash=b7596581-64db-4820-bcfe-74ad9f5ec657 version=2021.11.0 build=49512
    hash=fab706bb-ca1e-4cc9-b76a-a12df3cc984e version=2021.12.0 build=539
    hash=a3a49de8-dc40-4387-9784-5227fccb6caa version=2021.12.1 build=7
    hash=e20e3226-9264-41a0-bc18-6026d297e10d version=2021.13.0 build=717
    hash=ea625b1d-8a8a-4bd5-b31d-7ed55af45994 version=2021.13.1 build=768
    hash=44400f77-51cb-4f15-8424-0e11eeb41832 version=2021.14.0 build=785
    hash=e9f49ab3-babd-4753-a155-ceeb87e36674 version=2021.14.1 build=8
    hash=29e1f37b-f7c7-4cd6-988c-6ddf80aadf6a version=2021.14.2 build=901
    hash=edf463a6-a6ad-43d2-a588-daa2e30f8735 version=2021.15.0 build=496
    baseurl=https://registrationcenter-download.intel.com
    subpath=akdlm/IRC_NAS/$hash
    if test $version \< 2021.14.0; then
        package=w_mpi_oneapi_p_${version}.${build}_offline.exe
    else
        package=intel-mpi-${version}.${build}_offline.exe
    fi
    set -x
    curl -sO $baseurl/$subpath/$package
    ./$package -s -a --silent --eula accept
    set +x
}

setup_win_intel_oneapi_mpi_env () {
    ONEAPI_ROOT="C:\Program Files (x86)\Intel\oneAPI"
    I_MPI_ROOT="${ONEAPI_ROOT}\mpi\latest"
    I_MPI_OFI_LIBRARY_INTERNAL="1"
    I_MPI_BIN="${I_MPI_ROOT}\bin"
    I_MPU_LIBFABRIC_BIN="${I_MPI_ROOT}\opt\mpi\libfabric\bin"

    echo "ONEAPI_ROOT=${ONEAPI_ROOT}" >> $GITHUB_ENV
    echo "I_MPI_ROOT=${I_MPI_ROOT}" >> $GITHUB_ENV
    echo "I_MPI_OFI_LIBRARY_INTERNAL=${I_MPI_OFI_LIBRARY_INTERNAL}" >> $GITHUB_ENV
    echo "${I_MPI_BIN}" >> $GITHUB_PATH
    echo "${I_MPU_LIBFABRIC_BIN}" >> $GITHUB_PATH

    export PATH="$(cygpath -u "${I_MPI_BIN}"):$PATH"
    export PATH="$(cygpath -u "${I_MPU_LIBFABRIC_BIN}"):$PATH"
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
