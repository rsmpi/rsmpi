#!/bin/sh

set -e

# enable oversubscribing when using newer Open MPI
export OMPI_MCA_rmaps_base_oversubscribe=1

EXAMPLES_DIR="examples"
BINARIES_DIR="target/debug/examples"

if [ ! -d "${BINARIES_DIR}" ]
then
  echo "Examples not found in ${BINARIES_DIR}"
  exit 1
fi

binaries=$(ls ${EXAMPLES_DIR} | sed "s/\\.rs\$//")
num_binaries=$(printf "%d" "$(echo "${binaries}" | wc -w)")

printf "running %d examples\n" ${num_binaries}

num_ok=0
num_failed=0
result="ok"

for binary in ${binaries}
do
  printf "example ${binary} on 2...8 processes"
  output_file=${binary}_output
  for num_proc in $(seq 2 8)
  do
    if (mpiexec -n ${num_proc} "${BINARIES_DIR}/${binary}" > "${output_file}" 2>&1)
    then
      printf "."
      rm -f "${output_file}"
    else
      printf " failed on %d processes.\noutput:\n" ${num_proc}
      cat "${output_file}"
      rm -f "${output_file}"
      num_failed=$((${num_failed} + 1))
      result="failed"
      continue 2
    fi
  done
  printf " ok.\n"
  num_ok=$((${num_ok} + 1))
done

printf "\nexample result: ${result}. ${num_ok} passed; ${num_failed} failed\n\n"
exit ${num_failed}
