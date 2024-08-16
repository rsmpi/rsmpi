#!/bin/sh

set -e

# enable oversubscribing for Open-MPI 4.x
export OMPI_MCA_rmaps_base_oversubscribe=1
# Open-MPI 5.x: https://github.com/open-mpi/ompi/issues/8955#issuecomment-846272943
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

EXAMPLES_DIR="examples"

if [ -z "${examples}" ]; then
  examples=$(ls ${EXAMPLES_DIR} | sed "s/\\.rs\$//")
fi
num_examples=$(printf "%d" "$(echo "${examples}" | wc -w)")

maxnp=3
printf "running %d examples\n" ${num_examples}

num_ok=0
num_failed=0
result="ok"

for example in ${examples}
do
  if [ "port" = "${example}" ]; then
    range_min=1
    range_max=1
  else
    range_min=2
    range_max=${maxnp}
  fi
  printf "example ${example} on ${range_min}...${range_max} processes"
  output_file="/tmp/${example}_output"
  for num_proc in $(seq ${range_min} ${range_max})
  do
    if (cargo mpirun "$@" --verbose -n ${num_proc} --example "${example}" > "${output_file}" 2>&1)
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
