#!/usr/bin/env bash
# Reviewable command printer. This script NEVER launches a job or contacts Silas.
# Usage: bash scripts/codealign_doma_commands.sh opv2v /absolute/experiment/root
set -euo pipefail
dataset=${1:?choose opv2v or v2xset}
experiment=${2:?supply the reviewed experiment root}
case "$dataset" in opv2v|v2xset) ;; *) exit 2 ;; esac
printf '# Run from the CodeAlign repository after reviewing protocol_lock.yaml.\n'
printf '# All stage settings come from experiment_root/STAGE/config.yaml.\n'
printf 'python -B -m opencood.tools.run_codealign_doma prepare --dataset %q --experiment-root %q\n' "$dataset" "$experiment"
printf 'python -B -m opencood.tools.check_codealign_doma --check-data --dataset %q --output %q\n' "$dataset" "$experiment/data_audit.json"
printf 'python -B -m opencood.tools.check_codealign_doma --cuda-smoke --dataset %q --experiment-root %q\n' "$dataset" "$experiment"
for family in single code_space; do
    for modality in m1 m2 m3 m4; do
        printf 'python -B -m opencood.tools.run_codealign_doma train --experiment-root %q --stage %q\n' "$experiment" "$family/$modality"
    done
done
for modality in m2 m3 m4; do
    printf 'python -B -m opencood.tools.run_codealign_doma train --experiment-root %q --stage %q\n' "$experiment" "fcf/${modality}_to_m1"
done
printf 'python -B -m opencood.tools.run_codealign_doma assemble --experiment-root %q\n' "$experiment"
for cav in 1 2 3 4; do
    printf 'python -B -m opencood.tools.run_codealign_doma infer --experiment-root %q --use-cav %d\n' "$experiment" "$cav"
done
printf 'python -B -m opencood.tools.run_codealign_doma summarize --experiment-root %q\n' "$experiment"
