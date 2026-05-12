#!/usr/bin/env bash
#SBATCH --job-name=exp1-launch_separate_folds
#SBATCH -A NAISS2025-5-662 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 0-24:00:00
# Output files
#SBATCH --error=logs/launch_job_%J.err
#SBATCH --output=logs/launch_job_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=camillocaruso952@gmail.com
#SBATCH --array=0-5

# Load modules
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

EXPERIMENT_NAMES=(
  "MIMIC_multimodal"
  #"MIMIC_multimodal_zeros"
  #"MIMIC_multimodal_pooling"
  #"MIMIC_multimodal_model_selection"
  #"MIMIC_multimodal_zeros_maria"
  #"MIMIC_multimodal_model_selection_maria"
  #"MIMIC_multimodal_early"
)

#FOLDS=(0 1 2 3 4)
FOLDS=(2 3 4)

TRAIN_MISSING=(
  #"[[0.0, 0.0]]"
  #"[[0.0, 0.25]]"
  #"[[0.0, 0.5]]"
  "[[0.0, 0.75]]"
  #"[[0.75, 0.0]]"
  "[[0.5, 0.0]]"
  #"[[0.25, 0.0]]"
)

EXPERIMENTS=()

for EXP in "${EXPERIMENT_NAMES[@]}"; do
  for MISSING in "${TRAIN_MISSING[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
      EXPERIMENTS+=("${EXP}|${FOLD}|${MISSING}")
    done
  done
done

IDX=$SLURM_ARRAY_TASK_ID
if (( IDX < 0 || IDX >= ${#EXPERIMENTS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range (0..$((${#EXPERIMENTS[@]}-1)))."
  exit 1
fi

STEP=40

N_FOLDS=${#FOLDS[@]}
N_MISSING=${#TRAIN_MISSING[@]}
EXP_GROUP_SIZE=$((N_FOLDS * N_MISSING))

LOCAL_IDX=$(( IDX % EXP_GROUP_SIZE ))

sleep $(( LOCAL_IDX * STEP ))

# Activate venv
source /mimer/NOBACKUP/groups/naiss2023-6-336/ccaruso/venvGPU/bin/activate

# Executes the code
cd /mimer/NOBACKUP/groups/naiss2023-6-336/ccaruso/CMC_utils_project

RECORD="${EXPERIMENTS[$IDX]}"

IFS='|' read -r EXP_NAME FLD MISS <<< "$RECORD"

echo "Running: exp=$EXP_NAME fold=$FLD missing=$MISS (idx=$IDX)"

# Train
python ./main.py experiment="$EXP_NAME" missing_percentages="$MISS" +fold_to_do="$FLD"

# Deactivate venv
deactivate
