#!/usr/bin/env bash
# Idempotent bootstrap + submit for ml-research-kills-alpha

set -euo pipefail

# --------- Settings you can override via env vars when calling the script ---------
repo_url="${REPO_URL:-git@github.com:FeTiberga/ml-research-kills-alpha.git}"
repo_dir="${REPO_DIR:-$HOME/ml-research-kills-alpha}"
git_branch="${GIT_BRANCH:-main}"

year_start="${YEAR_START:-2005}"
year_end="${YEAR_END:-2023}"
submit="${SUBMIT:-both}"                 # cpu|gpu|both|none
run_data_build="${RUN_DATA_BUILD:-auto}" # auto|yes|no  (auto runs only if processed panel missing)

python_bin="${PYTHON_BIN:-python3}"

# CPU job resources
cpu_cpus="${HTC_CPUS:-8}"
cpu_mem="${HTC_MEM:-32G}"
cpu_time="${HTC_TIME:-01:30:00}"

# GPU job resources
gpu_per_task="${GPU_PER_TASK:-1}"
gpu_mem="${GPU_MEM:-32G}"
gpu_time="${GPU_TIME:-02:00:00}"

# --------- Helpers ---------
log(){ printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }

ensure_repo(){
  if [[ ! -d "$repo_dir/.git" ]]; then
    log "Cloning $repo_url → $repo_dir"
    git clone "$repo_url" "$repo_dir"
  fi
  cd "$repo_dir"
  log "Updating repo on branch $git_branch"
  git fetch --all -q
  git checkout -q "$git_branch"
  git pull --ff-only -q
}

ensure_venv(){
  if [[ ! -d ".venv" ]]; then
    log "Creating venv"
    "$python_bin" -m venv .venv
  fi
  # shellcheck source=/dev/null
  source .venv/bin/activate
  pip -q install --upgrade pip wheel
  if [[ -f requirements.txt ]]; then
    new_hash="$(sha256sum requirements.txt | awk '{print $1}')"
    old_hash="$(cat .venv/.requirements.sha256 2>/dev/null || true)"
    if [[ "$new_hash" != "$old_hash" ]]; then
      log "Installing/updating Python dependencies"
      pip install -r requirements.txt -e .
      echo "$new_hash" > .venv/.requirements.sha256
    else
      log "Dependencies unchanged — skipping pip install"
    fi
  fi
}

maybe_build_data(){
  # Adjust the check to your processed artifact path if different
  processed_dir="data/processed"
  master_panel="$processed_dir/master_panel.csv"
  case "$run_data_build" in
    yes) need_build=1 ;;
    no)  need_build=0 ;;
    auto)
      [[ -f "$master_panel" ]] && need_build=0 || need_build=1
      ;;
    *) need_build=0 ;;
  esac
  if [[ "$need_build" -eq 1 ]]; then
    log "Building processed data panel"
    python -m ml_research_kills_alpha.datasets.data_pipeline || {
      log "Data build failed"; exit 2;
    }
  else
    log "Processed data exists — skipping data build"
  fi
}

write_sbatch_if_missing(){
  mkdir -p slurm logs
  # CPU script
  if [[ ! -f slurm/predict_cpu.sbatch ]]; then
    cat > slurm/predict_cpu.sbatch <<'SBCPU'
#!/usr/bin/env bash
#SBATCH -J ml-cpu
#SBATCH -p htc
#SBATCH -c ${CPU_CPUS}
#SBATCH --mem=${CPU_MEM}
#SBATCH -t ${CPU_TIME}
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-${ARRAY_MAX}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

YEARS=(${YEARS_LIST})
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

python -m ml_research_kills_alpha.prediction_pipeline \
  --end_year "${YEAR}" \
  --target_col abret \
  --force_ml
SBCPU
    chmod +x slurm/predict_cpu.sbatch
  fi

  # GPU script
  if [[ ! -f slurm/predict_gpu.sbatch ]]; then
    cat > slurm/predict_gpu.sbatch <<'SBGPU'
#!/usr/bin/env bash
#SBATCH -J ml-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:${GPU_PER_TASK}
#SBATCH -c 4
#SBATCH --mem=${GPU_MEM}
#SBATCH -t ${GPU_TIME}
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-${ARRAY_MAX}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

YEARS=(${YEARS_LIST})
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

python -m ml_research_kills_alpha.prediction_pipeline \
  --end_year "${YEAR}" \
  --target_col abret \
  --force_ml
SBGPU
    chmod +x slurm/predict_gpu.sbatch
  fi
}

submit_jobs(){
  # Build year list and array size
  years_list=()
  for ((y=year_start; y<=year_end; y++)); do years_list+=("$y"); done
  array_max=$(( ${#years_list[@]} - 1 ))
  years_str="${years_list[*]}"

  # Substitute variables into the sbatch templates
  export YEARS_LIST="$years_str" ARRAY_MAX="$array_max"
  export CPU_CPUS="$cpu_cpus" CPU_MEM="$cpu_mem" CPU_TIME="$cpu_time"
  export GPU_PER_TASK="$gpu_per_task" GPU_MEM="$gpu_mem" GPU_TIME="$gpu_time"

  log "Submitting jobs for years: $years_str"
  case "$submit" in
    cpu)
      envsubst < slurm/predict_cpu.sbatch | sbatch
      ;;
    gpu)
      envsubst < slurm/predict_gpu.sbatch | sbatch
      ;;
    both)
      envsubst < slurm/predict_cpu.sbatch | sbatch
      envsubst < slurm/predict_gpu.sbatch | sbatch
      ;;
    none) log "Submission skipped (SUBMIT=none)";;
