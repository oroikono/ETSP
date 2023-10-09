#!/bin/bash
# export DATASETS_CACHE=/cluster/scratch/ooikonomou/huggingface_cache/datasets
cd "$(dirname "$0")"
export SCRIPT_DIR="$(dirname "$(pwd)")/ETSP"

sbatch  <<EOT
#!/bin/bash
###############################
# Define sbatch configuration #
###############################

#SBATCH -n 2
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --job-name=$job_name


echo "###############################################################"
echo ""
echo "Slurm JOB Logs"
echo "..."
echo ""
echo ""
echo "###############################################################"

python "${SCRIPT_DIR}/training_script.py" \
       
        
EOT
