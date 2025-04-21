#!/bin/bash
#SBATCH -p gpu					# Partition (queue)
#SBATCH -N 1					# Number of nodes
#SBATCH -n 1					# Number of tasks
#SBATCH --gres=gpu:1			# Request 1 GPU
#SBATCH --cpus-per-task=4		# Number of CPUs
#SBATCH --mem=16G				# Memory allocation
#SBATCH -t 0-02:00				# Max job time
#SBATCH -o slurm.%N,%j.out      # STDOUT log
#SBATCH -e slurm.%N,%j.err      # STDERR log
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=abigail.teitgen@csic.es  # Email for notifications


# Load modules
module load rama0.3
module load GCCcore/12.3.0
module load Python/3.11.3

# Activate virtual environment (stored in $HOME)
source ~/drago-env/bin/activate

# Scratch
SCRATCH_DIR=/lustre/scratch-global/cib/abiel/AMES_$SLURM_JOB_ID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# Define paths
WORKDIR=/lustre/home/cib/abiel/AMES
INPUT_FILE=$WORKDIR/train_sample.yml      # original input in home
SCRIPT=$WORKDIR/GNN_MTL.py                      # training script
OUTPUT_DIR=$SCRATCH_DIR/output
mkdir -p $OUTPUT_DIR

# Copy input data to scratch
cp $INPUT_FILE $SCRATCH_DIR

# Run training script
python GNN_MTL_GPU.py --input_file "$INPUT_FILE" --output_dir "$OUTPUT_DIR"

# Copy outputs (model checkpoints, logs) back to home
cp -r $OUTPUT_DIR $WORKDIR

rm -rf $SCRATCH_DIR

echo "Job completed."