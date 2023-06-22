import os
from os.path import join, abspath, dirname, exists
from os import makedirs
from subprocess import call

RESULTS_PATH = os.environ.get('PBRFF_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))
PROJECT_ROOT = dirname(abspath(__file__))
    
def launch_slurm_experiment(dataset, experiments, landmarks_method, n_cpu, time, dispatch_path, d):
    exp_file = join(dispatch_path, f"{dataset}__" + "__".join(experiments))
                        
    submission_script = ""
    submission_script += f"#!/bin/bash\n"
    submission_script += f"#SBATCH --account=def-pager47\n"
    submission_script += f"#SBATCH --nodes=1\n" 
    submission_script += f"#SBATCH --time={time}:00:00\n"
    submission_script += f"#SBATCH --mem=20G\n#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user=jacob.comeau.1@ulaval.ca\n" 
    submission_script += f"#SBATCH --output={exp_file + '.out'}\n\n" 
    submission_script += "module load python/3.8\n virtualenv --no-download $SLURM_TMPDIR/env\n"
    submission_script += "source $SLURM_TMPDIR/env/activate\n pip install --no-index --upgrade pip\n"
    submission_script += "pip install --no-index -r requirements.txt\n"
    submission_script += f"cd $HOME/dev/git/pbrff\n" 
    submission_script += f"date\n" 
    submission_script += f"python experiment.py -d {dataset} -e {' '.join(experiments)} -l {' '.join(landmarks_method)} -r {' '.join(d)} -n 1"

    submission_path = exp_file + ".sh"
    with open(submission_path, 'w') as out_file:
        out_file.write(submission_script)
        
    call(["sbatch", submission_path])

def main():
    datasets = ["ads"] #[ "ads", "adult", "mnist17", "mnist49", "mnist56"]
    experiments = ["greedy_kernel"]
    landmarks_method = ["random"]
    n_cpu = 40
    time = 1

    D_range = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


    dispatch_path = join(RESULTS_PATH, "dispatch")
    if not exists(dispatch_path): makedirs(dispatch_path)
    
    for dataset in datasets:
        for d in D_range:
            print(f"Launching {dataset} and D : {d}")
            launch_slurm_experiment(dataset, experiments, landmarks_method, n_cpu, time, dispatch_path, d)
    
    print("### DONE ###")

if __name__ == '__main__':
    main()