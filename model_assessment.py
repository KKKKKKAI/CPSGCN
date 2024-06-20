import os
import numpy as np

from pathlib import Path
from runner_script_manager import *


class ModelAssessor():
    def __init__(self, repeat, config, dataset_name, model_assess_dir):
        self.accs = [0] * repeat
        self.repeat = repeat
        self.config = config
        self.dataset_name = dataset_name
        self.model_assess_dir = model_assess_dir

    def assess(self, data_mask, assess_runner_path=Path("runner_scipts/")):
        self.config["test_run"] = True
        self.config["data_mask"] = data_mask
        script_generated = False
        if os.path.exists(assess_runner_path):
            script_generated = True

        for r in range(self.repeat):
            print("model assessment iteration: ", r)
            assessment_dir = Path(self.model_assess_dir / f"iteration_{r}/")
            assessment_dir.mkdir(parents=True, exist_ok=True)

            if not script_generated:
                generate_runner(self.config, assess_runner_path, assessment_dir)

        os.chmod(assess_runner_path, 0o755)
        execute_shell_script(assess_runner_path)

    def get_avg_acc(self):
        for i in range(self.repeat):
            output_dir = Path(self.model_assess_dir / f"iteration_{i}/")
            with open(output_dir / "experiment_log.txt", 'r') as file:
                for _ in range(2):
                    next(file)
                third_line = file.readline().strip()
                results = third_line.split(',')
                acc, _, _, _ = results[:4]
                self.accs[i] = float(acc)

        return sum(self.accs) / self.repeat
    
    def get_sd(self):
        return np.std(self.accs)
