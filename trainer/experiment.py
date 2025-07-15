import os
import json
import datetime
from pathlib import Path

class ExperimentManager:
    def __init__(self, section, experiment_id=None, base_dir="experiments"):
        """
        Manage experiment folders and configurations
        
        Args:
            section: section number (1, 2, or 3)
            experiment_id: optional experiment id, auto-generated if None
            base_dir: base directory for experiments
        """
        self.section = section
        self.base_dir = base_dir
        
        # Generate experiment ID if not provided
        if experiment_id is None:
            self.experiment_id = self._generate_experiment_id()
        else:
            self.experiment_id = experiment_id
            
        self.experiment_path = None
        self.setup_experiment_folder()
        
    def _generate_experiment_id(self):
        """Generate unique experiment ID"""
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Find next available experiment number
        base_pattern = f"{date_str}_section{self.section}_"
        experiment_num = 1
        
        if os.path.exists(self.base_dir):
            existing_folders = [f for f in os.listdir(self.base_dir) 
                              if f.startswith(base_pattern)]
            if existing_folders:
                # Extract numbers and find max
                nums = []
                for folder in existing_folders:
                    try:
                        num = int(folder.split('_')[-1])
                        nums.append(num)
                    except ValueError:
                        continue
                if nums:
                    experiment_num = max(nums) + 1
        
        return f"{base_pattern}{experiment_num:03d}"
    
    def setup_experiment_folder(self):
        """Create experiment folder structure"""
        self.experiment_path = os.path.join(self.base_dir, self.experiment_id)
        
        # Create main experiment folder
        Path(self.experiment_path).mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        subfolders = ['logs', 'plots', 'results']
        for folder in subfolders:
            Path(os.path.join(self.experiment_path, folder)).mkdir(exist_ok=True)
        
        # Create model save directory
        model_dir = "model/saved_models"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment folder created: {self.experiment_path}")
    
    def save_config(self, config):
        """Save experiment configuration"""
        config_path = os.path.join(self.experiment_path, "config.json")
        
        # Add metadata
        config['experiment_id'] = self.experiment_id
        config['section'] = self.section
        config['timestamp'] = datetime.datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved: {config_path}")
    
    def get_experiment_path(self):
        """Return experiment path"""
        return self.experiment_path
    
    def get_logs_path(self):
        """Return logs folder path"""
        return os.path.join(self.experiment_path, "logs")
    
    def get_plots_path(self):
        """Return plots folder path"""
        return os.path.join(self.experiment_path, "plots")
    
    def get_results_path(self):
        """Return results folder path"""
        return os.path.join(self.experiment_path, "results")
    
    def get_model_save_path(self):
        """Return model save path"""
        return f"model/saved_models/{self.experiment_id}_model"
    
    def load_config(self):
        """Load experiment configuration"""
        config_path = os.path.join(self.experiment_path, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return None
    
    def list_experiment_files(self):
        """List all files in experiment folder"""
        files = {}
        for root, dirs, filenames in os.walk(self.experiment_path):
            rel_path = os.path.relpath(root, self.experiment_path)
            if rel_path == '.':
                rel_path = 'root'
            files[rel_path] = filenames
        return files
    
    def cleanup_experiment(self):
        """Remove experiment folder (use with caution)"""
        import shutil
        if os.path.exists(self.experiment_path):
            shutil.rmtree(self.experiment_path)
            print(f"Experiment folder removed: {self.experiment_path}")