import pickle
import os

path = r'd:\WORK\VScode\Capstone\asd-federated-learning\saved_results.pkl'
if os.path.exists(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        # print the keys to understand the structure better if it's large
        # print(data.keys())
        
        # Accessing the accuracy based on plots.py observation
        exp = data.get("experimental_results", {})
        for exp_name, exp_data in exp.items():
            print(f"Experiment: {exp_name}")
            if "facial_experiment" in exp_data:
                fm = exp_data["facial_experiment"].get("final_metrics", {})
                print(f"  Facial Accuracy: {fm.get('global_accuracy', 'N/A')}%")
            if "behavioral_experiment" in exp_data:
                fm = exp_data["behavioral_experiment"].get("final_metrics", {})
                print(f"  Behavioral Accuracy: {fm.get('global_accuracy', 'N/A')}%")
else:
    print(f"File {path} does not exist.")
