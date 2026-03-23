import kagglehub
import shutil
import os

print("Downloading SSBD from Kaggle...")
path = kagglehub.dataset_download('shradheypathak/ssbd3')
print(f"Downloaded to: {path}")

ssbd_dir = r"D:\WORK\VScode\Capstone\SSBD-file"

# The downloaded directory structure from Kaggle usually looks like:
# path/
#   armflapping/
#   headbanging/
#   spinning/

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(ssbd_dir, item)
    
    if os.path.isdir(s):
        print(f"Moving directory {s} to {d}")
        # if folder exists (e.g. empty 'armflapping'), remove it so copytree works
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        print(f"Copying file {s} to {d}")
        shutil.copy2(s, d)
        
print("Dataset successfully moved to SSBD-file!")
