import os


root_path = "/mnt/nfs-ssd/data/DUD-E/"

targets = os.listdir(root_path)
print(targets)
# remove "aa2ar"
targets.remove("aa2ar")