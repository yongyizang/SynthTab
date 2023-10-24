# Script used for generating the smallest dev set.

import os, sys
import librosa
import random

# set random seed
random.seed(1234)

source_dir = "/home/finch/external_ssd/SynthTab"
target_dump_folder = "/home/finch/SynthTab_Dev"
randomly_select_for_each_timbre = 3

def get_n_random_files(directory, n):
    """
    directory: expects a string. The directory to search for files.
    n: number of files to return.
    """
    
    # get all *.flac files under the directory.
    flac_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".flac"):
                flac_files.append(os.path.join(root, file))
    
    # random select n files from flac_files
    return random.sample(flac_files, n)

partitions = ["acoustic", "electric_clean", "electric_distortion_di", "electric_muted"]
avoids = ["tmp", "allpaths_electric_clean", "all_path_electric_distortion"]
candidates = []

for partition in partitions:
    partition_dir = source_dir + "/" + partition
    subdirs = os.listdir(partition_dir)
    for subdir in subdirs:
        if os.path.isdir(os.path.join(partition_dir, subdir)):
            if subdir in avoids:
                continue
            curr_dir = os.path.join(partition_dir, subdir)
            candidates.extend(get_n_random_files(curr_dir, randomly_select_for_each_timbre))

# process the flac files, keep only the folder name
for i in range(len(candidates)):
    candidates[i] = candidates[i].split("/")[:-1]
    candidates[i] = "/".join(candidates[i])

for candidate in candidates:
    # copy to target folder
    curr_dump_folder = candidate.replace(source_dir, target_dump_folder)
    if not os.path.exists(curr_dump_folder):
        os.makedirs(curr_dump_folder)
    os.system("cp -r \"" + candidate + "\" \"" + curr_dump_folder + "\"")

# get jams file
for i in range(len(candidates)):
    candidates[i] = candidates[i].split("/")[-1]
    candidates[i] = source_dir + "/jams/" + candidates[i]
    
for candidate in candidates:
    os.system("cp -r \"" + candidate + "\" \"" + target_dump_folder + "/jams\"")