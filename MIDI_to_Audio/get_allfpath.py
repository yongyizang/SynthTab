import os
import sys

# 定义起始目录和输出文件名
start_directory = sys.argv[1]
output_flac_file = sys.argv[1].strip('/') + '_audio.txt'
output_f0_file = sys.argv[1].strip('/') + '_f0.txt'

file1 = open(output_flac_file, 'w')
file2 = open(output_f0_file, 'w')

for dirpath, dirnames, filenames in os.walk(start_directory):
    for filename in filenames:
        # 获取完整的文件路径
        full_path = os.path.join(dirpath, filename)
        # 将文件路径写入输出文件
        if full_path.endswith('.flac'):
            file1.write(full_path + '\n')
        elif full_path.endswith('.pkl'):
            file2.write(full_path + '\n')

