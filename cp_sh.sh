#!/bin/bash



# 查找并复制文件
find . -maxdepth 1 -type f -size -100M -exec cp {} ../VDIT \;

# 查找并复制文件夹
for dir in $(find . -maxdepth 1 -type d); do
    # 计算文件夹大小
    dir_size=$(du -sb "$dir" | awk '{print $1}')
    
    # 如果文件夹大小小于100M，则复制
    if [ "$dir_size" -lt $((100 * 1024 * 1024)) ]; then
        rsync -a "$dir" ../VDIT/
    fi
done
