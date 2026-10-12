#!/bin/bash
cd /home/ajian/PFA_radar-2026

env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
bash -lc 'source /opt/ros/humble/setup.bash && export PYTHONPATH=/home/ajian/anaconda3/envs/Qiqi/lib/python3.10/site-packages:$PYTHONPATH && /home/ajian/anaconda3/envs/Qiqi/bin/python -u main.py'
