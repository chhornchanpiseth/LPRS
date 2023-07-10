#!/bin/bash

# */5 * * * * ~/LPRS/auto_run.sh  to run bash command every 5 minutes
cd /home/ubuntu/LPRS


ip1="10.2.7.51:554"
if [[ ! $(pgrep -f "$ip1") ]]; then
    nohup /home/ubuntu/anaconda3/envs/wpod-net/bin/python -u /home/ubuntu/JP-SYSTEM/Japanese-LPRS/main_stream.py \
        --in_direction down \
        --debug False \
        --location "Ou Tasek" \
        --ch_url "http://127.0.0.1:5000" \
        --ip "rtsp://admin:lprs0ptimus@10.2.7.51:554/1" \
        --limit_top 0.40 \
        --limit_bottom 0.51 \
        --limit_top_motor 0.50 \
        --limit_bottom_motor 0.60 \
        --is_motor_only False \
        > logs/output_10_2_7_51.out 2>&1 &
    echo "rerun $ip1"
fi
sleep 5

ip2="10.2.7.17:554"
if [[ ! $(pgrep -f "$ip2") ]]; then
    nohup /home/ubuntu/anaconda3/envs/wpod-net/bin/python -u /home/ubuntu/JP-SYSTEM/Japanese-LPRS/main_stream.py \
        --in_direction down \
        --debug False \
        --location "White Shop" \
        --ch_url "http://127.0.0.1:5000" \
        --ip "rtsp://admin:lprs0ptimus@$ip2/1" \
        --limit_top 0.55 \
        --limit_bottom 0.65 \
        --limit_top_motor 0.60 \
        --limit_bottom_motor 0.75 \
        --is_motor_only False \
        > logs/output_10_2_7_17.out 2>&1 &
    echo "rerun $ip2"
fi
if [[ ! $(pgrep -f "10.2.7.52:554") ]]; then
    nohup /home/ubuntu/anaconda3/envs/wpod-net/bin/python -u /home/ubuntu/JP-SYSTEM/Japanese-LPRS/main_stream.py \
        --in_direction up \
        --debug False \
        --location "White Shop" \
        --ch_url "http://127.0.0.1:5000" \
        --ip "rtsp://admin:lprs0ptimus@10.2.7.52:554/1" \
        --limit_top_motor 0.7 \
        --limit_bottom_motor 0.8 \
        --is_motor_only True \
        --motion_top 0.3 \
        --motion_right 0.6 \
        > logs/output_10_2_7_52.out 2>&1 &
    echo "rerun 10.2.7.52:554"
fi
sleep 5
if [[ ! $(pgrep -f "10.2.7.50:554") ]]; then
    nohup /home/ubuntu/anaconda3/envs/wpod-net/bin/python -u /home/ubuntu/JP-SYSTEM/Japanese-LPRS/main_stream.py \
        --in_direction up \
        --debug False \
        --location "Ou Tasek" \
        --ch_url "http://127.0.0.1:5000" \
        --ip "rtsp://admin:lprs0ptimus@10.2.7.50:554/1" \
        --limit_top_motor 0.60 \
        --limit_bottom_motor 0.74 \
        --is_motor_only True \
        --motion_top 0.2 \
        --motion_right 0.6 \
        > logs/output_10_2_7_50.out 2>&1 &
    echo "rerun 10.2.7.50:554"
fi
exit
