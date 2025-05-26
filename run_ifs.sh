#!/bin/bash

export LD_LIBRARY_PATH=/home/jradford/anaconda3/envs/aiwp_realtime/lib/python3.11/site-packages/nvidia/cudnn/lib
aimodels=/home/jradford/anaconda3/envs/aiwp_realtime_ifs/bin/ai-models
python=/home/jradford/anaconda3/envs/aiwp_realtime_ifs/bin/python

rm -r /mnt/aiweathernas/aiwp_realtime/output_data/*

current_datetime=$(date -u +"%Y-%m-%d %H:%M:%S")

# Convert current_datetime to timestamp
current_timestamp=$(date -u -d "$current_datetime" +"%s")

# Calculate the nearest 6-HH interval (00, 06, 12, or 18 UTC)
nearest_interval=$((current_timestamp - current_timestamp % (12*3600)))

# Convert the nearest_interval back to formatted datetime
rounded_datetime=$(date -u -d "@$nearest_interval" +"%Y-%m-%d %H:%M:%S")

# Extract YEAR, MONTH, DAY, and HH from the rounded datetime
YEAR=$(date -u -d "$rounded_datetime" +"%Y")
MONTH=$(date -u -d "$rounded_datetime" +"%m")
DAY=$(date -u -d "$rounded_datetime" +"%d")
HH=$(date -u -d "$rounded_datetime" +"%H")

url="https://data.ecmwf.int/forecasts/${YEAR}${MONTH}${DAY}/${HH}z/ifs/0p25/oper/${YEAR}${MONTH}${DAY}${HH}0000-0h-oper-fc.grib2"

while true; do
    # Send a HEAD request to the URL and check the status code
    status_code=$(curl -o /dev/null -s -w "%{http_code}" "$url")

    # Check if the status code indicates the URL exists
    if [[ "$status_code" -ge 200 && "$status_code" -lt 400 ]]; then
        echo "$(date): $url exists (status code: $status_code)"
        break  # Exit the while loop if the URL exists
    else
        echo "$(date): $url does not exist or is not accessible (status code: $status_code)"
    fi

    # Wait for 30 seconds before checking again
    sleep 30
done

fcnv2head=/mnt/aiweathernas/aiwp_realtime/output_data/FOUR_v200_IFS/${YEAR}/${MONTH}${DAY}
pwhead=/mnt/aiweathernas/aiwp_realtime/output_data/PANG_v100_IFS/${YEAR}/${MONTH}${DAY}
gchead=/mnt/aiweathernas/aiwp_realtime/output_data/GRAP_v100_IFS/${YEAR}/${MONTH}${DAY}
auhead=/mnt/aiweathernas/aiwp_realtime/output_data/AURO_v100_IFS/${YEAR}/${MONTH}${DAY}

fcnv2tail=FOUR_v200_IFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
pwtail=PANG_v100_IFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
gctail=GRAP_v100_IFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
autail=AURO_v100_IFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06

fcnv2path=${fcnv2head}/${fcnv2tail}
pwpath=${pwhead}/${pwtail}
gcpath=${gchead}/${gctail}
aupath=${auhead}/${autail}

fcnv2assets=/mnt/aiweathernas/aiwp_realtime/assets/fcnv2
pwassets=/mnt/aiweathernas/aiwp_realtime/assets/pw
gcassets=/mnt/aiweathernas/aiwp_realtime/assets/gc
auassets=/mnt/aiweathernas/aiwp_realtime/assets/au

mkdir -p ${fcnv2head}
mkdir -p ${pwhead}
mkdir -p ${gchead}
mkdir -p ${auhead}

# This section runs the models serially and converts to netcdfs concurrently
# If there are issues with monopolizing disk I/O or CPU change these to
# all run serially.

# Run FourCastNetV2-small
${aimodels} --input opendata --assets ${fcnv2assets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${fcnv2path} fourcastnetv2-small
${python} /mnt/aiweathernas/aiwp_realtime/grib2nc.py ${fcnv2path} IFS fourcastnetv2-small ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run PanguWeather
${aimodels} --input opendata --assets ${pwassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${pwpath} panguweather
${python} /mnt/aiweathernas/aiwp_realtime/grib2nc.py ${pwpath} IFS panguweather ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run GraphCast
${aimodels} --input opendata --assets ${gcassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${gcpath} graphcast
${python} /mnt/aiweathernas/aiwp_realtime/grib2nc.py ${gcpath} IFS graphcast ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run Aurora
${aimodels} --input opendata --assets ${auassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${aupath} aurora --model-version 0.25-finetuned
${python} /mnt/aiweathernas/aiwp_realtime/grib2nc.py ${aupath} IFS aurora ${YEAR}${MONTH}${DAY} ${HH}00 &

wait
