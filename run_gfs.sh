#!/bin/bash

#Path to GFS environment
aiwp_realtime_env_path=~/anaconda3/envs/aiwp_realtime_test

#Path to working directory
aiwp_realtime_cwd_path=/mnt/aiweathernas/aiwp-realtime

export LD_LIBRARY_PATH=${aiwp_realtime_env_path}/lib/python3.11/site-packages/nvidia/cudnn/lib
aimodels=${aiwp_realtime_env_path}/bin/ai-models-gfs
python=${aiwp_realtime_env_path}/bin/python
aws=${aiwp_realtime_env_path}/bin/aws
s3bucket=s3://noaa-oar-mlwp-data

echo ${aiwp_realtime_cwd_path}/output_data/
rm -r ${aiwp_realtime_cwd_path}/output_data/*

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

YEAR=2025
MONTH=05
DAY=26
HH=00

# Check if initial conditions available
url="https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${YEAR}${MONTH}${DAY}/${HH}/atmos/gfs.t${HH}z.pgrb2.0p25.f000"
# Loop to check the URL every 30 seconds
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

fcnv2head=${aiwp_realtime_cwd_path}/output_data/FOUR_v200_GFS/${YEAR}/${MONTH}${DAY}
pwhead=${aiwp_realtime_cwd_path}/output_data/PANG_v100_GFS/${YEAR}/${MONTH}${DAY}
gchead=${aiwp_realtime_cwd_path}/output_data/GRAP_v100_GFS/${YEAR}/${MONTH}${DAY}
auhead=${aiwp_realtime_cwd_path}/output_data/AURO_v100_GFS/${YEAR}/${MONTH}${DAY}

fcnv2tail=FOUR_v200_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
pwtail=PANG_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
gctail=GRAP_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
autail=AURO_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06

fcnv2path=${fcnv2head}/${fcnv2tail}
pwpath=${pwhead}/${pwtail}
gcpath=${gchead}/${gctail}
aupath=${auhead}/${autail}

fcnv2assets=${aiwp_realtime_cwd_path}/assets/fcnv2
pwassets=${aiwp_realtime_cwd_path}/assets/pw
gcassets=${aiwp_realtime_cwd_path}/assets/gc
auassets=${aiwp_realtime_cwd_path}/assets/au

mkdir -p ${fcnv2head}
mkdir -p ${pwhead}
mkdir -p ${gchead}
mkdir -p ${auhead}

# This section runs the models serially and converts to netcdfs concurrently
# If there are issues with monopolizing disk I/O or CPU change these to
# all run serially.

# Run FourCastNetV2-small
${aimodels} --input gfs --assets ${fcnv2assets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${fcnv2path} fourcastnetv2-small
${python} ${aiwp_realtime_cwd_path}/grib2nc.py ${fcnv2path} GFS fourcastnetv2-small ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run PanguWeather
${aimodels} --input gfs --assets ${pwassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${pwpath} panguweather
${python} ${aiwp_realtime_cwd_path}/grib2nc.py ${pwpath} GFS panguweather ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run GraphCast
${aimodels} --input gfs --assets ${gcassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${gcpath} graphcast
${python} ${aiwp_realtime_cwd_path}/grib2nc.py ${gcpath} GFS graphcast ${YEAR}${MONTH}${DAY} ${HH}00 &

# Run Aurora
${aimodels} --input gfs --assets ${auassets} --date ${YEAR}${MONTH}${DAY} --time ${HH}00 --path ${aupath} aurora --model-version 0.25-finetuned
${python} ${aiwp_realtime_cwd_path}/grib2nc.py ${aupath} GFS aurora ${YEAR}${MONTH}${DAY} ${HH}00 &

wait

rm ${fcnv2path}
rm ${pwpath}
rm ${gcpath}
rm ${aupath}
