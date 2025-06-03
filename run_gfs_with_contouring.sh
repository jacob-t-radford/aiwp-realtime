#!/bin/bash

#----------------------------------------------------
#--------Things that might need modification---------
#----------------------------------------------------

#Working directories
aiwp_realtime_cwd_path=/mnt/aiweathernas/aiwp-realtime
geojson_path=/mnt/data1/jradford/GEOJSONS

#Environment paths
aiwp_realtime_env_path=~/anaconda3/envs/aiwp_realtime
aiwp_realtime_contouring_env_path=~/anaconda3/envs/aiwp_realtime_contouring
export LD_LIBRARY_PATH=${aiwp_realtime_env_path}/lib/python3.11/site-packages/nvidia/cudnn/lib


#----------------------------
#--------Binary paths--------
#----------------------------

aimodels=${aiwp_realtime_env_path}/bin/ai-models-gfs
python=${aiwp_realtime_env_path}/bin/python
python_contours=${aiwp_realtime_contouring_env_path}/bin/python
aws=${aiwp_realtime_env_path}/bin/aws
tippecanoe=${aiwp_realtime_contouring_env_path}/bin/tippecanoe


#------------------------------
#-------Remote addresses-------
#------------------------------

s3bucket=s3://noaa-oar-mlwp-data
aiweather_address="jradford@www2.cira.colostate.edu"

#---------------------------------
#-------Start of processing-------
#---------------------------------

#Remove old data

#rm -r ${aiwp_realtime_cwd_path}/output_data/*
rm -r ${geojson_path}

#Get current date + time
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

#Model output dirs
fcnv2head=${aiwp_realtime_cwd_path}/output_data/FOUR_v200_GFS/${YEAR}/${MONTH}${DAY}
pwhead=${aiwp_realtime_cwd_path}/output_data/PANG_v100_GFS/${YEAR}/${MONTH}${DAY}
gchead=${aiwp_realtime_cwd_path}/output_data/GRAP_v100_GFS/${YEAR}/${MONTH}${DAY}
auhead=${aiwp_realtime_cwd_path}/output_data/AURO_v100_GFS/${YEAR}/${MONTH}${DAY}

#Model output files
fcnv2tail=FOUR_v200_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
pwtail=PANG_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
gctail=GRAP_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06
autail=AURO_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06

#Full model output path
fcnv2path=${fcnv2head}/${fcnv2tail}
pwpath=${pwhead}/${pwtail}
gcpath=${gchead}/${gctail}
aupath=${auhead}/${autail}

#Model asset paths
fcnv2assets=${aiwp_realtime_cwd_path}/assets/fcnv2
pwassets=${aiwp_realtime_cwd_path}/assets/pw
gcassets=${aiwp_realtime_cwd_path}/assets/gc
auassets=${aiwp_realtime_cwd_path}/assets/au

#Create output dirs
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

#Remove grib files
rm ${fcnv2path}
rm ${pwpath}
rm ${gcpath}
rm ${aupath}

#Upload to S3 bucket
${aws} s3 cp ${fcnv2path}.nc s3://noaa-oar-mlwp-data/FOUR_v200_GFS/${YEAR}/${MONTH}${DAY}/FOUR_v200_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06.nc --profile noaa
${aws} s3 cp ${pwpath}.nc s3://noaa-oar-mlwp-data/PANG_v100_GFS/${YEAR}/${MONTH}${DAY}/PANG_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06.nc --profile noaa
${aws} s3 cp ${gcpath}.nc s3://noaa-oar-mlwp-data/GRAP_v100_GFS/${YEAR}/${MONTH}${DAY}/GRAP_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06.nc --profile noaa
${aws} s3 cp ${aupath}.nc s3://noaa-oar-mlwp-data/AURO_v100_GFS/${YEAR}/${MONTH}${DAY}/AURO_v100_GFS_${YEAR}${MONTH}${DAY}${HH}_f000_f240_06.nc --profile noaa

${python_contours} contouring.py ${YEAR} ${MONTH} ${DAY} ${HH} fcnv2 gfs ${fcnv2path}.nc ${geojson_path}
${python_contours} contouring.py ${YEAR} ${MONTH} ${DAY} ${HH} pw gfs ${pwpath}.nc ${geojson_path}
${python_contours} contouring.py ${YEAR} ${MONTH} ${DAY} ${HH} gc gfs ${gcpath}.nc ${geojson_path}
${python_contours} contouring.py ${YEAR} ${MONTH} ${DAY} ${HH} au gfs ${aupath}.nc ${geojson_path}

${python_contours} winds.py ${YEAR} ${MONTH} ${DAY} ${HH} fcnv2 gfs ${fcnv2path}.nc ${geojson_path}
${python_contours} winds.py ${YEAR} ${MONTH} ${DAY} ${HH} pw gfs ${pwpath}.nc ${geojson_path}
${python_contours} winds.py ${YEAR} ${MONTH} ${DAY} ${HH} gc gfs ${gcpath}.nc ${geojson_path}
${python_contours} winds.py ${YEAR} ${MONTH} ${DAY} ${HH} au gfs ${aupath}.nc ${geojson_path}

for file in ${geojson_path}/${YEAR}${MONTH}${DAY}_${HH}/*/*.geojson.simple; do
   filename="${file%%.simple*}"
   filename="${file%%.geojson*}"
   basename=$(basename "$filename")
   if [[ "$file" == *windbarb* ]]; then
	   ${tippecanoe} --drop-rate=0 --read-parallel -l "${YEAR}${MONTH}${DAY}_${HH}_${basename}" -e "${filename}_tiles" --minimum-zoom=2 --maximum-zoom=4 --no-tile-compression "${file}" --force
   else
       ${tippecanoe} --read-parallel -l "${YEAR}${MONTH}${DAY}_${HH}_${basename}" -e "${filename}_tiles" --minimum-zoom=2 --maximum-zoom=4 --no-tile-compression "${file}" --force
   fi
done

find ${geojson_path}/ -type f -name "*.geojson" -delete
find ${geojson_path}/ -type f -name "*.simple" -delete

cd ${geojson_path}
zip -r -0 ${YEAR}${MONTH}${DAY}_${HH}.zip ${YEAR}${MONTH}${DAY}_${HH}
scp ${YEAR}${MONTH}${DAY}_${HH}.zip ${aiweather_address}:/mnt/data1/aiweather/geojsons/
ssh ${aiweather_address} "unzip /mnt/data1/aiweather/geojsons/${YEAR}${MONTH}${DAY}_${HH}.zip -d /mnt/data1/aiweather/geojsons/"
