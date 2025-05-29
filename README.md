# AI Weather Pipeline (AIWP Realtime)

**Contact:**  
jacob.t.radford@gmail.com

## General Info

### Real-Time Pipeline

Cron job initiates `run_gfs.sh` and `run_ifs.sh` twice daily.

`run_gfs.sh` and `run_ifs.sh`:

- Wait for initial conditions to become available  
- `ai-models-gfs` and `ai-models` download ICs and run each model  
- `grib2nc.py` converts output of each to NetCDFs  
- `awscli` transfers NetCDFs to the S3 bucket (NODD) with credentials in `.aws/`  

### Model Versions

- FourCastNetv2-small (not fine-tuned)  
- Pangu-Weather (not fine-tuned)  
- GraphCast Operational 0.25 (fine-tuned on IFS-HRES)  
- Aurora 0.25 (fine-tuned on IFS-HRES)  

### NOAA Open Data Dissemination (NODD) AWS S3 Bucket

Ken Fenton can probably help if any NODD problems arise (ken.fenton@noaa.gov)  
Address: `s3://noaa-oar-mlwp-data`  
Credentials and config in `/mnt/aiweathernas/aiwp-realtime/.aws/`

```
cp -r /mnt/aiweathernas/aiwp-realtime/.aws ~/.aws
```

Other things in the bucket generally not requiring modification:

- `parquet/` (reference files to treat dataset as ZARR - hasn't been updated recently)  
- `colab_resources/` (files used by `ai-models-gfs`)  
- `Derived/` (Storm parameters calculated for subset of data)  
- `README.txt` if you need to update documentation  

### Local CIRA Archive

The S3 bucket is mostly backed up on Shannon in two places:

- `/mnt/mlnas01/ai-models/` (FourCastNetv2 and Pangu-Weather)  
- `/mnt/aiweathernas/ai-models-new/` (Aurora and GraphCast)  

Additionally, older data used by the TC team is in:  
- `/mnt/aiweathernas/ai-models/`

This latter directory can probably be deleted, but I did not want to risk interrupting TC work  
and there may be slight discrepancies due to changes in how incident solar radiation was calculated.  
In the event of any discrepancies, the version on the S3 bucket should be considered the most recent/valid.

## Instructions to Set Up from Scratch

```
cd /mnt/aiweathernas/
git clone https://github.com/jacob-t-radford/aiwp-realtime.git
cd aiwp-realtime
```

### Environment Configurations

```bash
conda create --name aiwp_realtime python=3.11
conda activate aiwp_realtime
pip install ai-models-gfs
pip install ai-models-graphcast-gfs
pip install ai-models-fourcastnetv2-gfs
pip install ai-models-panguweather-gfs
pip install ai-models-aurora-gfs
pip install git+https://github.com/deepmind/graphcast.git
pip install py3nvml
pip install awscli
pip install jax[cuda12]==0.5.0
pip install dm-haiku==0.0.13
pip install pygrib
conda deactivate

conda create --name aiwp_realtime_ifs python=3.11
conda activate aiwp_realtime_ifs
pip install ai-models
pip install ai-models-graphcast
pip install ai-models-fourcastnetv2
pip install ai-models-panguweather
pip install ai-models-aurora
pip install git+https://github.com/deepmind/graphcast.git
pip install py3nvml
pip install awscli
pip install jax[cuda12]==0.5.0
pip install dm-haiku==0.0.13
pip install pygrib

aws s3 cp --recursive --no-sign-request s3://noaa-oar-mlwp-data/colab_resources/fcnv2 /mnt/aiweathernas/aiwp-realtime/assets/fcnv2/
aws s3 cp --recursive --no-sign-request s3://noaa-oar-mlwp-data/colab_resources/pw /mnt/aiweathernas/aiwp-realtime/assets/pw/
aws s3 cp --recursive --no-sign-request s3://noaa-oar-mlwp-data/colab_resources/au /mnt/aiweathernas/aiwp-realtime/assets/au/
aws s3 cp --recursive --no-sign-request s3://noaa-oar-mlwp-data/colab_resources/gc /mnt/aiweathernas/aiwp-realtime/assets/gc/

conda deactivate
```

### Code Modifications  

#### In ~/anaconda3/envs/aiwp_realtime/lib/python3.11/site-packages/ai_models_gfs/model.py (GFS environment)

```
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])
```

#### In ~/anaconda3/envs/aiwp_realtime_ifs/lib/python3.11/site-packages/ai_models/model.py (IFS environment)  

```
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[0])
```

#### In ~/anaconda3/envs/aiwp_realtime_ifs/lib/python3.11/site-packages/ai_models_fourcastnetv2/model.py

```
checkpoint = torch.load(checkpoint_file, map_location=self.device)
# To:
checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
```

#### Update environment and cwd paths in run_gfs.sh

```
#Path to GFS environment (Currently assumes home directory)  
aiwp_realtime_env_path=~/anaconda3/envs/aiwp_realtime
#Path to working directory
aiwp_realtime_cwd_path=/mnt/aiweathernas/aiwp-realtime
```

#### Update environment and cwd paths in run_ifs.sh  

```
#Path to IFS environment (Currently assumes home directory)  
aiwp_realtime_env_path=~/anaconda3/envs/aiwp_realtime_ifs
#Path to working directory
aiwp_realtime_cwd_path=/mnt/aiweathernas/aiwp-realtime
```

### Cron

```
crontab -e
```

Add the following lines:

```
30 3,15 * * * bash /mnt/aiweathernas/aiwp-realtime/run_gfs.sh > /mnt/aiweathernas/aiwp-realtime/gfs.log 2>&1
30 7,19 * * * bash /mnt/aiweathernas/aiwp-realtime/run_ifs.sh > /mnt/aiweathernas/aiwp-realtime/ifs.log 2>&1
```

## Visualization Pipeline

This section includes instructions for producing vector tiles to be uploaded to the CIRA web server for interactive visualization.  

- Read NetCDF files
- Generate Matplotlib contours
- Convert contours to geojson files with geojsoncontour
- Simplify geojson files with Mapshaper
- Convert simplified geojsons to vector tiles with Tippecanoe
- Upload vector tiles to web server

### Environment Configuration

**The visualization code tends to be very sensitive to versioning.**

```bash
conda create --name aiwp_realtime_contouring python=3.11
conda activate aiwp_realtime_contouring
```




