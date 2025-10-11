<p align="center">
<h1 align="center">Map It Anywhere (MIA): Empowering Bird’s Eye View Mapping using Large-scale Public Data</h1>
<h3 class="is-size-5 has-text-weight-bold" style="color: orange;" align="center">
    NeurIPS 2024 Dataset and Benchmark Track
</h3>
  <p align="center">
    <a href="https://cherieho.com/"><strong>Cherie Ho*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/tonyjzou/"><strong>Jiaye (Tony) Zou*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/omaralama/"><strong>Omar Alama*</strong></a>
    <br>
    <a href="https://smj007.github.io/"><strong>Sai Mitheran Jagadesh Kumar</strong></a>
    ·
    <a href="https://github.com/chychiang"><strong>Benjamin Chiang</strong></a>
    ·
    <a href="https://www.linkedin.com/in/taneesh-gupta/"><strong>Taneesh Gupta</strong></a>
    ·
    <a href="https://sairlab.org/team/chenw/"><strong>Chen Wang</strong></a>
    <br>
    <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~./katia/"><strong>Katia Sycara</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
    <br>
  </p>

</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2407.08726">Paper</a> | <a href="https://MapItAnywhere.github.io/">Project Page</a> | <a href="https://mapitanywhere-dataengine.hf.space">Data Engine Demo</a> | <a href="https://mapitanywhere-mapper.hf.space">Map Prediction Demo</a></h3>
  <div align="center"></div>

<p align="center">
  <img src="assets/mia_gif.gif" />
</p>


![Map It Anywhere (MIA)](/assets/mia_pull_fig.png "Map It Anywhere (MIA)")


## Table of Contents
  - [Get Your Own Map Prediction Data: Using the MIA Data Engine](#get-map-prediction-data-using-the-mia-data-engine)
  - [Use Our Map Prediction Data: Downloading the MIA dataset](#get-our-map-prediction-data-downloading-the-mia-dataset)
  - [Let's Predict Maps! Set up Mapper Env](#lets-predict-maps-setting-up-mapper-environment)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Acknowledgement](#acknowledgement)


## Get Map Prediction Data: Using the MIA data engine

### 0. Setting up the environment
0. Install docker by following the instructions on their [website](https://www.docker.com/get-started/)
1. Pull our docker image to set up the MIA data engine: 

        docker pull theairlab/mia:latest
2. Launch the container while mounting this repository to the container file system.

        docker run -v <PATH_TO_THIS_REPO>:/home/MapItAnywhere --network=host -it theairlab/mia:latest

### 1. Getting FPVs

The first stage of the MIA data engine is to get the first person view (FPV) images.
To get started, if you'd prefer a quick demonstration, you can use the example configuration `mia/conf/mia_quick.yaml`, which is set up with a few small areas to let you experience the full pipeline quickly. To use your own locations, copy the example configuration file from `mia/conf/example.yaml` and modify the cities list to include your desired locations.
Feel free to explore the other well-documented FPV options in the configuration file.

Second, you need to acquire an access token for the [Mapillary API](https://www.mapillary.com/developer/api-documentation).

Once configuration is done and you have your token simply run the following from inside your docker container with working dir set to this repo:

    cd /home/MapItAnywhere
    python3.9 -m mia.fpv.get_fpv --cfg mia/conf/<YOUR_CONFIG>.yaml --token <MLY_TOKEN>

That's it ! The engine will now automatically fetch, filter, and process your FPV images. You may get a few errors specifying that some images were unable to be fetched due to permission limitations. That is normal and the engine will continue.

Once all your locations have been downloaded, you will see that parquet files, images, and raw_images, have been populated in your `dataset_dir` for each location. You can now move on to getting BEVs.

### 2. Getting BEVs
Once you have the FPV parquet dataframes downloaded, you are now ready to fetch and generate the BEV smenatic maps. 

Edit the documented bev options in your configuration file to suit your use case. The defaults are tuned to what we used to produce the MIA datasets and you can use them as is.

You may also want to edit the stylesheet in `mia/bev/styles/mia.yml` used for rendering BEVs. Namely, the `driving_side` and `infer_sidewalks` options should be updated depending on the regions you are pulling from. For urbanized areas, set `infer_sidewalks=True`, for rural, set it to False. 

Once configuration is done simply run the following from inside your docker container with working dir set to this repo:

    python3.9 -m mia.bev.get_bev --cfg mia/conf/<YOUR_CONFIG>.yaml

The data engine will now fetch, process, and save the semantic masks.

You now have FPV-BEV pairs with associated metadata and camera parameters !

**Note** to get satellite imagery for comparison you must first download it by toggling the store_sat option in the configuration and setting up a google earth project.

### 3. (Optional) Visualize your data
You can visualize a few samples using the tool `mia/misc_tools/vis_samples.py`. 

From inside the container with working dir set to this repo, run:

    python3.9 -m mia.misc_tools.vis_samples --dataset_dir <PATH_TO_DATASET_SEE_CONFIG_YML> --locations <LOCATION_OF_INTEREST>

If successful, the script will generate a PDF called `compare.pdf` in the location directory. Upon openning you should see the metadata, FPVs, and BEVs of a few samples of the dataset. 


## Get Our Map Prediction Data: Downloading the MIA dataset
Refer to [mia/dataset.md](mia/dataset.md) for instructions.

## Let's Predict Maps! Setting Up Mapper Environment

### Install using pip
You can install all requirements using pip by running:

    pip install -r mapper/requirements.txt

### Use Docker
To use Mapper using Docker, please follow the steps:
1. Build the docker image `mapper/Dockerfile` by running: 
        
        cd mapper/
        docker build -t mapper:release mapper
2. Launch the container while mounting this repository to the container file system.
    
        docker run -v <PATH_TO_THIS_REPO>:/home/mapper --network=host -it --gpus=all mapper:release

### Development with docker-compose (persist changes)

If you want to develop inside a container and have all edits persist on your host filesystem, use the provided `docker-compose.yml` which bind-mounts the project folder into the container. This is the recommended workflow for iterative development.

1. Build and start the service (rebuild after changing the Dockerfile):

```bash
docker compose build
docker compose up -d
```

2. Open a shell inside the running container:

```bash
docker compose exec mapper bash
```

3. Verify persistence from inside the container:

```bash
# inside the container
cd /workspace
echo "hello from container" > hello_from_container.txt
exit

# back on the host
ls -la /home/bevlog-2/Documents/MapItAnywhere/hello_from_container.txt
cat /home/bevlog-2/Documents/MapItAnywhere/hello_from_container.txt
```

The file created inside the container should appear on the host path above. The `docker-compose.yml` mounts the repo root at `/workspace` in the container so edits are immediate and persistent.

Notes and gotchas:
- Permissions: the image creates a user `user` (uid 1000). If your host user has a different uid/gid you may see permission differences. You can add `user: "${UID}:${GID}"` to the `docker-compose.yml` service (and export UID/GID in your shell) if needed.
- Cache and datasets: the compose file maps host cache and dataset directories into the container (`/home/user/.cache` and `/home/user/datasets`) to avoid re-downloading large files.
- Alternatives: if you prefer not to bind-mount the project, you can use named volumes or create/commit images (`docker commit`) but those approaches are less convenient for iterative code edits.

If you want, I can also add a short troubleshooting subsection about common permission fixes or add the `user: "${UID}:${GID}"` option to `docker-compose.yml` and document how to use it.

Quick helper script

We also provide a small helper script to simplify common actions (build/up/down/shell/run/logs):

```bash
./scripts/run_container.sh build
./scripts/run_container.sh up
./scripts/run_container.sh shell
```

This wrapper uses the repository's `docker-compose.yml` and forwards commands to `docker compose`. Use `UID`/`GID` environment variables when calling `up` if you need to align file ownership between host and container.

## Local development modifications

This repository in this workspace has a few small, local changes to make iterative development inside a container easier. These changes are intended for local development and are safe to revert if you prefer the original upstream behavior.

Summary of local edits made in this copy:

- `docker-compose.yml`: the service `mapper` was updated to bind-mount the project root from the host into the container at `/workspace` so edits inside the container persist on the host. The cache mount was remapped to `/home/user/.cache` to match the container user's HOME.
- `scripts/run_container.sh`: helper script added to simplify common `docker compose` operations (build, up, down, shell, run, logs). Executable at `./scripts/run_container.sh`.
- `README.md`: added the "Development with docker-compose (persist changes)" section and a short "Quick helper script" snippet referencing `scripts/run_container.sh` (this file you're reading now).

How to revert these local changes:

1. Restore the original `docker-compose.yml` from upstream (if you have the original file saved) or remove the bind-mount lines and restore the previous cache path.
2. Delete the helper script:

```bash
rm -f scripts/run_container.sh
```

3. Manually revert the README edits or replace `README.md` with the upstream copy.

If you'd like, I can:
- Add `user: "${UID}:${GID}"` to `docker-compose.yml` to force the container to run as the host user (helps permissions).
- Create a revert commit that returns these files to the upstream state instead of manual deletion.

Tell me which of the optional actions you'd like me to do next.

## Training

### Pre-train with MIA Dataset
To pretrain using our paper configuration simply run:

    python -m mapper.mapper data.split=<PATH TO SPLIT FILE> data.data_dir=<PATH TO MIA DATASET>

### Finetune with NuScenes Dataset
To finetune using NuScenes Dataset with our paper configuration, run:

    python -m mapper.mapper -cn mapper_nuscenes training.checkpoint=<PATH TO PRETRAINED MODEL> data.data_dir=<PATH TO NUSCENES DATA> data.map_dir=<PATH TO GENERATED NUSCENES MAP>

## Reproduction
#### Dataset Setup
**MIA**: Follow download instructions in [Downloading the MIA Dataset](#downloading-the-mia-dataset)

**NuScenes**: Follow the data generation instructions in [Mono-Semantic-Maps](https://github.com/tom-roddick/mono-semantic-maps?tab=readme-ov-file#nuscenes). To match the newest available information, we use v1.3 of the NuScenes' map expansion pack. 

**KITTI360-BEV**: Follow the KITTI360-BEV dataset instructions in [SkyEye](https://github.com/robot-learning-freiburg/SkyEye?tab=readme-ov-file#skyeye-datasets)

#### Inference
To generate MIA dataset prediction results(on test split), use:

    python -m mapper.mapper data.split=<PATH TO SPLIT FILE> data.data_dir=<PATH TO MIA DATASET> training.checkpoint=<TRAINED WEIGHTS> training.eval=true
*To specify location, add `data.scenes` in the argument. For example, for held-out cities `data.scenes="[pittsburgh, houston]"`*

To Generate NuScenes dataset prediction results(on validation split), use:

    python -m mapper.mapper -cn mapper_nuscenes training.checkpoint=<PATH TO PRETRAINED MODEL> data.data_dir=<PATH TO NUSCENES DATA> data.map_dir=<PATH TO GENERATED NUSCENES MAP> training.eval=true

To Generate KITTI360-BEV dataset prediction results (on validation split), use:

    python -m mapper.mapper -cn mapper_kitti training.checkpoint=<PATH TO PRETRAINED MODEL> data.seam_root_dir=<PATH TO SEAM ROOT> data.dataset_root_dir=<PATH TO KITTI DATASET> training.eval=true

## Inference
We have also provided a script in case you want to map a custom image. To do so, first set up the environment, then run the following:

    python -m mapper.customized_inference training.checkpoint="<YOUR CHECKPOINT PATH>" image_path="<PATH TO YOUR IMAGE>" save_path="<PATH TO SAVE THE OUTPUT>"

## Trained Weights
We have hosted trained weights for Mapper model using MIA dataset on Huggingface. [Click Here](https://huggingface.co/mapitanywhere/mapper) to download.

## License
The FPVs were curated and processed from Mapillary and have the same CC by SA license. These include all images files, parquet dataframes, and dump.json. The BEVs were curated and processed from OpenStreetMap and has the same Open Data Commons Open Database (ODbL) License. These include all semantic masks and flood masks. The rest of the data is licensed under CC by SA license.

Code is licensed under CC by SA license.

## Acknowledgement
We thank the authors of the following repositories for their open-source code:
- [OrienterNet](https://github.com/facebookresearch/OrienterNet)
- [Map Machine](https://github.com/enzet/map-machine)
- [Mono-Semantic-Maps](https://github.com/tom-roddick/mono-semantic-maps)
- [Translating Images Into Maps](https://github.com/avishkarsaha/translating-images-into-maps)
- [SkyEye](https://github.com/robot-learning-freiburg/SkyEye)

## Citation

If you find our paper, dataset or code useful, please cite us:

```bib
@inproceedings{ho2024map,
title = {Map It Anywhere (MIA): Empowering Bird's Eye View Mapping using Large-scale Public Data},
author = {Ho, Cherie and Zou, Jiaye and Alama, Omar and Kumar, Sai Mitheran Jagadesh and Chiang, Benjamin and Gupta,
Taneesh and Wang, Chen and Keetha, Nikhil and Sycara, Katia and Scherer, Sebastian},
year = {2024},
booktitle = {Advances in Neural Information Processing Systems},
url = {https://arxiv.org/abs/2407.08726},
code = {https://github.com/MapItAnywhere/MapItAnywhere}
}
```