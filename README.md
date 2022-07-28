# Motion Policy Networks
This repo has the expert data generation infrastructure and Pytorch implementation of [MPiNets](https://mpinets.github.io/).

<img src="assets/readme1.gif" width="256" height="172" title="readme1">  |  <img src="assets/readme2.gif" width="256" height="172" title="readme2">

## Installation
The easiest way to install the code here is to build the attached docker container. We use Docker for several reasons:

1. For the data generation pipeline, many of our dependencies cannot be
   installed easily via PyPI
   - [OMPL](https://ompl.kavrakilab.org/) requires a lot of system dependencies and then building from source
   - [Geometric Fabrics](https://sites.google.com/nvidia.com/geometric-fabrics) are not open source, but are included with [Nvidia
     Omniverse](https://www.nvidia.com/en-us/omniverse/).
2. Docker makes it easy to manage the entire system environment, e.g. the CUDA
   runtime, the various upstream dependencies, etc.

If you have a strong need to build this repo on your host machine, you can follow the same steps as are outlined in the [Dockerfile](docker/Dockerfile). If you have your own expert data generation pipeline or intend to use our publicly available datasets, you only need to install the learning dependencies (the different areas should be well-documented in the Dockerfile). However, if you want to run the data generation pipeline, you will either need to download Nvidia Isaac Sim to get access to Geometric Fabrics or build the Docker container we include.

In order to build the Docker container included in this repo, you will need access
to the Isaac Sim docker container (our container is built on top of it). You can find detailed instructioned on how
to do this
[here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_advanced.html#isaac-sim-app-install-advanced),
but we provide summarized instructions here as well.

First you will need to join
the [Nvidia Developer Program](https://developer.nvidia.com/developer-program).
After joining, you can generate your NGC API key [as described here](https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-keyget).

With your NGC API key, you must log in to NGC from your host computer. As your
username, use the string `$oauthtoken`. As your password, use your NGC API key.
To log in, use the command
```
sudo docker login nvcr.io
```
Next, clone this repo using:
```
git clone https://gitlab-master.nvidia.com/srl/motion-policy-networks
```
Navigate inside the repo (e.g. `cd motion-policy-networks`) and build the docker with
```
docker build --tag mpinets --network=host --file docker/Dockerfile .
```
After this is built, you should be able to launch the docker using this command
(be sure to use the correct paths on your system for the `/PATH/TO/THE/REPO` arg)
```
docker run --interactive --tty --rm --gpus all --network host --env DISPLAY=unix$DISPLAY --env XAUTHORITY --env NVIDIA_DRIVER_CAPABILITIES=all --env "ACCEPT_EULA=Y" --volume /PATH/TO/THE/REPO:/root/mpinets mpinets /bin/bash -c 'export PYTHONPATH=/root/mpinets:$PYTHONPATH; git config --global --add safe.directory /root/mpinets; /bin/bash'
```
In order to run any GUI-based code in the docker, be sure to add the correct
user to `xhost` on the host machine. You can do this by running `xhost
+si:localuser:root` in another terminal on the host machine.

Our suggested development setup would be to have two terminals open, one
running the docker (use this one for running the code) and another editing
code on the host machine. The `docker run` command above will mount your
checkout of this repo into the docker, allowing you to edit the files from
either inside the docker or on the host machine.

## Usage

All usage described below must happen inside the docker container. For all the commands below,
assume that they should be run inside the docker container.

### Using our pregenerated data
We provide the dataset of Hybrid Expert trajectories that we used to train our
model in Motion Policy Networks. [You can download this data here]().

If you are using the Docker container, you will need to mount this data after
downloading it. If the path this data is downloaded to is
`/PATH/TO/THE/DATA`, you can run the Docker with a similar command to the
one above:
```
docker run --interactive --tty --rm --gpus all --network host --env DISPLAY=unix$DISPLAY --env XAUTHORITY --env NVIDIA_DRIVER_CAPABILITIES=all --env "ACCEPT_EULA=Y" --volume /PATH/TO/THE/REPO:/root/mpinets --volume /PATH/TO/THE/DATA:/data mpinets /bin/bash -c 'export PYTHONPATH=/root/mpinets:$PYTHONPATH; /bin/bash'
```
### Data Generation
If you would like to generate the data yourself, we provide scripts we used to
generate our dataset. These scripts are designed
to work in a cloud-based system, but generating a large dataset will require
some data management. To generate the data in Motion Policy Networks, we used a
cluster of 80 server nodes running these scripts in parallel.

You can use [gen_data.py][data_pipeline/gen_data.py] to generate the data. This
file should have a self-explanatory help string, which you can access with
`data_pipeline/gen_data.py --help`.

Some examples of data generation:

To visualize some examples generated from a set of dressers where the robot
reaches between drawers, you can do
```
cd /root/mpinets/data_pipeline
python3 gen_data.py dresser test-environment
```
To run the a similar visualization test where the trajectories all start or end
with a collision-free neutral pose, you can use
```
cd /root/mpinets/data_pipeline
python3 gen_data.py dresser test-environment --neutral
```
To test the data pipeline (i.e. including how the data is saved on disk)
and generate a small dataset within randomized tabletop environments using
only task-oriented poses, you can run
```
cd /root/mpinets/data_pipeline
mkdir -p data/tabletop/task_oriented/1
python3 gen_data.py tabletop test-pipeline /root/mpinets/data_pipeline/data/tabletop/task_oriented/1/
```
To generate a large dataset of trajectories in 2x2 cubby environments where each trajectory begins or
ends with a neutral configuration.
```
cd /root/mpinets/data_pipeline
mkdir -p data/cubby/neutral/1
python3 gen_data.py cubby full-pipeline /root/mpinets/data_pipeline/data/cubby/neutral/1/ --neutral
```

### Data Cleaning
After generating the data, you will need to clean it and merge it into a single
dataset. In the Motion Policy Network dataset, we use the following
proportions:

- Cubby Neutral: 1 / 12
- Cubby Task Oriented: 1 / 12
- Merged Cubby Neutral: 1 / 12
- Merged Cubby Task Oriented: 1 / 12
- Dresser Neutral: 1 / 6
- Dresser Task Oriented: 1 / 6
- Tabletop Neutral: 1 / 6
- Tabletop Task Oriented: 1 / 6

We provide a script [process_data.py](data_pipeline/process_data.py) that can
take the output of `gen_data.py` and clean it up. After running the `full-pipeline` mode
of`gen_data.py`, it will produce a file in the specified directory called
`all_data.py`. This is all the data of that multi-process run, which has data
in a single environmental class where either every trajectory starts or ends
with a neutral pose or does not. The next step is to clean this data (e.g. keep
only the problems with a hybrid solution), downsize the dataset to have matching
sizes across problem types, split the dataset into train, validation, test groups,
and merge all these individual datasets across various scene types.

These methods are all documented in `process_data.py --help`.

After running the various stages of this script, you will have a dataset that
looks like
```
FINAL_DATA_PATH/
  train/
    train.hdf5
  val/
    val.hdf5
  test/
    test.hdf5

```
### Training Motion Policy Networks

Once you have the data, either by generating the dataset or by downloading
ours, you can train the model using [run_training.py](mpinets/run_training.py).
This script expects a configuration file and we provide a sample at
[jobconfig.yaml](jobconfig.jaml). Look through this configuration file and
modify the necessary paths before training.

Then, to run training, use:
```
python3 mpinets/run_training.py jobconfig.yaml
```
We use Weights and Biases for logging training jobs, but you can disable this
logger using:
```
python3 mpinets/run_training.py jobconfig.yaml --no-logging
```
## License
This work is released with the MIT License.

## Citation
If you find this work useful in your research, please cite:

	@inproceedings{fishman2022mpinets,
	  title={Motion Policy Networks},
	  author={Fishman, Adam and Murali, Adithyavairavan and Eppner, Clemens and Peele, Bryan and Boots, Byron and Fox, Dieter},
	  year={2022}
	}
