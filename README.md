# Spectra To Structure : Deep Reinforcement Learning for Molecular Inverse Problem


## Setting up the environment

### Conda

The most convenient way to get the code up and running is by setting up a conda
environment and install the necesary packages in a new environment. The `yml` 
file can be found at [conda_environment.yml](./requirements/conda_requirements.yml). 

```
conda env create -f conda_environment.yml
conda activate spectra2structure
```

The above would create and activate a conda environment with the name spectra2structure.

### pip

After installing `python 3.8.3`, you can install all the dependensies. The dependendies can 
be found at [pip_requirements.txt](./requirements/pip_requirements.txt). 

```
python -m pip install -r pip_requirements.txt
```

## Directory Structure

```
.
├── data
│   └── RPNMRData
│       └── graph_conv_many_nuc_pipeline.datasets
├── requirements
├── source
│   ├── test
│   │   ├── environment
│   │   │   └── RPNMR
│   │   ├── test
│   │   └── utils
│   └── train
│       ├── environment
│       │   ├── RPNMR
│       ├── saved_models
│       └── utils
└── trainedModels
    ├── RPNMRModels
    └── crossvalidationModels
```

## Training the model with Training Mode On

If you have a slurm workload manager then the framework can be run by simply 
modifying and running the batch script provided in the `train` directory.
If not, then you can begin training by the following set of commands:

```
ulimit -n 40960
ray start --head --num-cpus=40 --num-gpus=2 --object-store-memory 50000000000
python parallel_agent.py
```

Change `--num-cpus` to the number of cores that you have available in your machine
and also change the value in `parallel_agent.py` for the variable `num_processes`.
`num_processes = 1.5 * num-cpus` is usually handlable by any machine.

The models would be saved inside the directory `saved_models` and the running 
loss values would be appended to the files `prior_log.txt` and `value_log.txt`
for the prior model and the value model respectively.

For each episode, the environment chooses a target spectra randomly from the 
training dataset and creates datapoint through MCTS Runs. This training dataset
can be found in the directory [data](./data/).
Already trainined model can be found in the directory [trainedModels](./trainedModels/).

## Testing the model

If you have a slurm workload manager then the framework can be run by simply 
modifying and running `batch.sh` file in the `test` directory. If not, 
you can begin testing on target spectra by the following set of commands:

```
ulimit -n 40960
ray start --head --num-cpus=40 --num-gpus=4 --object-store-memory 50000000000
python parallel_agent.py
ray stop
```
Change `--num-cpus` and `--num-gpus` to the number of cores and gpus available. 
Also change the value of variable `NUM_GPU` and `num_processes` in the file
`parallel_agent.py`.

The target spectra to best tested are specified by the array `smiles_idx` in the 
file `parallel_agent.py`. These indices are passed as a parameter to the function
`env.reset(idx)` present in the file 
[`environment/environment.py`](source/test/environment/environment.py) which then
chooses the target spectra from the test dataset based on the passed parameter.
To run test on custom molecules, one needs to only modify the function `reset` in
the environment class to load the target information appropriately.
After a series of episodes, the resulting guesses are pickled and stored for each
target spectra in the directory [`source/test/test/`](source/test/test/). 
The information of each episode completed can also be seen in the log_file 
generated in the `source/test/` directory.

One can also run `source/test/rescheck.py` from the `test` directory to see a 
summary of the testing where in, the accuracy and the closest molecule to each
target molecule(by rdkit similarity) are reported.

## Preparing the Forward Model

A well trained forward model is reqd to run the agent for training and testing.
You can train the model by running the batch script in the directory 
`source/forwardModel/batch.sh`, this training is tracked by wandb if you have 
configured with your account, the model would be saved in the directory
`source/forwardModel/checkpoints/best_model.00000000.state`. Copy this model to
the location: `trainedModels/RPNMRModels/best_model.00000000.state`
Once saved, you can go ahead with training and testing the inverse framework.


---

### Hyperparameters

Information about hyperparameters used for the experiments can be found in the file
[hyper_params.txt](./hyper_params.txt).

---










