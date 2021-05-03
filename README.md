# The Baseline Generator Framework for Deep Survival Analysis
- Deep Survival Learning for Alzheimer's Disease Prediction. 
- Framework for Explaining Deep Survival Analysis Models for Heterogenous Data (Simulated Data)

## Introduction
This GitHub repository is developed by Moritz Wagner. This project introduces the `Baseline Generator`. 
To make predictions interpretable, Shapley values depict a prominent choice. Yet, computing Shapley values relies on a specification of a baseline against which the original prediction is compared. 
To identify a suitable baseline is still an (seemingly) unsolved challenge. The literature refers to that as the `baseline selection problem`.

Within this project, I implemented the `Baseline Generator` to address this challenge within the setting of survival analysis.
While the general idea of this framework is likely applicable to more complex settings, I have limited myself to verify the framework on simulated data.


## Code Structure
This framework is structured as follows:
```bash
├── configs
├── notebooks
├── src
├── experiment.py
├── explainer.py
├── run.py
├── sksurv_train.py
└── tuner.py
```

### `configs`
Contains the config files for each model depending on the chosen data set.

### `notebooks`
Contains only some jupyter notebook files that are absolutely not important to consider.

### `src`
This folder stores all functions and (model) classes that are used throughout this work. The package contains different modules for each step in this pipeline. 

```bash
├── architectures
├── baselines
├── data
├── dsap
├── models
├── evaluator.py
├── helpers.py
├── integrated_gradients.py
├── occlusion.py
├── postprocessing.py
├── shapley_sampling.py
└── visualizer.py
```

### `experiment.py`
This file contains the class `DeepSurvExperiment`. This class builds on the general framework from `PyTorchLightning`.
The whole training procedure for all models/experiments is defined within this class.

This includes:
1. Training a Deep Cox-PH model on heterogenous data (image and tabular data). 
2. Evaluation of the model performance with concordance index [https://scikit-survival.readthedocs.io/en/latest/api/generated/sksurv.metrics.concordance_index_censored.html](c-index). 
2. Training the `Baseline Generator` to generate for each image the optimal and unique baseline.

### `explainer.py`
After the model is trained and the baselines are generated, we can calculate the attribution maps.
The file includes the computations for: 
1. Integrated Gradients 
2. Sampled Shapley values

Both methods are valid approximations for the true Shapley values. For each attribution method, the attribution maps are derived for three different samples which contain four observations each.

### `run.py`
This file is the main file where a specific run or experiment is set up. The following is included:
1. the config files are updated when arguments are passed via the command line
2. the model is initialized
3. the `MLFlowLogger` for logging is initialized 
4. the experiment class `DeepSurvExperiment` from `experiment.py` is initialized 
5. finally the model is trained as specified in `DeepSurvExperiment` and consequently validated on the test data.

The following command must be executed from the command line:
`python run.py --config configs/DATAFOLDER/<confi_file>.yaml --args`

### `sksurv_train.py`
The ADNI data consists of unstructured MRIs and structured tabular data. 
By executing this file, a Linear Cox-PH model from the package `sksurv` is trained on the tabular data only. 

### `tuner.py`
This file works almost equivalently to `run.py`. By executing this file, the model can be tuned with Hyperband from `Optuna`.
The following command must be executed from the command line:
`python tuner.py --config configs/DATAFOLDER/<confi_file>.yaml --args`


## Packaging 
* package `src` enables pip installation and symling creation
* fork the repository and execute `pip install -e .` from the parent tree.
* packaging allows also for updating the packages in the scripts via `importlib.reload()`

## Setup 

You can setup `Deep Survival` as follows: Note, all commands must be run from the parent level of the repository.

1. Install miniconda
2. Create a conda environment for python 3.7 (`conda create -n <env name> python=3.7`)
3. Clone this repository 
4. Install the required packages via `pip install -r requirements.txt`
5. Install the local package `src` as described above. 


## Run Experiments 
To train models in general, run `python run.py --config configs/DATAFOLDER/config_file.yaml --args` from the command line and from the parent level of this directory. 

### ADNI 
For running the experiments on the 2D-ADNI data run the following commands:
- image data only: `python split_images.py`
- tabular data only: `python split_sksurv.py` 
- image data and tabular data: `python split_imgtab.py`

All experiments are then conducted and evaluated with 5-fold cross-validation.

### Simulations 
For running the experiments on the simulated data, I distinguish between two simulation settings:

#### Simulation 1 (Colored Rectangles)
1. `python run.py --configs/SIM/coloredrectangles.yaml experiment_name --sim1 --run_name sim1`
    * trains survival model
    * trains the `Baseline Generator` and generates baseline images on the test data
2. Evaluate visually after which training step the `Baseline Generator` generates the best baselines. 
3. `python explainer.py --run_name sim1 --step <optimal_steps>`
    * computes and visualizes Integrated Gradients 
    * computes and visualizes sampled Shapley values

#### Simulation 2 (Colored Geometric Shapes)
1. `python run.py --configs/SIM/coloredshapes.yaml experiment_name --sim2 --run_name sim2`