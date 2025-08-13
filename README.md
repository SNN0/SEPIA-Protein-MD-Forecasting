# SEPIA: A Developing Project for Molecular Dynamics Trajectory Forecasting

SEPIA is a work-in-progress project for predicting molecular dynamics (MD) trajectories using a Graph Neural Network (GNN). The project provides a full pipeline to extract structural features from MD simulations, train a predictive model, forecast future frames, and analyze the results.

### Project Status

This is a developing project and is actively being worked on. The current version contains a refactored and modular codebase for the core functionalities.

### Key Features

* **Data Preprocessing**: Extracts windowed features from MD trajectory files (DCD) and PDB topology files.
* **Feature Extraction**: Generates key structural features for each frame, including coordinates, radius of gyration, contact maps, backbone torsions, and secondary structure.
* **Model Architecture**: Utilizes a GraphRNN model composed of GCN blocks and a GRU layer to process sequences of graphs.
* **Training**: Employs a curriculum learning strategy to dynamically adjust loss function weights during training.
* **Composite Loss Function**: A weighted loss function combines multiple structural metrics, including RMSD, contact map similarity, radius of gyration, torsions, secondary structure cross-entropy, and Wasserstein distance on distances.
* **Iterative Prediction**: Forecasts new trajectory frames iteratively, with an optional teacher-forcing mechanism that uses ground truth data for a specified probability.
* **Comprehensive Analysis**: Compares predicted and original trajectories across various metrics, such as RMSD, RMSF, Rg, secondary structure content, and dihedral angle distributions.

### Modular Project Structure

The codebase is organized into a modular structure to separate concerns and improve maintainability.

SEPIA/
├── core/
│   ├── dataset.py      # Data loading and batching logic.
│   ├── model.py        # The GraphRNN model architecture.
│   └── losses.py       # Weighted and composite loss functions.
├── scripts/
│   ├── preprocess.py   # Extracts and saves features from raw MD data.
│   ├── train.py        # Main script for training the model.
│   ├── predict.py      # Predicts a new trajectory using a trained model.
│   ├── analyze.py      # Compares predicted and original trajectories.
│   └── inspect.py      # A utility to verify the content of feature files.
├── data/
│   └── ...             # Directory for input/output data.
├── models/
│   └── ...             # Directory for trained model weights.
└── analysis_results/
└── ...             # Directory for analysis outputs.

### Quick Start Guide

The following steps outline a typical workflow using the project scripts.

#### 1. Preprocess Data

Extract features from an MD trajectory (`.dcd`) and its topology file (`.pdb`) into a pickle file.

```bash
python scripts/preprocess.py --pdb protein.pdb --dcd trajectory.dcd --out features.pkl --window 30

```
*Description*: This command extracts features using 30-frame sliding windows and saves the output to features.pkl.

#### 2. Train the Model

Train the GraphRNN model using the preprocessed features.

```bash

python scripts/train.py --features features.pkl --output my_model_dir --epochs 120 --hidden 512
```

*Description*: This script trains the model for 120 epochs with data from `features.pkl` and saves the best model to the `my_model_dir` directory.

#### 3. Predict a Trajectory
Use the trained model to forecast a new trajectory.

```bash

python scripts/predict.py --features features.pkl --model my_model_dir/best.pth --pdb protein.pdb --out predicted_trajectory.dcd --steps 10 --teacher_forcing 0.5
```
*Description*: This command predicts 10 new windows of a trajectory using the model `my_model_dir/best.pth`. It uses teacher forcing with a 50% probability.

#### 4. Predict a Trajectory
Compare the predicted trajectory with the original trajectory and generate plots and a summary report.

```bash

python scripts/analyze.py --pred predicted_trajectory.dcd --orig original_trajectory.dcd --pdb protein.pdb --out_dir analysis_results
```
*Description*: This script compares `predicted_trajectory.dcd` and `original_trajectory.dcd`, saving all analysis outputs to the `analysis_results` directory.

Optional: Inspect Features
This utility can be used to perform a sanity check on the preprocessed feature file.

```bash

python scripts/inspect.py --features features.pkl --samples 5
```
*Description*: This command prints shapes, data ranges, and details for 5 sample windows from `features.pkl`.

#### Dependencies
The project requires the following Python libraries:

* `torch`

* `torch_geometric`

* `mdtraj`

* `numpy`

* `tqdm`

* `scikit-learn`

* `scipy`

* `matplotlib`

* `seaborn`

* `PyYAML`