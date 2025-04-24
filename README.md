# [ICLR 2025 Spotlight] TabWak

We finally cleaned the code.

TODO:
- [ ] Docker support  
- [ ] Add pre-trained checkpoints

This is the repository for **TabWak**: A watermark for Tabular Diffusion Models.

The backbone model of TabWak is based on [Tabsyn](https://github.com/amazon-science/tabsyn/tree/main). Therefore, the installation and usage of TabWak are similar to Tabsyn. The following installation steps are based on Tabsyn's instructions.


---

## 🛠 Installing Dependencies

**Python version**: 3.10

### Step 1: Create Environment

```bash
conda create -n tabsyn python=3.10
conda activate tabsyn
```

### Step 2: Install PyTorch

Using `pip`:

```bash
pip install torch torchvision torchaudio
```

Or via `conda`:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Dependencies for GOGGLE

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

### Step 5: Install Quality Metric Dependencies (synthcity)

Create another environment for the quality metric:

```bash
conda create -n synthcity python=3.10
conda activate synthcity

pip install synthcity
pip install category_encoders
```

---

## 📦 Preparing Datasets

### Using the Datasets from the Paper

Download the raw dataset:

```bash
python download_dataset.py
```

Process the dataset:

```bash
python process_dataset.py
```

---

## 🏋️‍♀️ Training Models

For Tabsyn, use the following commands for training:

1. Train the VAE model first:

    ```bash
    python main.py --dataname [NAME_OF_DATASET] --method vae --mode train
    ```

2. After the VAE is trained, train the diffusion model:

    ```bash
    python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
    ```

---

## 💧 Watermarking During Sampling

To watermark the data during the sampling process, run:

```bash
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample --steps 1000 --with_w [Name_of_Watermark] --num_samples 5000 
```

**`[Name_of_Watermark]` options**: `treering`, `GS`, `TabWak`, `TabWak*`

---

## 🔍 Watermark Detection

For watermark detection, use:

```bash
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode detect --steps 1000 --with_w [Name_of_Watermark] --num_samples 5000
```
