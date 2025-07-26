# Beyond Pairwise Graphs: Hypergraph Representation of scRNA-seq Data for Modeling Cell–Cell Structure with Gumbel-Softmax

This repository implements a simple pipeline for **imputing missing values in single-cell RNA-seq (scRNA-seq) data** using a **hypergraph** constructed via a Gumbel-Softmax stratage.

<p align="center">
  <img src="scHGCL.jpg"/>
</p>



### Usage

Run the script：
```bash
python train.py --data_path data/counts.csv --drop_rate 0.55 --p 0.85 --tau 0.7 --epochs 200
```

### Arguments

| Argument       | Description                                       | Default                       |
|----------------|---------------------------------------------------|-------------------------------|
| `--use_gpu`    | Whether to use GPU (1 for CUDA, 0 for CPU)        | `1`                           |
| `--data_path`  | Path to the input expression matrix               | `counts.csv` |
| `--drop_rate`  | Dropout simulation rate                           | `0.55`                        |
| `--p`          | Top-p threshold for hyperedge selection           | `0.85`                        |
| `--tau`        | Gumbel-Softmax temperature parameter              | `0.7`                         |
| `--epochs`     | Number of training epochs                         | `200`                         |

---
