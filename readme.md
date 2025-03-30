# BMF: t-CycIF Image Processing Pipeline

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-TBD-blue)](LICENSE)

Generic readme file generated with IA, pending improvement and corrections.

![Pipeline Overview](https://via.placeholder.com/800x200.png?text=Workflow+Diagram+-+Under+Construction)

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features
### Core Modules
1. **Illumination Correction**  
   Docker-based BaSiC implementation for shading correction
2. **Image Registration**  
   Multi-cycle alignment with Ashlar workflow
3. **Interactive Analysis**  
   Napari Viewer with cell quantification and Voronoi tools
4. **Stardist Segmentation**  
   Nuclear/cell segmentation with TensorFlow backend
5. **Quantification**  
   Parallelized intensity measurement and metadata handling

---

## Installation
### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Conda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Linux (recommended)

### 1. Illumination Correction
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Build image
docker build -t mybasic-image .
```

### 2. Image Registration
```bash
conda env create -f image_registration.yml
conda activate image_registration
```

### 3. Napari Viewer
```bash
conda env create -f napari-env.yml
conda activate napari-env
```

### 4. Stardist Segmentation
```bash
# Option 1: YAML file
conda env create -f stardist_env.yml

# Option 2: Shell script
chmod +x install_stardist.sh && ./install_stardist.sh
```

### 5. Quantification
```bash
conda env create -f quantification.yml
conda activate quantification
```

## Usage
### Illumination Correction (Docker)
```bash
docker run --privileged -it -m 32g --cpus=8 \
  --mount type=bind,source="/path/to/input",target=/data/input \
  --mount type=bind,source="/path/to/output",target=/data/output \
  mybasic-image /scripts/BaSiC_run.sh
```

### Image Registration
```bash
conda activate image_registration
python stitching.py -c 8  # Use 8 CPU threads
```

### Interactive Analysis (Napari)
```bash
conda activate napari-env
python napari_viewer.py
```
- GUI Features: Multichannel visualization, shape annotation, Voronoi diagrams, metadata display

### Stardist Segmentation
```bash
conda activate stardist
python stardist_segmentation.py
```

### Quantification (Parallel)
```bash
conda activate quantification
./quantification_parallel.sh  # Processes 8 images concurrently
```

## Contributing
This project welcomes contributions! Key needs:
- Documentation improvements
- Unit tests
- Docker optimization
- Additional visualization tools

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See CONTRIBUTING.md (under development) for guidelines.

## License
TBD - Pending final license selection

## Contact
### Maintainers
- Cruz Osuna
- Pablo Siliceo Portugal

### Source Attribution
Adapted from Färkkilä Lab and LSP

