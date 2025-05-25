# BMF: t-CycIF Image Processing Pipeline

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-TBD-blue)](LICENSE)

> A modular and scalable pipeline for processing and analyzing t-CycIF multiplexed microscopy images, integrating illumination correction, registration, segmentation, and spatial analysis.

![Pipeline Overview](https://github.com/CruzOsuna/BMF/blob/main/workflow.png)

---

## Table of Contents

- [Features](#features)
  - [Processing Modules](#processing-modules)
  - [Analysis Modules](#analysis-modules)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [References](#references)

---

## Features

### Processing Modules
1. **Illumination Correction**  
   Dockerized BaSiC implementation for robust shading and background correction.
2. **Image Registration**  
   Accurate multi-cycle alignment using the Ashlar workflow.
3. **Interactive Analysis**  
   Napari viewer integration with custom widgets for visualization and annotation.
4. **Stardist Segmentation**  
   Deep learning-based nuclear/cell segmentation powered by a TensorFlow backend.
5. **Quantification**  
   Fast, parallelized feature quantification with integrated metadata handling.

### Analysis Modules
1. **Diversity Indices**  
   Computes spatial diversity indices and microenvironment statistics.
2. **STalign**  
   Aligns t-CycIF cellular data with H&E histopathology images for multimodal integration.

---

## Installation

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Conda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Linux OS (Ubuntu 24.04.2 LTS recommended)

> ⚠️ Windows support is currently experimental and not fully tested.

---

## Contributing

We welcome contributions from the community! You can help us by:
- Enhancing documentation
- Adding unit tests
- Optimizing Docker builds
- Developing additional visualization modules

### Getting Started:
1. Fork this repository
2. Create a new feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to your fork (`git push origin feature-name`)
5. Open a Pull Request

> See `CONTRIBUTING.md` (coming soon) for detailed contribution guidelines.

---

## License

**TBD** — License to be finalized. Contributions will be subject to the final license.

---

## Contact

**Maintainers:**
- **Cruz Osuna**
- **Pablo Siliceo Portugal**

For questions, suggestions, or collaboration inquiries, please reach out via GitHub issues or pull requests.

**Source Attribution:**  
Some code sections are adapted from the [Färkkilä Lab](https://github.com/farkkilalab) and [LSP](https://github.com/labsyspharm) repositories.

---

## References

### Illumination Correction
- Peng, T. *et al.* (2017). A BaSiC tool for background and shading correction of optical microscopy images. *Nature Communications*, **8**, 14836.

### Image Registration
- Muhlich, J. L. *et al.* (2022). Stitching and registering highly multiplexed whole-slide images of tissues and tumors using ASHLAR. *Bioinformatics*, **38**(19), 4613–4621.

### Interactive Analysis
- Chiu, C. L. & Clack, N. (2022). Napari: a multi-dimensional image viewer for the research community. *Microscopy and Microanalysis*, **28**(S1), 1576–1577.

### Stardist Segmentation
- Schmidt, U. *et al.* (2018). Cell detection with star-convex polygons. *MICCAI*.
- Weigert, M. *et al.* (2020). Star-convex polyhedra for 3D object detection. *WACV*.
- Weigert, M. & Schmidt, U. (2022). Nuclei segmentation and classification in histopathology. *ISBIC*.

### Quantification
- Schapiro, D. *et al.* (2022). MCMICRO: a scalable, modular pipeline for multiplexed tissue imaging. *Nature Methods*, **19**(3), 311–315.
