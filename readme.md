# BMF: t-CycIF Image Processing Pipeline

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-TBD-blue)](LICENSE)


![Pipeline Overview](https://github.com/CruzOsuna/BMF/blob/main/workflow.png)

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
   Napari Viewer with widgets
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

Verified support for Ubuntu 24.04.2 LTS, Windows support has not been verified for all scripts.



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
Some sections of the code were adapted from the Färkkilä Lab and LSP repositories.

## References
1. **Illumination Correction**  
   Peng, T., Thorn, K., Schroeder, T., Wang, L., Theis, F. J., Marr, C., & Navab, N. (2017). A BaSiC tool for background and shading correction of optical microscopy images. Nature communications, 8(1), 14836.
   
3. **Image Registration**  
   Muhlich, J. L., Chen, Y. A., Yapp, C., Russell, D., Santagata, S., & Sorger, P. K. (2022). Stitching and registering highly multiplexed whole-slide images of tissues and tumors using ASHLAR. Bioinformatics, 38(19), 4613-4621.
   
5. **Interactive Analysis**  
   Chiu, C. L., & Clack, N. (2022). Napari: a Python multi-dimensional image viewer platform for the research community. Microscopy and Microanalysis, 28(S1), 1576-1577.
   
7. **Stardist Segmentation**  
   Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell Detection with Star-convex Polygons. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.
   Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.
   
   Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy.
   The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020.
   Martin Weigert and Uwe Schmidt.
   
   Nuclei Instance Segmentation and Classification in Histopathology Images with Stardist.
   The IEEE International Symposium on Biomedical Imaging Challenges (ISBIC), Kolkata, India, March 2022.

9. **Quantification**  
   Schapiro, D., Sokolov, A., Yapp, C., Chen, Y. A., Muhlich, J. L., Hess, J., ... & Sorger, P. K. (2022). MCMICRO: a scalable, modular image-processing pipeline for multiplexed tissue imaging. Nature methods, 19(3), 311-315.

   



