## Instrucciones para corrección de iluminación usando Docker

Este documento proporciona instrucciones detalladas para ejecutar la corrección de iluminación en diferentes dispositivos de almacenamiento utilizando Docker.


# Docker Setup


1)Add your user to the Docker group (Replace $USER with your username (e.g., John)).

sudo usermod -aG docker $USER 


2) Close session 

newgrp docker


3) Verify Docker access

docker run hello-world



4) Build the Docker Image

docker build -t mybasic-image .



---


1. Illumination Correction on an External Device (Mice HDD in this example)

# Run the following command to start the Docker container with the corresponding folders:

sudo docker run \
  --privileged -it -m 32g --cpus=8 \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="/media/cruz-osuna/Mice",target=/mnt/external \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS/",target=/data/input \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image /scripts/BaSiC_run.sh

# Replace the information in RAM (-m 32g), the CPU (--cpus=8) and the paths to adapt the command to your system.



2. Illumination Correction on Internal Storage

# Build the Docker image if you haven't done so already:

sudo docker build -t mybasic-image .


# Run the container using internal storage:

sudo docker run --privileged -it -m 120g --cpus=4 --mount type=bind,source="$(pwd)",target=/data mybasic-image bash

# Replace the information in RAM (-m 32g), the CPU (--cpus=8) and the paths to adapt the command to your system.


# Inside the container, navigate to the working directory and execute the script:

cd /data  # Navigate to the directory where the script is located
bash BaSiC_run.sh  # Run the script to process the images


