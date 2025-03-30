## Illumination Correction Using Docker - Installation and Usage

### Script Information
These scripts are a modified version of the code from the following repository:  
[https://github.com/farkkilab/image_processing/tree/main/pipeline/0_illumination_correction](https://github.com/farkkilab/image_processing/tree/main/pipeline/0_illumination_correction)  
**Author:** Cruz Francisco Osuna Aguirre (cruzosuna2003@gmail.com)

---

## Docker Setup

### 1. Add Your User to the Docker Group
To avoid using `sudo` with every Docker command, add your user to the Docker group. Replace `$USER` with your actual username (e.g., `john`).
```bash
sudo usermod -aG docker $USER
```

### 2. Restart Your Session
Apply the changes by running:
```bash
newgrp docker
```

### 3. Verify Docker Access
Check if Docker is running correctly:
```bash
docker run hello-world
```

### 4. Build the Docker Image
Before running the script, build the Docker image:
```bash
docker build -t mybasic-image .
```

---

## Running Illumination Correction

### 1. Using an External Device
Run the following command, replacing the placeholders with appropriate values:
```bash
sudo docker run \
  --privileged -it -m <RAM>g --cpus=<CORES> \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="/path/to/external-device",target=/mnt/external \
  --mount type=bind,source="/path/to/input-data",target=/data/input \
  --mount type=bind,source="/path/to/output-data",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image /scripts/BaSiC_run.sh
```

- Replace `<RAM>` with the amount of RAM to allocate (e.g., `32`).
- Replace `<CORES>` with the number of CPU cores to allocate (e.g., `8`).
- Modify the paths to match your system setup.

#### Example: Running on an External HDD (e.g., "Mice" Drive)
```bash
sudo docker run \
  --privileged -it -m 32g --cpus=8 \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="/media/cruz-osuna/Mice",target=/mnt/external \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS/",target=/data/input \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image /scripts/BaSiC_run.sh
```

---

### 2. Using Internal Storage
#### Step 1: Build the Docker Image (if not done already)
```bash
sudo docker build -t mybasic-image .
```

#### Step 2: Run the Container
```bash
sudo docker run --privileged -it -m <RAM>g --cpus=<CORES> \
  --mount type=bind,source="$(pwd)",target=/data mybasic-image bash
```

- Replace `<RAM>` with the required memory allocation (e.g., `32g`).
- Replace `<CORES>` with the number of CPU cores to allocate (e.g., `8`).

#### Step 3: Execute the Script Inside the Container
Once inside the container, navigate to the working directory and execute the script:
```bash
cd /data  # Navigate to the directory where the script is located
bash BaSiC_run.sh  # Run the script to process the images
```

---

These instructions ensure a smooth execution of the illumination correction process using Docker, regardless of whether the data is stored on an external device or internal storage.

