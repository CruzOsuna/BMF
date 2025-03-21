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

### 1. Corrección de iluminación en un dispositivo externo (Spatial SSD)

Ejecuta el siguiente comando para iniciar un contenedor de Docker y montar las carpetas necesarias:

```bash
sudo docker run \
  --privileged -it -m 120g --cpus=8 \
  --mount type=bind,source="/media/cruz-osuna/Spatial",target=/mnt/external \
  --mount type=bind,source="/media/cruz-osuna/Spatial/CycIF_mice_p53/1_Registration/RCPNLS/",target=/data/input \
  --mount type=bind,source="/media/cruz-osuna/Spatial/CycIF_mice_p53/00_Illumination_correction/Output",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image bash
```

Luego, dentro del contenedor, sigue estos pasos:

```bash
cd /scripts  # Acceder al directorio donde está el script
chmod +x BaSiC_run.sh  # Dar permisos de ejecución si es necesario
./BaSiC_run.sh  # Ejecutar el script
```

---

### 2. Corrección de iluminación en un dispositivo externo (Mice HDD)

Ejecuta el siguiente comando para iniciar el contenedor de Docker con las carpetas correspondientes:

Nitro5-Cruz

```bash
sudo docker run \
  --privileged -it -m 120g --cpus=8 \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="/media/cruz-osuna/Mice",target=/mnt/external \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS/",target=/data/input \
  --mount type=bind,source="/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image /scripts/BaSiC_run.sh
```

Dentro del contenedor, sigue estos pasos:

```bash
cd /scripts  # Acceder al directorio donde está el script
chmod +x BaSiC_run.sh  # Dar permisos de ejecución si es necesario
./BaSiC_run.sh  # Ejecutar el script
```

---

Aorus-Cruz

```bash
sudo docker run \
  -it -m 30g --cpus=20 \
  -u $(id -u):$(id -g) \
  --mount type=bind,source="/media/cruz/Mice",target=/mnt/external \
  --mount type=bind,source="/media/cruz/Mice/CycIF_mice_4NQO/1_Stitching/Unstitched_images",target=/data/input \
  --mount type=bind,source="/media/cruz/Mice/CycIF_mice_4NQO/0_Iluminattion_Correction/output",target=/data/output \
  --mount type=bind,source="$(pwd)",target=/scripts \
  mybasic-image /scripts/BaSiC_run.sh
```

Dentro del contenedor, sigue estos pasos:

```bash
cd /scripts  # Acceder al directorio donde está el script
chmod +x BaSiC_run.sh  # Dar permisos de ejecución si es necesario
./BaSiC_run.sh  # Ejecutar el script
```

---







### 3. Corrección de iluminación en almacenamiento interno

1. Construye la imagen de Docker si no lo has hecho previamente:

```bash
sudo docker build -t mybasic-image .
```

2. Ejecuta el contenedor con el almacenamiento interno:

```bash
sudo docker run --privileged -it -m 120g --cpus=4 --mount type=bind,source="$(pwd)",target=/data mybasic-image bash
```

3. Dentro del contenedor, accede al directorio de trabajo y ejecuta el script:

```bash
cd /data  # Acceder al directorio donde está el script
bash BaSiC_run.sh  # Ejecutar el script para procesar las imágenes
```

---


