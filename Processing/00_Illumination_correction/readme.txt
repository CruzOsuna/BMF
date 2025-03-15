




# Run iluminattion correction on a external device


sudo docker run \
  --privileged -it -m 120g --cpus=4 \
  --mount type=bind,source="/media/cruz-osuna/Spatial",target=/mnt/external \
  --mount type=bind,source="$(pwd)",target=/data \
  mybasic-image bash


sudo chmod -R a+rw /media/cruz-osuna/Spatial  # Si hay errores de acceso


cd /data


./BaSiC_run.sh





# Run ilum,ination correction on a internal device


sudo docker build -t mybasic-image .

sudo docker run --privileged -it -m 120g --cpus=4 --mount type=bind,source="$(pwd)",target=/data mybasic-image bash 

cd /data

bash BaSic_run.sh # run bash script to process images 

