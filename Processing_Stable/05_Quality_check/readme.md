# Quality check

Source: https://labsyspharm.github.io/cylinter/

1) Setup CyLinter

# Miniconda instalation
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -f ~/miniconda3/miniconda.sh
```

# Create the Cylinter enviroment

```
conda create -n cylinter -c conda-forge -c labsyspharm cylinter=0.0.50 napari=0.4.18 vispy=0.12.1 pydantic=1.10.14 lxml=5.2.1 -y
conda activate cylinter
pip install lxml_html_clean
```


# Running Cylinter

`cylinter <input_dir>/cylinter_config.yml`


# Internal running

- Mice-4NQO
`cylinter /media/cruz-osuna/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml`

- Mice-P53
`cylinter /media/cruz-osuna/Mice/CycIF_mice_p53/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml`

Pendientes de especificar paths...

- Human-2024
`cylinter /media/cruz-osuna/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml`

- Human-2025
`cylinter /media/cruz-osuna/Mice/CycIF_mice_4NQO/5_QC/Cylinter/INPUT_DIR/cylinter_config.yml`
