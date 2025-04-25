import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Pending to translate to english :)

# Configuración personalizable
archivo1 = "/media/cruz/Mice/CycIF_mice_p53/8_Results/Datasets/0_Raw_data/3_FA2664P53_2_5_3_FA2664P53_2_5.csv"
archivo2 = "/media/cruz/Mice/CycIF_mice_p53/4_Quantification/output/3_FA2664P53_2_5_3_FA2664P53_2_5.csv"
output_dir = "/home/cruz/Escritorio/image_registration_temporal/3_FA2664P53_2_5"
marcadores = ['TIM3', 'CD3', 'CD11c', 'FOXP3', 'PAN-CK', 'PD-1', 'CD163',
              'PD-L1', 'CD8a', 'Vimentin', 'MHC-II', 'IBA1', 'NKG2D', 'gH2AX',
              'Ki67', 'CD20', 'E-cadherin', 'CD4', 'Neutrophil elastase']

os.makedirs(output_dir, exist_ok=True)

# Cargar y combinar datos
df1 = pd.read_csv(archivo1).assign(Fuente='Raw Image')
df2 = pd.read_csv(archivo2).assign(Fuente='Corrected lighting image')
datos_combinados = pd.concat([df1, df2], ignore_index=True)

# Configuración de estilo
sns.set(style="whitegrid", font_scale=0.8)
paleta = ["#1f77b4", "#ff7f0e"]

# Crear y guardar los plots
plt.figure(figsize=(20, 25))
cols = 3
rows = int(np.ceil(len(marcadores) / cols))

for i, marcador in enumerate(marcadores, 1):
    plt.subplot(rows, cols, i)
    sns.violinplot(
        x='Fuente',
        y=marcador,
        hue='Fuente',  # Añadido para nuevo formato
        data=datos_combinados,
        palette=paleta,
        cut=0,
        inner="quartile",
        bw_adjust=0.2,  # Parámetro actualizado
        legend=False      # Evita leyenda redundante
    )
    plt.title(marcador, weight='bold', pad=12)
    plt.ylabel('Intensidad', fontsize=9)
    plt.xlabel('')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=7)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(output_dir, 'comparacion_marcadores.png'), dpi=600, bbox_inches='tight')
plt.close()

print(f"Los plots se han guardado en: {output_dir}")