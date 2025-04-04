import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Configuración
INPUT_CSV = 'shannon_results.csv'
OUTPUT_IMAGE = 'shannon_lineplot.png'
FONT_SCALE = 1.5
FIG_SIZE = (12, 6)
DPI = 600

def plot_shannon_trend_with_error(csv_path):
    """Genera un gráfico de línea mostrando la tendencia del índice de Shannon con desviación estándar"""
    # Estilo de seaborn
    sns.set(style="whitegrid", font_scale=FONT_SCALE)

    # Cargar datos
    df = pd.read_csv(csv_path)

    # Extraer y ordenar columnas step
    step_columns = sorted(
        [col for col in df.columns if col.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )

    # Calcular la media y la desviación estándar del índice de Shannon para cada paso
    mean_values = df[step_columns].mean(axis=0)
    std_values = df[step_columns].std(axis=0)
    step_numbers = [int(col.split('_')[1]) for col in step_columns]

    # Crear gráfico
    plt.figure(figsize=FIG_SIZE)
    ax = sns.lineplot(x=step_numbers, y=mean_values.values, marker='o', linewidth=2, label='Índice de Shannon (promedio)')

    # Añadir la banda de error (desviación estándar)
    ax.fill_between(
        step_numbers,
        mean_values - std_values,
        mean_values + std_values,
        color='b', alpha=0.2, label='Desviación estándar'
    )

    # Etiquetas y título
    ax.set_xlabel('Número de paso', labelpad=10)
    ax.set_ylabel('Índice de Shannon (promedio)', labelpad=10)
    plt.title('Tendencia del Índice de Shannon a lo largo de los pasos', pad=20)

    # Notación científica en el eje Y
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Leyenda
    ax.legend()

    # Guardar gráfico
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=DPI)
    plt.close()
    print(f"Gráfico de tendencia guardado en: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_shannon_trend_with_error(INPUT_CSV)
