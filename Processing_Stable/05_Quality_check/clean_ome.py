import os

def rename_files(directory):
    """
    Renombra archivos en el directorio dado eliminando ".ome" del nombre sin afectar 
    el formato ".tif" ni el nombre previo al ".ome".
    """
    for filename in os.listdir(directory):
        if filename.endswith(".ome.tif"):
            new_filename = filename.replace(".ome", "")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f'Renombrado: {filename} -> {new_filename}')

if __name__ == "__main__":
    dir_path = input("Ingresa la ruta del directorio: ").strip()
    if os.path.isdir(dir_path):
        rename_files(dir_path)
    else:
        print("Error: El directorio especificado no existe.")
