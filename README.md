# Passive Radar Processing Chain

Repositorio para simulación y procesamiento de una cadena de radar pasivo bistático en Python.

El proyecto permite:

- generar señales simuladas de referencia y vigilancia,
- cargar señales reales,
- aplicar canal y ruido opcional,
- filtrar clutter/direct path,
- aplicar ventaneo sobre la referencia,
- calcular la Cross-Ambiguity Function (CAF),
- ejecutar detección CA-CFAR.

El repositorio incluye:

- un archivo `environment.yml` para crear el entorno Conda,
- un archivo `pyproject.toml` para instalar el código como paquete Python.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```
###2. Crear el entorno Conda
'''bash
conda env create -f environment.yml
conda activate pr
'''

###3. Instalar el paquete en modo editable
'''bash
pip install -e .
'''
