#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# Inteligencia de Negocios - Remanente 2025
# Punto 1: Lectura y estructura del dataset UBER
# ============================================

import pandas as pd

path = r"C:\Users\julla\Downloads\Datos_1\uber.csv"
df = pd.read_csv(path)

print(" Dimensiones del dataset:", df.shape)
print("\n Primeras filas:")
display(df.head())

print("\n Información general:")
df.info()

print("\n Descripción estadística:")
display(df.describe(include='all'))


# In[2]:


# Bloque 2 — Diagnóstico de estructura y faltantes

# Conteo de valores faltantes por columna
print("\n Valores faltantes por columna:")
print(df.isnull().sum())

# Tipos de datos
print("\n Tipos de columnas:")
print(df.dtypes.value_counts())

# Porcentaje de faltantes
faltantes = df.isnull().mean().sort_values(ascending=False)
print("\n Porcentaje de datos faltantes:")
print((faltantes * 100).round(2))


# In[3]:


# ============================================
# Punto 2a: Cálculo de la distancia Haversine
# Con depuración de coordenadas fuera de rango
# ============================================

import numpy as np
import pandas as pd

# Copia de seguridad del DataFrame original
df_clean = df.copy()

# ------------------------------------------------
# DEPURACIÓN DE COORDENADAS
# ------------------------------------------------
print(" Depurando coordenadas fuera de rango...")

# Rango válido global
cond_valid_global = (
    df_clean['pickup_latitude'].between(-90, 90) &
    df_clean['dropoff_latitude'].between(-90, 90) &
    df_clean['pickup_longitude'].between(-180, 180) &
    df_clean['dropoff_longitude'].between(-180, 180)
)

# Opcional: rango específico de NYC
cond_valid_nyc = (
    df_clean['pickup_latitude'].between(40.5, 41.0) &
    df_clean['dropoff_latitude'].between(40.5, 41.0) &
    df_clean['pickup_longitude'].between(-74.3, -73.5) &
    df_clean['dropoff_longitude'].between(-74.3, -73.5)
)

# Combinar condiciones (usa global o NYC según análisis)
df_valid = df_clean[cond_valid_global & cond_valid_nyc].copy()

# Reporte de limpieza
print(f"Registros originales: {len(df_clean):,}")
print(f"Registros válidos: {len(df_valid):,}")
print(f"Eliminados: {len(df_clean) - len(df_valid):,} "
      f"({(1 - len(df_valid)/len(df_clean))*100:.3f}%)")

# ------------------------------------------------
# FÓRMULA DEL SEMIVERSENO (HAVERSINE)
# ------------------------------------------------
R = 6371  # Radio de la Tierra en km

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos usando la fórmula del semiverseno (Haversine).
    Las coordenadas deben estar en grados decimales.
    Retorna la distancia en kilómetros.
    """
    # Conversión de grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Diferencias
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula del semiverseno
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ------------------------------------------------
# CÁLCULO DE DISTANCIAS
# ------------------------------------------------
df_valid['distance_km'] = haversine_distance(
    df_valid['pickup_latitude'],
    df_valid['pickup_longitude'],
    df_valid['dropoff_latitude'],
    df_valid['dropoff_longitude']
)

# Vista previa
print("\n Variable 'distance_km' creada correctamente.")
display(df_valid[['fare_amount', 'distance_km']].head())

# Estadísticas descriptivas
print("\n Estadísticas de la distancia (dataset depurado):")
display(df_valid['distance_km'].describe())


# In[4]:


# ============================================
# Análisis detallado de variables geoespaciales
# ============================================

geo_cols = ['pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude']

print(" Rango de coordenadas:")
display(df[geo_cols].describe())


# In[5]:


# Verificación de dispersión (valores extremos)

# Coordenadas fuera del rango esperado
mask_outliers = (
    (df['pickup_latitude'] < 40.5) | (df['pickup_latitude'] > 41.0) |
    (df['dropoff_latitude'] < 40.5) | (df['dropoff_latitude'] > 41.0) |
    (df['pickup_longitude'] < -74.5) | (df['pickup_longitude'] > -73.5) |
    (df['dropoff_longitude'] < -74.5) | (df['dropoff_longitude'] > -73.5)
)

outliers_geo = df[mask_outliers]
print(f" Registros fuera del rango geográfico: {len(outliers_geo)} ({len(outliers_geo)/len(df)*100:.2f}%)")
display(outliers_geo.head(5))


# In[6]:


# ============================================
# Visualización de coordenadas geográficas limpias
# ============================================

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 7))

# Usamos el dataset depurado (df_valid)
plt.scatter(df_valid['pickup_longitude'], df_valid['pickup_latitude'],
            s=1, alpha=0.4, label='Pickup', color='steelblue')
plt.scatter(df_valid['dropoff_longitude'], df_valid['dropoff_latitude'],
            s=1, alpha=0.4, label='Dropoff', color='darkorange')

plt.title(" Distribución geográfica de viajes (Área NYC)", fontsize=12)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.legend(markerscale=5)

# Ajuste de límites centrado en Nueva York
plt.xlim(-74.3, -73.5)
plt.ylim(40.5, 41.0)

plt.grid(alpha=0.3, linestyle='--')
plt.show()


# In[7]:


# ============================================
# Distribución de viajes sobre mapa base de NYC
# ============================================

import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Crear GeoDataFrame con pickups y dropoffs
pickup_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(df_valid['pickup_longitude'], df_valid['pickup_latitude']),
    crs="EPSG:4326"
)
dropoff_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(df_valid['dropoff_longitude'], df_valid['dropoff_latitude']),
    crs="EPSG:4326"
)

# Convertir a la proyección Web Mercator (necesaria para contextily)
pickup_gdf = pickup_gdf.to_crs(epsg=3857)
dropoff_gdf = dropoff_gdf.to_crs(epsg=3857)

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))
pickup_gdf.plot(ax=ax, markersize=2, alpha=0.3, color='royalblue', label='Pickup')
dropoff_gdf.plot(ax=ax, markersize=2, alpha=0.3, color='darkorange', label='Dropoff')

# Agregar mapa base
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Personalización
ax.set_title(" Distribución geográfica de viajes Uber (NYC sobre mapa base)", fontsize=12)
ax.legend()
ax.set_axis_off()
plt.show()


# In[8]:


# ============================================
# Visualización de densidad geográfica (Hexbin)
# ============================================

import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))
plt.hexbin(df_valid['pickup_longitude'], df_valid['pickup_latitude'],
           gridsize=80, cmap='plasma', bins='log', alpha=0.8)
plt.colorbar(label='Densidad logarítmica de pickups')
plt.title(" Densidad geográfica de viajes Uber (Área NYC)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.xlim(-74.3, -73.5)
plt.ylim(40.5, 41.0)
plt.grid(alpha=0.3, linestyle='--')
plt.show()


# In[9]:


# ============================================
# Hexbin con mapa base NYC (enfocado en zonas con datos)
# ============================================

import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# --- Coordenadas originales (EPSG:4326) ---
x = df_valid['pickup_longitude']
y = df_valid['pickup_latitude']

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))

# Hexbin optimizado
hb = ax.hexbin(
    x, y,
    gridsize=250,         # resolución adecuada para áreas urbanas
    cmap='magma_r',       # buen contraste sobre fondo gris
    bins='log',
    alpha=0.75
)
cb = fig.colorbar(hb, ax=ax, label='Densidad logarítmica de pickups')

# 🔹 Ajuste de límites centrados en área activa (Manhattan + Brooklyn + JFK)
ax.set_xlim(-74.1, -73.75)
ax.set_ylim(40.55, 40.9)

# Agregar mapa base reproyectado
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=12)

# Títulos y estilo
ax.set_title(" Densidad geográfica de viajes Uber (Área urbana NYC, Hexbin + mapa base)", fontsize=13)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()


# In[10]:


# ============================================
# Mapa de calor de densidad (versión rápida con submuestreo)
# ============================================

import seaborn as sns
import matplotlib.pyplot as plt

# Submuestreo para acelerar el KDE
sample = df_valid.sample(n=10000, random_state=42)

plt.figure(figsize=(7,7))
sns.kdeplot(
    x=sample['pickup_longitude'],
    y=sample['pickup_latitude'],
    fill=True, cmap='rocket', thresh=0.05, levels=50
)

plt.title("Mapa de calor de densidad de pickups (muestra 10k puntos)")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.xlim(-74.3, -73.5)
plt.ylim(40.5, 41.0)
plt.grid(alpha=0.3, linestyle='--')
plt.show()


# In[11]:


# ============================================
# Mapa de calor de densidad (KDE + mapa base NYC, centrado en área urbana)
# ============================================

import seaborn as sns
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# Submuestreo para acelerar el KDE (más puntos = mejor resolución)
sample = df_valid.sample(n=20000, random_state=42)

# Convertimos a GeoDataFrame y reproyectamos
gdf_sample = gpd.GeoDataFrame(
    sample,
    geometry=gpd.points_from_xy(sample['pickup_longitude'], sample['pickup_latitude']),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Extraer coordenadas reproyectadas
x = gdf_sample.geometry.x
y = gdf_sample.geometry.y

# Crear figura
fig, ax = plt.subplots(figsize=(8, 8))

# KDE con ajustes visuales óptimos
sns.kdeplot(
    x=x, y=y,
    fill=True, cmap='magma_r', thresh=0.02, levels=200,
    alpha=0.8, ax=ax
)

# Agregar mapa base (CartoDB Positron)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=12)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronOnlyLabels, zoom=12)

# 🔹 Límites reajustados: centrados en NYC (Manhattan, Brooklyn, Queens, JFK)
ax.set_xlim([-8255000, -8205000])
ax.set_ylim([4955000, 5005000])

# Estilo final
ax.set_title(" Mapa de calor de densidad de pickups Uber (Área urbana NYC, muestra 20k puntos)", fontsize=13, pad=12)
ax.set_xlabel("Longitud (EPSG:3857)")
ax.set_ylabel("Latitud (EPSG:3857)")
ax.grid(alpha=0.25, linestyle='--')

plt.tight_layout()
plt.show()


# In[12]:


# Comprobación de simetría de coordenadas (valores negativos o invertidos)
print("Coordenadas positivas (potenciales errores):")
for col in geo_cols:
    positive_vals = (df[col] > 0).sum()
    print(f"{col}: {positive_vals} ({positive_vals/len(df)*100:.3f}%)")


# In[13]:


# ============================================
# Punto 2.b: Matriz de correlaciones (Pearson vs Spearman)
# ============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Dataset filtrado o depurado ---
# (si ya tienes df_valid con distance_km y fare_amount)
df_corr = df_valid.copy()

# Seleccionamos solo variables numéricas
num_cols = df_corr.select_dtypes(include='number')

# --- Matriz de correlaciones Pearson ---
corr_pearson = num_cols.corr(method='pearson')

plt.figure(figsize=(6,5))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title(" Matriz de correlaciones (Pearson)", fontsize=13)
plt.show()

# Valor específico entre tarifa y distancia
pearson_value = corr_pearson.loc["fare_amount", "distance_km"]
print(f" Correlación Pearson entre fare_amount y distance_km: {pearson_value:.3f}")

# --- Matriz de correlaciones Spearman ---
corr_spearman = num_cols.corr(method='spearman')

plt.figure(figsize=(6,5))
sns.heatmap(corr_spearman, annot=True, cmap='crest', center=0, fmt=".2f")
plt.title(" Matriz de correlaciones (Spearman)", fontsize=13)
plt.show()

# Valor específico entre tarifa y distancia
spearman_value = corr_spearman.loc["fare_amount", "distance_km"]
print(f" Correlación Spearman entre fare_amount y distance_km: {spearman_value:.3f}")

# --- Interpretación automática (texto explicativo) ---
print("\n Interpretación:")
if pearson_value > 0.8 and spearman_value > 0.8:
    print("Ambos métodos muestran una correlación fuertemente positiva: "
          "a mayor distancia, mayor tarifa promedio.")
else:
    print("Las correlaciones difieren ligeramente. Pearson mide relación lineal, "
          "mientras Spearman captura relaciones monótonas incluso no lineales. "
          "Diferencias pueden deberse a outliers o a no linealidades en las tarifas cortas/largas.")


# In[14]:


# ============================================
# Punto 2.c (actualizado): Limpieza de outliers y análisis de fare_amount
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Filtrado básico de outliers y valores inválidos ---

# Eliminamos tarifas negativas o nulas (no válidas)
df_valid = df_valid[df_valid['fare_amount'] > 0]

# Eliminamos outliers extremos (p.ej. top 1%)
p99 = df_valid['fare_amount'].quantile(0.99)
df_valid = df_valid[df_valid['fare_amount'] < p99]

print(f" Registros luego del filtrado: {len(df_valid):,}")
print(f" Tarifa máxima después del filtrado: USD {df_valid['fare_amount'].max():.2f}")

# --- Estadísticas descriptivas ---
fare = df_valid['fare_amount']
stats_fare = fare.describe().round(2)

print("\n Estadísticas descriptivas de la variable 'fare_amount' (filtrada):\n")
display(stats_fare)

# --- Histograma y distribución ---
plt.figure(figsize=(8,5))
sns.histplot(fare, bins=60, kde=True, color='royalblue', alpha=0.7)
plt.title("Distribución de la tarifa (fare_amount) - Dataset filtrado", fontsize=13)
plt.xlabel("Tarifa del viaje (USD)")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3, linestyle='--')
plt.show()

# --- Boxplot ---
plt.figure(figsize=(7,1.8))
sns.boxplot(x=fare, color='skyblue')
plt.title("Boxplot de la tarifa (fare_amount) - Dataset filtrado", fontsize=12)
plt.xlabel("Tarifa (USD)")
plt.grid(alpha=0.3, linestyle='--')
plt.show()

# --- Comentario automático ---
print("\n💬 Interpretación:")
print(f"- Luego del filtrado, la tarifa promedio es USD {stats_fare['mean']:.2f}, "
      f"con una mediana de USD {stats_fare['50%']:.2f}.")
print(f"- El rango plausible de tarifas es ahora {stats_fare['min']:.2f}–{stats_fare['max']:.2f} USD.")
print("- Se eliminaron registros negativos y valores extremos, "
      "dejando una distribución más realista y centrada en viajes urbanos típicos.")
print("- La distribución sigue mostrando una asimetría positiva, "
      "propia de los servicios de transporte con mayoría de trayectos cortos y pocos viajes largos.")


# In[15]:


# Ver cuántos registros están fuera de los bigotes:
Q1 = df_valid['fare_amount'].quantile(0.25)
Q3 = df_valid['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

outliers = df_valid[df_valid['fare_amount'] > upper_limit]
print(f"Número de outliers (tarifas > {upper_limit:.2f}): {len(outliers):,}")

# limitar el eje x o usar una escala logarítmica para que se aprecie la distribución completa
plt.figure(figsize=(7, 1.8))
sns.boxplot(x='fare_amount', data=df_valid, color='skyblue')
plt.xlim(0, 30)   # cortar visualmente el eje para evitar la línea negra
plt.title("Boxplot de fare_amount (zoom hasta 30 USD)")
plt.show()


# In[16]:


# ============================================
# Punto 2.d: Scatterplot fare_amount vs distance_km (versión mejorada)
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Usamos el dataset filtrado ---
df_scatter = df_valid.copy()

# --- Scatterplot general ---
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_scatter,
    x='distance_km',
    y='fare_amount',
    alpha=0.3, s=10, color='steelblue'
)
plt.title("Relación entre tarifa (fare_amount) y distancia (km)", fontsize=13)
plt.xlabel("Distancia del viaje (km)")
plt.ylabel("Tarifa del viaje (USD)")
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 20)      # Mejora: enfoque en viajes urbanos más frecuentes
plt.ylim(0, 55)      # Mejora: recorte de eje Y para evitar saturación visual
plt.show()

# --- Scatterplot con línea de tendencia (regresión lineal simple) ---
plt.figure(figsize=(8,6))
sns.regplot(
    data=df_scatter,
    x='distance_km',
    y='fare_amount',
    scatter_kws={'alpha':0.3, 's':10},
    line_kws={'color':'red', 'lw':2}
)
plt.title("Tarifa vs Distancia con línea de tendencia (dataset filtrado)", fontsize=13)
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.grid(alpha=0.3, linestyle='--')
plt.xlim(0, 20)
plt.ylim(0, 55)
plt.show()

# --- Comentario automático ---
corr_pearson = df_scatter[['fare_amount','distance_km']].corr().iloc[0,1]
print("\n Interpretación:")
print(f"- La correlación lineal (Pearson) entre tarifa y distancia es de {corr_pearson:.3f}.")
print("- Se observa una relación positiva: a mayor distancia, mayor tarifa promedio.")
print("- Sin embargo, se detectan algunas anomalías:")
print("   Tarifas muy altas en trayectos cortos (errores de GPS o recargos especiales).")
print("   Tarifas bajas en viajes largos (errores de medición o descuentos).")
print("- La nube de puntos muestra dispersión creciente con la distancia, "
      "lo que sugiere variabilidad adicional por tráfico, peajes o zona de destino.")
print("- La escala log-log opcional permite comprobar si la relación es proporcional "
      "en todo el rango de distancias (útil para validar la linealidad).")


# In[17]:


# ============================================
# Punto 2.e: Evaluación de outliers univariados (sin recorte adicional)
# ============================================

import numpy as np
import pandas as pd

# Función para calcular límites IQR y proporción de outliers
def check_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2.0 * IQR
    upper = Q3 + 2.0 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    prop = 100 * len(outliers) / len(df)
    print(f"{column}: {len(outliers):,} observaciones fuera del rango IQR "
          f"({prop:.2f}% del total). Rango teórico [{lower:.2f}, {upper:.2f}]")

check_outliers_iqr(df_valid, 'fare_amount')
check_outliers_iqr(df_valid, 'distance_km')


# In[18]:


# ============================================
# 2.e: Eliminación formal de outliers univariados (IQR truncado a 0)
#      y preparación del dataset "oficial" para el resto del TP
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Punto de partida: dataset ya depurado en pasos previos (2.c–2.d)
df_2c = df_valid.copy()

def iqr_bounds_nonneg(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = max(0, Q1 - 2.0 * IQR)   # las variables no pueden ser negativas
    upper = Q3 + 2.0 * IQR
    return lower, upper, Q1, Q3, IQR

# Calcular límites IQR para cada variable
fa_l, fa_u, fa_Q1, fa_Q3, fa_IQR = iqr_bounds_nonneg(df_2c['fare_amount'])
dk_l, dk_u, dk_Q1, dk_Q3, dk_IQR = iqr_bounds_nonneg(df_2c['distance_km'])

n_before = len(df_2c)

# Filtrado IQR en ambas variables (intersección)
mask_fare = (df_2c['fare_amount'] >= fa_l) & (df_2c['fare_amount'] <= fa_u)
mask_dist = (df_2c['distance_km'] >= dk_l) & (df_2c['distance_km'] <= dk_u)
df_2e = df_2c[mask_fare & mask_dist].copy()

n_after = len(df_2e)

print("Límites IQR aplicados (truncados a 0 cuando corresponde):")
print(f" - fare_amount: [{fa_l:.2f}, {fa_u:.2f}]  (Q1={fa_Q1:.2f}, Q3={fa_Q3:.2f}, IQR={fa_IQR:.2f})")
print(f" - distance_km: [{dk_l:.2f}, {dk_u:.2f}]  (Q1={dk_Q1:.2f}, Q3={dk_Q3:.2f}, IQR={dk_IQR:.2f})")
print(f"\nRegistros antes: {n_before:,} | después: {n_after:,} | removidos: {n_before - n_after:,} "
      f"({100*(n_before-n_after)/n_before:.2f}%)")

# Guardar nombres "oficiales" para siguientes pasos del TP
df_filtered_official = df_2e  # usar este en partición y modelado

# ============================================
# 2.f: Rehacer correlaciones e histograma de fare_amount (antes vs después)
# ============================================

# Correlación Pearson fare vs distance (antes/después)
corr_before = df_2c[['fare_amount','distance_km']].corr(method='pearson').iloc[0,1]
corr_after  = df_2e[['fare_amount','distance_km']].corr(method='pearson').iloc[0,1]

print("\nCorrelación Pearson fare_amount vs distance_km:")
print(f" - Antes (2.c): {corr_before:.3f}")
print(f" - Después (2.e): {corr_after:.3f}")

# Histograma comparativo de fare_amount (antes vs después)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

sns.histplot(df_2c['fare_amount'], bins=60, kde=True, ax=axes[0])
axes[0].set_title("fare_amount — antes (2.c)")
axes[0].set_xlabel("Tarifa (USD)")
axes[0].grid(alpha=0.3, linestyle='--')

sns.histplot(df_2e['fare_amount'], bins=60, kde=True, ax=axes[1])
axes[1].set_title("fare_amount — después (2.e)")
axes[1].set_xlabel("Tarifa (USD)")
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Texto de apoyo para el notebook
fa_stats_before = df_2c['fare_amount'].describe().round(2)
fa_stats_after  = df_2e['fare_amount'].describe().round(2)

print("\nResumen estadístico de fare_amount (antes vs después):")
print(pd.DataFrame({'antes_2c': fa_stats_before, 'después_2e': fa_stats_after}))


# In[19]:


# ============================================
# 2.f: Matriz de correlaciones (antes vs después del filtrado IQR)
# ============================================

import seaborn as sns
import matplotlib.pyplot as plt

# Variables numéricas de interés
numeric_cols = ['fare_amount', 'distance_km', 'pickup_latitude', 'pickup_longitude', 
                'dropoff_latitude', 'dropoff_longitude']

# Correlaciones antes y después
corr_before = df_2c[numeric_cols].corr(method='pearson')
corr_after  = df_2e[numeric_cols].corr(method='pearson')

# --- Visualización comparativa ---
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(corr_before, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Matriz de correlaciones — antes (2.c)")

sns.heatmap(corr_after, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Matriz de correlaciones — después (2.e)")

plt.tight_layout()
plt.show()


# In[ ]:




