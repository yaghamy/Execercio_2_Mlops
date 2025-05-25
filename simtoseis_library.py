import segyio
from scipy.stats import ks_2samp
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from os.path import join as pjoin
import shutil
import os
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor


def depth_signal_checking(simulation_data, seismic_data):
    """
    Ensures that the depth (Z) values in both simulation and seismic datasets are positive.
    Input:
        simulation_data: numpy array
                         Simulation data with columns [X, Y, Z, Property].
        seismic_data: numpy array
                      Seismic data with columns [X, Y, Z, (Property)].
    Output:
        Tuple:
            - simulation_data: numpy array with positive Z values.
            - seismic_data: numpy array with positive Z values.
    """
    
    simulation_data[:, 2] = np.abs(simulation_data[:, 2])
    seismic_data[:, 2] = np.abs(seismic_data[:, 2])
    
    print("Done!")
    
    return simulation_data, seismic_data


def simulation_nan_treatment(simulation, value=0, method='replace'):
    """
    Treats NaN values in the simulation dataset by either replacing them or removing affected cells.
    Input:
        simulation: numpy array
                    Simulation data with shape (n_samples, n_features), last column is the property.
        value: numeric (default=0)
               Value to replace NaNs with when method='replace'.
        method: string ('replace' or 'remove')
                - 'replace': replaces NaNs in the property column with the specified value.
                - 'remove': removes rows where the property column is NaN.
    Output:
        simulation: numpy array
                    Updated simulation data after NaN treatment.
    """
    
    initial_samples = simulation.shape[0]
    
    if method == "replace":
        print(f"Method: {method}")
        print(f"Shape Prior to NaN treatment: {simulation.shape}")
        print(f"Prior to NaN treatment:\n{simulation}")
        simulation[:, -1] = np.where(np.isnan(simulation[:, -1]), value, simulation[:, -1])
        print(f"Shape After NaN treatment: {simulation.shape}")
        print(f"After NaN treatment:\n{simulation}")
    else:
        print(f"Method: {method}")
        print(f"Shape Prior to NaN treatment: {simulation.shape}")
        print(f"Prior to NaN treatment:\n{simulation}")
        simulation = simulation[np.isnan(simulation[:, -1]) != True]
        print(f"Shape After NaN treatment: {simulation.shape}")
        print(f"After NaN treatment:\n{simulation}")

    final_samples = simulation.shape[0]

    if initial_samples != final_samples:
        residual = initial_samples - final_samples
        print(f"WARNING!\n{residual} CELLS WERE REMOVED!!")
        
    print("Done!")        

    return simulation


def simulation_data_cleaning(simulation_data=None, value_to_clean=None):
    """
    Cleans the simulation dataset by removing samples with a specified unwanted value.
    Input:
        simulation_data: numpy array
                         Simulation data with columns [X, Y, Z, Property].
        value_to_clean: numeric
                        Value in the property column to be removed (e.g., -99.0 for invalid samples).
    Output:
        simulation_data: numpy array
                         Cleaned simulation data with specified values removed.
    """
    
    original_data = simulation_data.shape[0]
    print(f"Original number of samples in simulation model: {original_data}")
    
    # Filter out the unwanted value
    simulation_data = simulation_data[simulation_data[:, -1] != value_to_clean]
    print(f"Final number of samples after cleaning: {simulation_data.shape[0]}")
    
    # Calculate and report the percentage of data removed
    percentage_loss = ((original_data - simulation_data.shape[0]) / original_data) * 100
    print(f"Percentage loss: {round(percentage_loss, 2)}%")
    
    print("Done!")
    
    return simulation_data
	


def plot_simulation_distribution(sim_array_xyzprop, bins=35, title=None):
    plt.figure(figsize=(8,6))
    plt.hist(sim_array_xyzprop[:, -1], bins=bins, color='steelblue', edgecolor='black')
    plt.title(title if title else "Distribuição da Propriedade")
    plt.xlabel("Valor da Propriedade")
    plt.ylabel("Frequência")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# ### Treinamento/Validação do Modelo de ML


def ML_model_evaluation(dados_simulacao=None, proporcao_treino=0.75, modelo="extratrees"):    
    """
    Treina e avalia um modelo de machine learning ExtraTreesRegressor com dados de simulação.
    Entrada:
        dados_simulacao: numpy array
                         Conjunto de dados com colunas [X, Y, Z, Propriedade], onde a última coluna é a variável alvo.
        proporcao_treino: float (padrão=0.7)
                          Proporção dos dados utilizada para o treinamento (o restante será usado para teste).
    Saída:
        sim_estimado: numpy array
                      Previsões do modelo para o conjunto de teste.
        y: numpy array
           Valores reais da propriedade para o conjunto completo (antes da divisão).
    """

    global X, y, ET, X_treino, X_teste, y_treino, y_teste, sim_estimado, dict_params

    # Separar atributos e variável alvo
    X = dados_simulacao[:,:-1]
    y = dados_simulacao[:,-1]

    # Dividir em conjuntos de treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, train_size=proporcao_treino, random_state=0)
    
    dict_params = {"n_estimators":100,"max_depth":20, "n_jobs":-1,"proporcao_treino":proporcao_treino}
    
    if modelo == "extratrees":    
        # Inicializar e treinar o ExtraTreesRegressor
        ET = ExtraTreesRegressor(n_estimators = dict_params["n_estimators"], max_depth = dict_params["max_depth"], n_jobs = dict_params["n_jobs"])
        ET.fit(X_treino, y_treino)

        # Prever nos conjuntos de treino e teste
        sim_estimado = ET.predict(X_teste)
        sim_treinado = ET.predict(X_treino)

    # Avaliar o desempenho do modelo
    tolerancia = 0.01

    # Métricas de desempenho no treino
    nrms_treino = (np.sqrt(mean_squared_error(sim_treinado, y_treino)) / np.std(y_treino)) * 100
    r2_treino = r2_score(y_treino, sim_treinado)
    mape_treino = np.mean(np.abs((sim_treinado - y_treino) / (y_treino + tolerancia)))

    print("Desempenho no conjunto de treino")
    print(f'Erro percentual absoluto médio: {round(mape_treino, 1)}%')
    print(f'NRMS: {round(nrms_treino, 1)}%')
    print(f'R²: {round(r2_treino, 2)}')

    # Métricas de desempenho no teste
    nrms_teste = (np.sqrt(mean_squared_error(sim_estimado, y_teste)) / np.std(y_teste)) * 100
    r2_teste = r2_score(y_teste, sim_estimado)
    mape_teste = np.mean(np.abs((sim_estimado - y_teste) / (y_teste + tolerancia)))

    print("Desempenho no conjunto de teste")
    print(f'Erro percentual absoluto médio: {round(mape_teste, 1)}%')
    print(f'NRMS: {round(nrms_teste, 1)}%')
    print(f'R²: {round(r2_teste, 2)}')

    print("Concluído!")

    return sim_estimado, y, nrms_teste, r2_teste, mape_teste, ET

def transfer_to_seismic_scale(modelo_treinado, dados_sismicos):
    coordenadas_sismicas = dados_sismicos[:, :3].copy()
    vetor_prop_sismica = modelo_treinado.predict(coordenadas_sismicas)
    prop_sismica_reshape = vetor_prop_sismica.reshape(len(vetor_prop_sismica), 1)
    sismica_estimada = np.hstack((coordenadas_sismicas, prop_sismica_reshape))
    print("Inferência concluída!")
    return vetor_prop_sismica, sismica_estimada

def calcular_residuos(dados_referencia, dados_estimados):
    residuos = dados_referencia[:, -1] - dados_estimados[:, -1]
    residuos_reshape = residuos.reshape(len(residuos), 1)
    dados_residuos = np.hstack([dados_estimados[:, :-1], residuos_reshape])
    print("Cálculo dos resíduos concluído!")
    return dados_residuos

def plot_seismic_slice(seismic_slice, title="Slice at Depth ~5000m", cmap='seismic'):
    vmin = seismic_slice[:, -1].min()
    vmax = seismic_slice[:, -1].max()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        seismic_slice[:, 0],
        seismic_slice[:, 1],
        c=seismic_slice[:, -1],
        cmap=cmap,
        norm=norm,
        s=10,
        edgecolors='none'
    )
    plt.colorbar(sc, label="Property/Amplitude")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
