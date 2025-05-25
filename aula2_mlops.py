
#!/usr/bin/env python
# coding: utf-8

### Importando Bibliotecas

import numpy as np
import simtoseis_library as sts
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import TwoSlopeNorm
import mlflow
import mlflow.sklearn

# ### Carregando os Dados

# Dados Treino
dados_treino = np.load("sim_slice.npy")
print(f"Shape dados de treino: {dados_treino.shape}")

# Dados de Inferência
dados_inferencia = np.load("seismic_slice.npy")
print(f"Shape dados de inferência: {dados_inferencia.shape}")

# Dados de Referência (Software Comercial)
dados_referencia_comercial = np.load("seismic_slice_GT.npy")
print(f"Shape dados de referência comercial: {dados_referencia_comercial.shape}")


# ### Tratamento dos Dados

# Limpeza dos dados simulados (remoção de valores inválidos)
dados_treino = sts.simulation_data_cleaning(simulation_data=dados_treino, value_to_clean=-99.0)

# Tratamento de NaNs
dados_treino = sts.simulation_nan_treatment(simulation=dados_treino, value=0, method='replace')

# Garantindo que profundidades são positivas
dados_treino, dados_inferencia = sts.depth_signal_checking(simulation_data=dados_treino, seismic_data=dados_inferencia)


# ### Análise da Distribuição dos Dados

sts.plot_simulation_distribution(dados_treino, bins=35)


# ### Modelagem - Treinamento e Validação

# Treinamento do modelo
dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo_treinado = sts.ML_model_evaluation(
    dados_simulacao=dados_treino,
    proporcao_treino=0.75,
    modelo="extratrees"
)

mlflow.set_experiment("Experimento_Exercio_2")

with mlflow.start_run(run_name="ExtraTrees_Model"):

    # ----------------------
    # Treinamento do Modelo
    # ----------------------
    dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo_treinado = sts.ML_model_evaluation(
        dados_simulacao=dados_treino,
        proporcao_treino=0.75
    )

    # ----------------------
    # Logging no MLflow
    # ----------------------
    mlflow.log_param("Modelo", "ExtraTreesRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("proporcao_treino", 0.75)

    mlflow.log_metric("NRMS_teste", nrms_teste)
    mlflow.log_metric("R2_teste", r2_teste)
    mlflow.log_metric("MAPE_teste", mape_teste)

    mlflow.sklearn.log_model(modelo_treinado, "modelo_extratrees")

    print("Experimento registrado no MLflow com sucesso!")
# ### Inferência - Aplicando Modelo aos Dados Sísmicos

# Realizando inferência
dados_estimados_prop_vector, dados_estimados = sts.transfer_to_seismic_scale(
    modelo_treinado=modelo_treinado,
    dados_sismicos=dados_inferencia
)


# ### Verificando Distribuição dos Dados Estimados

sts.plot_simulation_distribution(dados_estimados, bins=35, title="Distribuição Dados Estimados")


# ### Cálculo dos Resíduos

dados_estimados_residual_final = sts.calcular_residuos(
    dados_referencia=dados_referencia_comercial,
    dados_estimados=dados_estimados
)


# ### Visualização dos Resultados

# Slice dos dados
sts.plot_seismic_slice(dados_treino, title="Slice a ~5000m dos Dados de Treino")
sts.plot_seismic_slice(dados_referencia_comercial, title="Slice a ~5000m - Referência (Software Comercial)")
sts.plot_seismic_slice(dados_estimados, title="Slice a ~5000m - Inferência ML")
sts.plot_seismic_slice(dados_estimados_residual_final, title="Slice a ~5000m - Resíduo Inferência")


# ### Métricas - Simulação

# Organizando métricas
metricas_nome = ["nrms_teste", "r2_teste", "mape_teste"]
metricas_valores = [nrms_teste, r2_teste, mape_teste]

dict_metrics = dict(zip(metricas_nome, metricas_valores))
print("Métricas de Validação:", dict_metrics)
