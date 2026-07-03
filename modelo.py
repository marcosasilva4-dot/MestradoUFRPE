import random
import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. SIMULAÇÃO DE DADOS (Entradas do Modelo)
# ==========================================
np.random.seed(42)

# Cadastro de 100 usuários (Nós)
usuarios = [f"user_{i}" for i in range(100)]

# Características textuais de PLN (Simulando a métrica de Tom de Hamm & McKeever)
dados_pln = {
    "user_id": usuarios,
    "tom_manipulador": np.random.uniform(0.1, 0.9, 100)
}
df_pln = pd.DataFrame(dados_pln)

# Simulação de Arestas (Conexões/Interações na rede)
arestas = []
for i in range(len(usuarios) - 1):
    arestas.append((usuarios[i], usuarios[i+1]))

# Inserindo anomalias topológicas (Um usuário se conectando em massa de forma isolada)
predador_simulado = "user_99"
df_pln.loc[df_pln["user_id"] == predador_simulado, "tom_manipulador"] = 0.95

for i in range(0, 30, 2): 
    arestas.append((predador_simulado, f"user_{i}"))

# ==========================================
# 2. CONSTRUÇÃO DO GRAFO E NODE2VEC
# ==========================================

# --- CORREÇÃO 3: Instanciar o Grafo G com as arestas geradas acima ---
print("Construindo o Grafo de Interações...")
G = nx.Graph()
G.add_nodes_from(usuarios)
G.add_edges_from(arestas)

# --- FUNÇÃO DO NODE2VEC (Usando random importado) ---
def gerar_caminhadas_aleatorias(grafo, tamanho_caminhada, num_caminhadas):
    """
    Simula o comportamento de exploração topológica do Node2Vec.
    Gera sequências de nós que representam os caminhos na rede social.
    """
    caminhadas = []
    nos = list(grafo.nodes())
    
    for _ in range(num_caminhadas):
        random.shuffle(nos) 
        for no in nos:
            caminhada = [no]
            while len(caminhada) < tamanho_caminhada:
                vizinhos = list(grafo.neighbors(caminhada[-1]))
                if len(vizinhos) > 0:
                    caminhada.append(random.choice(vizinhos))
                else:
                    break
            caminhadas.append([str(n) for n in caminhada])
            
    return caminhadas

# --- EXECUÇÃO NO PIPELINE ---
print("Gerando caminhadas aleatórias pelo grafo...")
caminhadas_amostradas = gerar_caminhadas_aleatorias(G, tamanho_caminhada=10, num_caminhadas=40)

print("Gerando embeddings dos nós via Word2Vec...")
modelo_node2vec = Word2Vec(
    sentences=caminhadas_amostradas, 
    vector_size=16,   
    window=5,         
    min_count=1, 
    sg=1,             
    workers=2
)

vetor_topologico = modelo_node2vec.wv['user_99']
print("Vetor estrutural gerado com sucesso para o 'user_99'!")

# --- CORREÇÃO 4: Gerar o df_embeddings extraindo os vetores de todos os nós ---
embeddings_estruturais = {str(nodo): modelo_node2vec.wv[str(nodo)] for nodo in G.nodes()}
df_embeddings = pd.DataFrame.from_dict(embeddings_estruturais, orient='index').reset_index()
df_embeddings.rename(columns={'index': 'user_id'}, inplace=True)

# ==========================================
# 3. FUSÃO DE CARACTERÍSTICAS (Feature Fusion)
# ==========================================
print("Executando a fusão multimodal (Texto + Estrutura)...")
df_final = pd.merge(df_pln, df_embeddings, on="user_id")

# Definindo labels para o treino
df_final["label"] = 0
df_final.loc[df_final["user_id"] == predador_simulado, "label"] = 1
df_final.loc[(df_final["tom_manipulador"] > 0.82) & (df_final["user_id"] != predador_simulado), "label"] = 1

# ==========================================
# 4. TREINAMENTO DO CLASSIFICADOR FINAL (SVM)
# ==========================================
X = df_final.drop(columns=["user_id", "label"]).values
y = df_final["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Treinando o Classificador Final (Support Vector Machine - SVM)...")
classificador = SVC(kernel='linear', probability=True)
classificador.fit(X_train, y_train)

y_pred = classificador.predict(X_test)

print("\n=== RESULTADOS DO MODELO HÍBRIDO ===")
print(f"Acurácia Geral: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=0))