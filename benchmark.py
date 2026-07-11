import time
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("Carregando modelo e dataset...")
model = SentenceTransformer('all-MiniLM-L6-v2')
dataset = load_dataset("dair-ai/emotion", split="train[:5000]")

textos_totais = dataset['text']
print("Gerando embeddings (pode levar alguns segundos)...")
vetores_totais = model.encode(textos_totais).tolist()
dimensao = len(vetores_totais[0])

textos_db = textos_totais[:4900]
vetores_db = vetores_totais[:4900]
vetores_query = vetores_totais[4900:]

pg_conn = psycopg2.connect(dbname="vetores", user="postgres", password="password", host="localhost")
q_client = QdrantClient(host="localhost", port=6333)

iteracoes = 30
metricas = {
    "pg_insert": [], "qdrant_insert": [],
    "pg_latency": [], "qdrant_latency": [],
    "pg_recall": [], "qdrant_recall": []
}

print(f"\nIniciando Benchmark de {iteracoes} iterações...")

for rodada in range(iteracoes):
    print(f"\n--- Rodada {rodada + 1}/{iteracoes} ---")
    
    # POSTGRE
    cur = pg_conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("DROP TABLE IF EXISTS documentos;")
    cur.execute(f"CREATE TABLE documentos (id integer PRIMARY KEY, texto text, embedding vector({dimensao}));")
    
    start_pg_ins = time.time()
    for i, (txt, vet) in enumerate(zip(textos_db, vetores_db)):
        cur.execute("INSERT INTO documentos (id, texto, embedding) VALUES (%s, %s, %s)", (i, txt, str(vet)))
    pg_conn.commit()
    cur.execute("CREATE INDEX ON documentos USING hnsw (embedding vector_cosine_ops);")
    pg_conn.commit()
    metricas["pg_insert"].append(time.time() - start_pg_ins)

    cur.execute("SET enable_indexscan = off;") # Desliga o índice
    gabarito = []
    for q_vet in vetores_query:
        cur.execute("SELECT id FROM documentos ORDER BY embedding <=> %s LIMIT 10;", (str(q_vet),))
        gabarito.append([row[0] for row in cur.fetchall()])

    cur.execute("SET enable_indexscan = on;") # Religa o índice
    start_pg_q = time.time()
    resultados_pg = []
    for q_vet in vetores_query:
        cur.execute("SELECT id FROM documentos ORDER BY embedding <=> %s LIMIT 10;", (str(q_vet),))
        resultados_pg.append([row[0] for row in cur.fetchall()])
    
    tempo_total_q_pg = time.time() - start_pg_q
    latencia_media_pg_ms = (tempo_total_q_pg / len(vetores_query)) * 1000
    metricas["pg_latency"].append(latencia_media_pg_ms)
    

    # QDRANT
    if q_client.collection_exists(collection_name="documentos"):
        q_client.delete_collection(collection_name="documentos")
        
    q_client.create_collection(
        collection_name="documentos",
        vectors_config=VectorParams(size=dimensao, distance=Distance.COSINE),
    )
    
    start_q_ins = time.time()
    pontos = [PointStruct(id=i, vector=vet, payload={"texto": txt}) for i, (txt, vet) in enumerate(zip(textos_db, vetores_db))]
    tamanho_lote = 500
    for i in range(0, len(pontos), tamanho_lote):
        lote = pontos[i : i + tamanho_lote]
        q_client.upsert(collection_name="documentos", points=lote)
    metricas["qdrant_insert"].append(time.time() - start_q_ins)

    start_q_q = time.time()
    resultados_qdrant = []
    for q_vet in vetores_query:
        res = q_client.query_points(collection_name="documentos", query=q_vet, limit=10)
        resultados_qdrant.append([p.id for p in res.points])
    
    tempo_total_q_qdrant = time.time() - start_q_q
    latencia_media_q_ms = (tempo_total_q_qdrant / len(vetores_query)) * 1000
    metricas["qdrant_latency"].append(latencia_media_q_ms)

    acertos_pg = sum(len(set(g) & set(r)) for g, r in zip(gabarito, resultados_pg))
    acertos_qdrant = sum(len(set(g) & set(r)) for g, r in zip(gabarito, resultados_qdrant))
    
    total_esperado = len(vetores_query) * 10
    metricas["pg_recall"].append((acertos_pg / total_esperado) * 100)
    metricas["qdrant_recall"].append((acertos_qdrant / total_esperado) * 100)

cur.close()
pg_conn.close()

print("\nGerando estatísticas e gráficos em PDF...")

def gerar_grafico(dados_pg, dados_qdrant, titulo, ylabel, filename):
    media_pg, std_pg = np.mean(dados_pg), np.std(dados_pg)
    media_q, std_q = np.mean(dados_qdrant), np.std(dados_qdrant)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(['pgvector', 'Qdrant'], [media_pg, media_q], 
                    yerr=[std_pg, std_q], capsize=5, 
                    color=['#3366cc', '#dc3912'], alpha=0.8)
    
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for barra in barras:
        yval = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2, yval + (yval*0.05), 
                f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.close()

gerar_grafico(metricas["pg_insert"], metricas["qdrant_insert"], 
              'Tempo Médio de Inserção + Indexação (n=30)', 'Segundos', 'grafico_insercao.pdf')

gerar_grafico(metricas["pg_latency"], metricas["qdrant_latency"], 
              'Latência Média de Consulta (n=30)', 'Milissegundos (ms)', 'grafico_latencia.pdf')

print("\n--- RESUMO FINAL DO BENCHMARK ---")
print(f"pgvector -> Latência: {np.mean(metricas['pg_latency']):.2f}ms | Recall: {np.mean(metricas['pg_recall']):.2f}%")
print(f"Qdrant   -> Latência: {np.mean(metricas['qdrant_latency']):.2f}ms | Recall: {np.mean(metricas['qdrant_recall']):.2f}%")
print("\nArquivos 'grafico_insercao.pdf' e 'grafico_latencia.pdf' gerados com sucesso para upload no Overleaf!")