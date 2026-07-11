# Benchmark de Bancos de Dados Vetoriais: pgvector vs Qdrant

Este repositório contém o código-fonte e a infraestrutura necessários para reproduzir o experimento comparativo de desempenho entre o `pgvector` (extensão do PostgreSQL) e o banco nativo vetorial `Qdrant`. 

O estudo avalia o tempo de indexação em lote (*bulk insert*), a latência média de consulta e a precisão (*Recall@10*) da busca aproximada (HNSW) em um ambiente controlado.

## Pré-requisitos

Para executar este benchmark, o ambiente hospedeiro precisa ter instalado:
* Docker e Docker Compose
* Python 3 (testado no Pop!_OS / Ubuntu)
* Pip e módulo `venv`

## Instruções de Execução

Siga o passo a passo abaixo no terminal para configurar o ambiente isolado e rodar o teste de estresse.

**1. Clone o repositório**

```bash
git clone https://github.com/SQL-2026-Trabalho/ep2
cd ep2
```

**2. Suba a infraestrutura de banco de dados**

Inicie os contêineres do PostgreSQL (com a extensão pgvector) e do Qdrant em segundo plano:
```bash
docker-compose up -d
```

**3. Crie e ative o ambiente virtual Python**

Isso garante que as dependências não entrem em conflito com os pacotes do seu sistema:
```bash
python3 -m venv venv
source venv/bin/activate
```

**4. Instale as dependências do projeto**

```bash
pip install psycopg2-binary qdrant-client sentence-transformers datasets matplotlib numpy
```

**5. Execute o benchmark**

Inicie o script principal. Ele fará o download do dataset (dair-ai/emotion), gerará os embeddings de 384 dimensões localmente e iniciará as 30 iterações do teste:
```bash
python benchmark.py
```

## Resultados Esperados

O script levará alguns minutos para ser concluído, dependendo do hardware do hospedeiro. Ao final da execução, ele irá:
* Imprimir no terminal um resumo consolidado com as médias de latência e porcentagens de Recall.
* Gerar automaticamente dois arquivos em alta resolução: grafico_insercao.pdf e grafico_latencia.pdf, contendo as margens de erro (desvio padrão) das execuções.

## Encerramento

Após a coleta dos dados, você pode derrubar os contêineres e limpar os volumes gerados executando:
```bash
docker-compose down
```
