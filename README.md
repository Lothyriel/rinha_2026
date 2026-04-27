# rinha_2026

Implementação inicial em Rust para a Rinha de Backend 2026.

## Status atual

- API com `GET /ready` e `POST /fraud-score`
- Vetorização 14D conforme a spec
- Busca KNN exata com VP-tree flattenizada + distância euclidiana quadrática
- Carregamento do dataset oficial via `resources/references.json.gz`
- `docker-compose.yml` com nginx + 2 instâncias da API

## Rodando localmente

```bash
cargo run
```

Servidor sobe na porta `9999`.

Para rodar em socket Unix ao invés de TCP:

```bash
RINHA_UNIX_SOCKET_PATH=/tmp/rinha-api.sock cargo run
```

Quando `RINHA_UNIX_SOCKET_PATH` está definido, a API passa a escutar apenas no socket informado. Quando não está definido, o comportamento padrão continua sendo TCP na porta `9999`.

Para compartilhar o dataset/index entre processos via arquivo `mmap`:

```bash
RINHA_SHARED_MMAP_PATH=/tmp/rinha-shared-dataset.bin cargo run
```

O primeiro processo cria o arquivo mapeado e os próximos apenas o reutilizam em modo leitura.

## Testes

```bash
cargo test
cargo build --release
```

## Benchmark

Benchmark simples de scoring em processo:

```bash
cargo run --release --example fraud_score_bench
```

O benchmark usa payloads reais extraídos da spec e mede throughput do `engine.score()`.

## Docker Compose

Quando houver Docker disponível no host:

```bash
docker compose up
```

Observações:

- load balancer expõe a porta `9999`
- compose fixa `platform: linux/amd64`
- backend padrão no compose é `exact`, para manter aderência à spec
- o backend experimental pode ser testado alterando `RINHA_SEARCH_BACKEND`

## Publicação de imagem

Ao criar um git tag no formato `v*`, o GitHub Actions publica a imagem da API no GHCR com a mesma tag.

Exemplo:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Imagem publicada:

```text
ghcr.io/lothyriel/rinha_2026:v0.*.*
```

## Recursos

Por padrão a aplicação lê:

- `resources/references.json.gz`
- `resources/mcc_risk.json`
- `resources/normalization.json`

Você pode sobrescrever o diretório com:

```bash
RINHA_RESOURCES_DIR=/algum/path cargo run
```
