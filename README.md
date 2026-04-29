# rinha_2026

Implementação inicial em Rust para a Rinha de Backend 2026.

## Status atual

- API com `GET /ready` e `POST /fraud-score`
- Vetorização 14D conforme a spec
- Busca KNN sobre vetores quantizados `i16`, usando apenas dataset reduzido por K-means embutido no binário
- Parsing/serialização JSON via `sonic-rs`
- Servidor HTTP/1.1 manual com keep-alive atrás do nginx em socket Unix
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

Antes de rodar ou compilar a API, gere o artefato versionável `spec/resources/index.mmap` com centroides/medoids por K-means:

```bash
cargo run --release --bin prebuild_shared_dataset spec/resources spec/resources/index.mmap 2048
```

Ou via script do repositório, já escrevendo o artefato versionável em `spec/resources/index.mmap`:

```bash
./scripts/generate-centroid-mmap.sh
```

Com parâmetros explícitos:

```bash
RINHA_KMEANS_K=2048 RINHA_KMEANS_SEED=67 ./scripts/generate-centroid-mmap.sh spec/resources spec/resources/index.mmap
```

Esse arquivo `spec/resources/index.mmap` deve ser adicionado ao repositório via Git LFS. O build agora usa `include_bytes!`, então compilar a API falha se esse artefato não existir.

Para rodar a API localmente depois disso:

```bash
cargo run --release
```

Para controlar a geração dos centroides:

```bash
RINHA_KMEANS_K=2048 cargo run --release --bin prebuild_shared_dataset -- spec/resources spec/resources/index.mmap 2048
RINHA_KMEANS_SEED=67 ./scripts/generate-centroid-mmap.sh
```

O artefato embutido contém:

- cabeçalho com versão/layout da quantização
- vetores `[i16; 16]` alinhados
- labels em bloco separado

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

O benchmark mede throughput usando o dataset centroid-reduzido embutido no binário.

## Docker Compose

Quando houver Docker disponível no host:

```bash
docker compose up
```

Observações:

- load balancer expõe a porta `9999`
- compose fixa `platform: linux/amd64`
- nginx reutiliza conexões HTTP/1.1 upstream sobre socket Unix
- as duas instâncias usam o mesmo artefato centroid-reduzido versionado no repositório

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
- `resources/index.mmap` embutido no binário em tempo de compilação

Você pode sobrescrever o diretório com:

```bash
RINHA_RESOURCES_DIR=/algum/path cargo run
```
