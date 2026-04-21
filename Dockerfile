FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef

WORKDIR /app

FROM chef AS planner

COPY Cargo.toml ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

COPY --from=planner /app/recipe.json recipe.json
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml ./
COPY src ./src
COPY ./spec/resources ./spec/resources
RUN cargo build --release

FROM debian:trixie-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rinha_2026 /usr/local/bin/rinha-api
COPY ./spec/resources /app/resources

ENV PORT=9999
ENV RINHA_RESOURCES_DIR=/app/resources

EXPOSE 9999

CMD ["rinha-api"]
