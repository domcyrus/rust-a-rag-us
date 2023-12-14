# rust-a-rag-us

This project, developed in `rust`, offers "retrieval-augmented generation" (RAG) capabilities for content gathered from the web.

Its core concept revolves around operating all essential components locally, thus avoiding reliance on external public APIs. This approach shares similarities with initiatives like privateGPT. The choice of rust as the programming language is primarily for educational exploration, rather than for performance optimization.

The project's name creatively merges "Rust" and "RAG," with a playful twist reminiscent of "ragamuffin," or even an allusion to "asparagus."

It comprises both a server and a client binary, with the flexibility to select either through the cargo --bin option.

There is a server and client binary. Use the cargo `--bin` to choose between.

## core components

- web scraper <https://github.com/causal-agent/scraper>
- qdrant <https://github.com/qdrant/rust-client>
- ollama-rs <https://github.com/pepperoni21/ollama-rs>
- embeddings via rust-bert <https://github.com/guillaume-be/rust-bert>

## run rust-bert

In order to be able to run rust-bert on MAC:

 ```sh
 export LIBTORCH=$(brew --cellar pytorch)/$(brew info --json pytorch | jq -r '.[0].installed[0].version')
 export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
 ```

## how to use the server

```sh
RUST_LOG='info,rust_a_rag_us=debug' cargo run --bin server
```

### server environment variables

- qdrant client address, defaults to `http://localhost:6334`: QDRANT_CLIENT_ADDRESS
- server listen address, defaults to `127.0.0.1:3000`: ADDRESS
- base collection, defaults to `rura_collection`: BASE_COLLECTION
- ollama model, defaults to `openhermes2.5-mistral:7b-q6_K`: OLLAMA_MODEL
- ollama host, defaults to `localhost`: OLLAMA_HOST
- ollama port, defaults to `11434`: OLLAMA_PORT

### swagger ui

Be default point your browser to `http://127.0.0.1:3000/swagger-ui/`

## how to use the client

 ```text
 Usage: rust-a-rag-us [OPTIONS] <COMMAND>

Commands:
  upload  
  query   
  drop    
  help    Print this message or the help of the given subcommand(s)

Options:
  -a, --address <ADDRESS>        Address of the Qdrant client [default: http://localhost:6334]
  -c, --collection <COLLECTION>  collection used with the Qdrant client [default: rura_collection]
  -h, --help                     Print help
  -V, --version                  Print version
 ```

### upload data

Point it to upload some data like this:

```sh
# basic example
rust-a-rag-us upload --url https://docs.lagoon.sh/

# setting logger and using collections
RUST_LOG='info,rust_a_rag_us=debug' rust-a-rag-us --filter-collections="basic,summary" upload --url='https://docs.lagoon.sh/'
```

### cleanup data

```sh
rust-a-rag-us drop
```

### query data

```sh
rust-a-rag-us query --query 'what lagoon service types can you in the docker compose yaml?'

# using cargo run
RUST_LOG=info cargo run --bin client -- --filter-collections="basic,summary" query --query 'what lagoon service types can you in the docker compose yaml?' --ollama_model openhermes2.5-mistral:7b-q6_K
```

You can also switch the model used by providing e.g. --ollama_model 'openhermes2.5-mistral:7b-q6_K'

## TODOs

- sitemap lookup does not recursively resolve sitemap pointing to another sitemap
- the ollama-rs streaming seems to be a bit brittle and fails with: Failed to deserialize response: EOF while parsing a list at line 1 column 8186
- make prompts configurable?
