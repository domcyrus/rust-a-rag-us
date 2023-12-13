# rust-a-rag-us

A blend of "Rust", "RAG", and the whimsical "ragamuffin" or perhaps even a play on "asparagus".

There is a server and client binary. Use the cargo `--bin` to choose between.

## core components

- qdrant <https://github.com/qdrant/rust-client>
- using ollama-rs
- using embeddings from <https://github.com/guillaume-be/rust-bert>

## run rust-bert

 ```sh
 export LIBTORCH=$(brew --cellar pytorch)/$(brew info --json pytorch | jq -r '.[0].installed[0].version')
 export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
 ```

## how to use the server

```sh
RUST_LOG='info,rust_a_rag_us=debug' cargo run --bin server
```

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
