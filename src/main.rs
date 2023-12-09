use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use log::info;
use ollama_rs::Ollama;
use qdrant_client::client::QdrantClient;
use qdrant_client::client::QdrantClientConfig;
use rust_a_rag_us::embedding::text_embedding_async;
use rust_a_rag_us::embedding::{Model, EMBEDDING_SIZE};
use rust_a_rag_us::ollama::{Llm, PROMPT};
use rust_a_rag_us::qdrant::{add_documents, create_collections, search_documents};
use rust_a_rag_us::retriever::{single_doc, sitemap};
use tiktoken_rs::p50k_base;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address of the Qdrant client
    #[clap(short, long, default_value = "http://localhost:6334")]
    address: String,

    /// collection used with the Qdrant client
    #[clap(short, long, default_value = "rura_collection")]
    collection: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
#[clap(rename_all = "snake_case")]
enum Command {
    Upload {
        #[clap(short, long)]
        url: String,
    },
    Query {
        #[clap(short, long)]
        query: String,

        #[clap(short, long, default_value = "7")]
        limit: u64,

        #[clap(long, default_value = "http://localhost")]
        ollama_host: String,

        #[clap(long, default_value = "11434")]
        ollama_port: u16,

        #[clap(long, default_value = "orca2:13b")]
        ollama_model: String,
    },
    Drop {},
    SingleDoc {
        #[clap(short, long)]
        url: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();
    let args = Args::parse();

    let config = QdrantClientConfig::from_url(&args.address);
    let client = QdrantClient::new(Some(config))?;
    create_collections(&client, &args.collection, EMBEDDING_SIZE).await?;

    match args.command {
        Command::Upload { url } => {
            info!("Fetching {}", url);
            let docs = sitemap(&url).await?;
            info!("Fetched {} docs from {}", docs.len(), url);

            let (_handle, model) = Model::spawn();
            let total_docs = docs.len();
            info!("Adding {} documents", total_docs);
            for (i, doc) in docs.iter().enumerate() {
                let embeddings = model.encode(doc.clone()).await?;
                add_documents(&client, &args.collection, embeddings).await?;
                if i == total_docs - 1 {
                    info!("Added {} documents", total_docs);
                    return Ok(());
                } else if i % 10 == 0 {
                    info!("Added {} documents", i);
                }
            }
        }
        Command::Query {
            query,
            limit,
            ollama_host,
            ollama_port,
            ollama_model,
        } => {
            info!("Creating Ollama client");
            let ollama = Ollama::new(ollama_host.to_string(), ollama_port);
            let llm = Llm::new(ollama);

            info!("Querying {} with limit {}", query, limit);
            let embeddings = text_embedding_async(query.clone()).await;
            let docs = search_documents(&client, &args.collection, embeddings, limit).await?;
            // concat all the retrieved documents into one string
            let mut text = String::new();
            for doc in docs {
                info!(
                    "Found doc: id: {:?}, text: {}",
                    doc.metadata.hash, doc.metadata.text
                );
                text.push_str(&format!("- {}\n", doc.metadata.text.as_str()));
            }
            let formatted_prompt = PROMPT
                .replace("{context}", &text)
                .replace("{question}", &query.clone());
            info!("Formatted prompt: {}", formatted_prompt);
            let bpe = p50k_base().unwrap();
            let tokens = bpe.encode_with_special_tokens(&formatted_prompt);
            println!("Token count: {}", tokens.len());
            let start = std::time::Instant::now();
            let answer = llm
                .generate(ollama_model.clone(), formatted_prompt.clone())
                .await?;
            info!("Answer: {}, took: {}", answer, start.elapsed().as_secs());

            let start = std::time::Instant::now();
            let answer = llm
                .generate(ollama_model.clone(), formatted_prompt.clone())
                .await?;
            info!("Answer: {}, took: {}", answer, start.elapsed().as_secs());
        }
        Command::Drop {} => {
            info!("Dropping collection {}", args.collection);
            client.delete_collection(&args.collection).await?;
        }
        Command::SingleDoc { url } => {
            info!("Fetching {}", url);
            let doc = single_doc(&url).await?;
            info!("Fetched doc: {:?}", doc);
            let bpe = p50k_base().unwrap();
            let tokens = bpe.encode_with_special_tokens(&doc.text);
            println!("Token count: {}", tokens.len());
        }
    }

    Ok(())
}
