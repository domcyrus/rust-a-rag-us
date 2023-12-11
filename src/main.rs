use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use log::{debug, info};
use ollama_rs::Ollama;
use qdrant_client::client::QdrantClient;
use qdrant_client::client::QdrantClientConfig;
use rust_a_rag_us::data::Collection;
use rust_a_rag_us::embedding::text_embedding_async;
use rust_a_rag_us::embedding::{Model, EMBEDDING_SIZE};
use rust_a_rag_us::ollama::{Llm, PROMPT};
use rust_a_rag_us::qdrant::{add_documents, create_collections, search_documents};
use rust_a_rag_us::retriever::{fetch_content, sitemap};
use tiktoken_rs::p50k_base;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address of the Qdrant client
    #[clap(short, long, default_value = "http://localhost:6334")]
    address: String,

    /// collection used with the Qdrant client
    #[clap(short, long, default_value = "rura_collection")]
    base_collection: String,

    /// filter_collections is a comma separated list of collections to filter by
    /// if not specified, all collections will be searched
    /// valid values are: basic, summary
    /// example: --filter_collections=basic,summary
    #[clap(short, long, default_value = "basic", use_value_delimiter = true, value_delimiter = ',', num_args = 1..)]
    filter_collections: Vec<Collection>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
#[clap(rename_all = "snake_case")]
enum Command {
    Upload {
        #[clap(short, long)]
        url: String,

        #[clap(long, default_value = "http://localhost")]
        ollama_host: String,

        #[clap(long, default_value = "11434")]
        ollama_port: u16,

        #[clap(long, default_value = "orca2:13b")]
        ollama_model: String,
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

        #[clap(long, default_value = "http://localhost")]
        ollama_host: String,

        #[clap(long, default_value = "11434")]
        ollama_port: u16,

        #[clap(long, default_value = "orca2:13b")]
        ollama_model: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    env_logger::init();
    let args = Args::parse();

    let config = QdrantClientConfig::from_url(&args.address);
    let client = QdrantClient::new(Some(config))?;
    create_collections(
        &client,
        &args.base_collection,
        args.filter_collections.clone(),
        EMBEDDING_SIZE,
    )
    .await?;

    match args.command {
        Command::Upload {
            url,
            ollama_host,
            ollama_port,
            ollama_model,
        } => {
            info!("Fetching {}", url);
            let mut docs = sitemap(&url).await?;
            info!("Fetched {} docs from {}", docs.len(), url);

            info!("Creating Ollama client");
            let ollama = Ollama::new(ollama_host.to_string(), ollama_port);
            let llm = Llm::new(ollama);

            let (_handle, model) = Model::spawn();
            let total_docs = docs.len();
            info!("Adding {} documents", total_docs);

            let make_summary = args.filter_collections.contains(&Collection::Summary);

            for (i, doc) in docs.iter_mut().enumerate() {
                if make_summary {
                    info!("Creating summary document");
                    doc.add_summary(&ollama_model, &llm).await?;
                }
                let embeddings = model.encode(doc.clone()).await?;
                add_documents(
                    &client,
                    &args.base_collection,
                    args.filter_collections.clone(),
                    embeddings,
                )
                .await?;
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
            let docs = search_documents(
                &client,
                &args.base_collection,
                args.filter_collections,
                embeddings,
                limit,
            )
            .await?;
            // concat all the retrieved documents into one string
            let mut text = String::new();
            for doc in docs {
                debug!(
                    "Found doc: id: {:?}, text: {}",
                    doc.metadata.id, doc.metadata.text
                );
                text.push_str(&format!("- {}\n", doc.metadata.text.as_str()));
            }
            let formatted_prompt = PROMPT
                .replace("{context}", &text)
                .replace("{question}", &query.clone());
            debug!("Formatted prompt: {}", formatted_prompt);
            let bpe = p50k_base().unwrap();
            let tokens = bpe.encode_with_special_tokens(&formatted_prompt);
            println!("Token count: {}", tokens.len());
            let start = std::time::Instant::now();
            let answer = llm.generate(&ollama_model, &formatted_prompt).await?;
            info!(
                "Answer: {}, took: {} seconds",
                answer,
                start.elapsed().as_secs()
            );

            let start = std::time::Instant::now();
            let answer = llm.generate(&ollama_model, &formatted_prompt).await?;
            info!(
                "Answer: {}, took: {} seconds",
                answer,
                start.elapsed().as_secs()
            );
        }
        Command::Drop {} => {
            for collection in args.filter_collections {
                let collection_name =
                    format!("{}_{}", args.base_collection, collection.to_string());
                info!("Dropping collection {}", collection_name);
                client.delete_collection(&collection_name).await?;
            }
        }
        Command::SingleDoc {
            url,
            ollama_host,
            ollama_port,
            ollama_model,
        } => {
            info!("Creating Ollama client");
            let ollama = Ollama::new(ollama_host.to_string(), ollama_port);
            let llm = Llm::new(ollama);

            info!("Fetching {}", url);
            let mut doc = fetch_content(&url).await?;
            info!("Fetched doc: {:?}", doc);

            let basic_text = doc.text.get(&Collection::Basic).ok_or(anyhow::anyhow!(
                "Could not find basic text for document: {:?}",
                doc
            ))?;
            let bpe = p50k_base().unwrap();
            let tokens = bpe.encode_with_special_tokens(basic_text);
            println!("Token count: {}", tokens.len());

            let start = std::time::Instant::now();
            doc.add_summary(&ollama_model, &llm).await?;

            let summary = doc.text.get(&Collection::Summary).ok_or(anyhow::anyhow!(
                "Could not find summary for document: {:?}",
                doc
            ))?;
            info!("Answer: {}, took: {}", summary, start.elapsed().as_secs());
            let bpe = p50k_base().unwrap();
            let tokens = bpe.encode_with_special_tokens(&summary);
            println!("Token count: {}", tokens.len());
        }
    }

    Ok(())
}
