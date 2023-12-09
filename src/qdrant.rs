use crate::data::EmbeddedMetadata;
use anyhow::Result;
use log::{debug, error, info};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, SearchPoints, VectorParams, Vectors, VectorsConfig};
use qdrant_client::serde::PayloadConversionError;
use serde_json::json;
use std::time::Instant;

use crate::data::EmbeddedDocument;

// create_collections creates two collections one for text and one for meta with the given name and size
pub async fn create_collections(client: &QdrantClient, collection: &str, size: u64) -> Result<()> {
    // we create two collctions, one for text embeddings and one for meta embeddings
    let text_collection = format!("{}_text", collection);
    let meta_collection = format!("{}_meta", collection);

    create_collection(client, &text_collection, size).await?;
    create_collection(client, &meta_collection, size).await?;

    Ok(())
}

async fn create_collection(client: &QdrantClient, collection: &str, size: u64) -> Result<()> {
    if !client.has_collection(&collection).await? {
        info!("Creating text collection: {}", collection);
        client
            .create_collection(&CreateCollection {
                collection_name: collection.into(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: size,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;
    } else {
        info!("Text collection: {} already exists", collection);
    }

    Ok(())
}

// add_documents adds documents to a collection
pub async fn add_documents(
    client: &QdrantClient,
    collection: &str,
    documents: Vec<EmbeddedDocument>,
) -> Result<()> {
    let text_collection = format!("{}_text", collection);
    let meta_collection = format!("{}_meta", collection);
    for collection_name in vec![text_collection.clone(), meta_collection.clone()] {
        if !client.has_collection(&collection_name).await? {
            return Err(anyhow::anyhow!(
                "Collection: {} does not exist",
                collection_name
            ));
        }
    }
    let mut text_points: Vec<PointStruct> = vec![];
    let mut meta_points: Vec<PointStruct> = vec![];
    let time_to_add = Instant::now();
    for document in documents {
        let payload: Result<Payload, PayloadConversionError> = json!(document.metadata).try_into();
        match payload {
            Ok(payload) => {
                text_points.push(PointStruct {
                    id: Some(document.metadata.id.clone().into()),
                    payload: payload.clone().into(),
                    vectors: Some(Vectors::from(document.text_embeddings.clone())),
                });
                meta_points.push(PointStruct {
                    id: Some(document.metadata.id.clone().into()),
                    payload: payload.into(),
                    vectors: Some(Vectors::from(document.meta_embeddings.clone())),
                });
            }
            Err(e) => {
                error!("Error converting payload: {}", e);
                return Err(anyhow::anyhow!("Error converting payload: {}", e));
            }
        }
    }
    let num_text_points = text_points.len();
    let num_meta_points = meta_points.len();
    client
        .upsert_points_blocking(&text_collection, text_points, None)
        .await?;
    info!(
        "Added {} documents to text collection: {} in elapsed time: {:?}",
        num_text_points,
        text_collection,
        time_to_add.elapsed(),
    );
    client
        .upsert_points_blocking(&meta_collection, meta_points, None)
        .await?;
    info!(
        "Added {} documents to meta collection: {} in elapsed time: {:?}",
        num_meta_points,
        meta_collection,
        time_to_add.elapsed(),
    );

    Ok(())
}

// search_documents searches for documents in a collection based on cosine distance of embeddings
pub async fn search_documents(
    client: &QdrantClient,
    collection: &str,
    embeddings: Vec<f32>,
    limit: u64,
) -> Result<Vec<EmbeddedDocument>> {
    let text_collection = format!("{}_text", collection);
    let meta_collection = format!("{}_meta", collection);

    // we want to have 2 thirds of the limit for text and 1 third for meta
    let text_limit = (limit as f64 * 0.50) as u64;
    info!("text_limit: {}", text_limit);

    let meta_limit = (limit as f64 * 0.50) as u64;
    info!("meta_limit: {}", meta_limit);

    let search_text_result = client
        .search_points(&SearchPoints {
            collection_name: text_collection.into(),
            vector: embeddings.clone(),
            filter: None,
            limit: text_limit,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    debug!("text results: {:?}", &search_text_result);
    let search_meta_result = client
        .search_points(&SearchPoints {
            collection_name: meta_collection.into(),
            vector: embeddings,
            filter: None,
            limit: meta_limit,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    debug!("meta results: {:?}", &search_meta_result);
    let mut results = vec![];
    for result in search_text_result.result {
        let metadata_json = serde_json::to_value(&result.payload)?;
        let metadata: Result<EmbeddedMetadata, serde_json::Error> =
            serde_json::from_value(metadata_json);

        match metadata {
            Ok(metadata) => {
                let embedded_document = EmbeddedDocument {
                    text_embeddings: vec![],
                    meta_embeddings: vec![],
                    metadata: metadata,
                };
                results.push(embedded_document);
            }
            Err(e) => {
                error!("Error converting metadata: {}", e);
                return Err(anyhow::anyhow!("Error converting metadata: {}", e));
            }
        }
    }
    for result in search_meta_result.result {
        let metadata_json = serde_json::to_value(&result.payload)?;
        let metadata: Result<EmbeddedMetadata, serde_json::Error> =
            serde_json::from_value(metadata_json);

        match metadata {
            Ok(metadata) => {
                let embedded_document = EmbeddedDocument {
                    text_embeddings: vec![],
                    meta_embeddings: vec![],
                    metadata: metadata,
                };
                results.push(embedded_document);
            }
            Err(e) => {
                error!("Error converting metadata: {}", e);
                return Err(anyhow::anyhow!("Error converting metadata: {}", e));
            }
        }
    }

    debug!("result: {:?}", &results);
    Ok(results)
}

// drop_collection drops a collection for both the text and meta collection
pub async fn drop_collections(client: &QdrantClient, collection: &str) -> Result<()> {
    let text_collection = format!("{}_text", collection);
    let meta_collection = format!("{}_meta", collection);
    for collection_name in vec![text_collection.clone(), meta_collection.clone()] {
        if client.has_collection(&collection_name).await? {
            info!("Dropping collection: {}", collection);
            client.delete_collection(&collection_name).await?;
        } else {
            info!("Collection: {} does not exist", collection);
        }
    }

    Ok(())
}
