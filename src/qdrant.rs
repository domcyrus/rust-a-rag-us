use crate::data::EmbeddedMetadata;
use anyhow::Result;
use log::{error, info};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, SearchPoints, VectorParams, Vectors, VectorsConfig};
use qdrant_client::serde::PayloadConversionError;
use serde_json::json;
use std::time::Instant;

use crate::data::EmbeddedDocument;

pub async fn create_collection(client: &QdrantClient, collection: &str, size: u64) -> Result<()> {
    if !client.has_collection(&collection).await? {
        info!("Creating collection: {}", collection);
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
        info!("Collection: {} already exists", collection);
    }

    Ok(())
}

pub async fn add_documents(
    client: &QdrantClient,
    collection: &str,
    documents: Vec<EmbeddedDocument>,
) -> Result<()> {
    if !client.has_collection(&collection).await? {
        return Err(anyhow::anyhow!("Collection: {} does not exist", collection));
    }
    let mut points: Vec<PointStruct> = vec![];
    let time_to_add = Instant::now();
    for document in documents {
        let payload: Result<Payload, PayloadConversionError> = json!(document.metadata).try_into();
        match payload {
            Ok(payload) => {
                points.push(PointStruct {
                    id: Some(document.metadata.id.clone().into()),
                    payload: payload.into(),
                    vectors: Some(Vectors::from(document.embeddings.clone())),
                });
            }
            Err(e) => {
                error!("Error converting payload: {}", e);
                return Err(anyhow::anyhow!("Error converting payload: {}", e));
            }
        }
    }
    let num_points = points.len();
    client
        .upsert_points_blocking(collection, points, None)
        .await?;
    info!(
        "Added {} documents to collection: {} in elapsed time: {:?}",
        num_points,
        collection,
        time_to_add.elapsed(),
    );
    Ok(())
}

pub async fn search_documents(
    client: &QdrantClient,
    collection: &str,
    embeddings: Vec<f32>,
    limit: u64,
) -> Result<Vec<EmbeddedDocument>> {
    let search_result = client
        .search_points(&SearchPoints {
            collection_name: collection.into(),
            vector: embeddings,
            filter: None,
            limit: limit,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    dbg!(&search_result);

    let mut results = vec![];
    for result in search_result.result {
        let metadata_json = serde_json::to_value(&result.payload)?;
        let metadata: Result<EmbeddedMetadata, serde_json::Error> =
            serde_json::from_value(metadata_json);

        match metadata {
            Ok(metadata) => {
                let embedded_document = EmbeddedDocument {
                    embeddings: vec![],
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
    dbg!(&results);
    Ok(results)
}

pub async fn drop_collection(client: &QdrantClient, collection: &str) -> Result<()> {
    if client.has_collection(&collection).await? {
        info!("Dropping collection: {}", collection);
        client.delete_collection(&collection).await?;
    } else {
        info!("Collection: {} does not exist", collection);
    }

    Ok(())
}
