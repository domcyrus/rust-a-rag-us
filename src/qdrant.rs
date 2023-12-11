use crate::data::{Collection, EmbeddedMetadata};
use anyhow::Result;
use log::{error, info};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, SearchPoints, VectorParams, Vectors, VectorsConfig};
use qdrant_client::serde::PayloadConversionError;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

use crate::data::EmbeddedDocument;

// create_collections creates two collections one for text and one for meta with the given name and size
pub async fn create_collections(
    client: &QdrantClient,
    collection_base: &str,
    collections: Vec<Collection>,
    size: u64,
) -> Result<()> {
    info!("Creating collections, with base: {}", collection_base);
    for collection in collections {
        let collection_name = format!("{}_{}", collection_base, collection.to_string());
        create_collection(client, &collection_name, size).await?;
    }
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
    collection_base: &str,
    filter_by_collections: Vec<Collection>,
    documents: Vec<EmbeddedDocument>,
) -> Result<()> {
    for collection_name in filter_by_collections.clone() {
        let collection_name = format!("{}_{}", collection_base, collection_name.to_string());
        if !client.has_collection(&collection_name).await? {
            return Err(anyhow::anyhow!(
                "Collection: {} does not exist",
                collection_name
            ));
        }
    }
    let mut text_points: HashMap<Collection, Vec<PointStruct>> = HashMap::new();
    let time_to_add = Instant::now();
    for document in documents {
        // check if document by filter_by_collections
        if !filter_by_collections.contains(&document.metadata.collection) {
            info!(
                "Skipping document: {} because it is not in filter_by_collections: {:?}",
                document.metadata.id, filter_by_collections
            );
            continue;
        }

        let payload: Result<Payload, PayloadConversionError> = json!(document.metadata).try_into();
        match payload {
            // get text_points for collection
            Ok(payload) => {
                if let Some(point_vec) = text_points.get_mut(&document.metadata.collection) {
                    point_vec.push(PointStruct {
                        id: Some(document.metadata.id.clone().into()),
                        payload: payload.clone().into(),
                        vectors: Some(Vectors::from(document.text_embeddings.clone())),
                    });
                } else {
                    text_points.insert(
                        document.metadata.collection.clone(),
                        vec![PointStruct {
                            id: Some(document.metadata.id.clone().into()),
                            payload: payload.clone().into(),
                            vectors: Some(Vectors::from(document.text_embeddings.clone())),
                        }],
                    );
                }
            }
            Err(e) => {
                error!("Error converting payload: {}", e);
                return Err(anyhow::anyhow!("Error converting payload: {}", e));
            }
        }
    }
    let mut num_text_points = 0;

    for (collection, points) in text_points {
        let collection_name = format!("{}_{}", collection_base, collection.to_string());
        info!(
            "Adding {} documents to text collection: {}",
            points.len(),
            collection_name
        );
        num_text_points += points.len();
        client
            .upsert_points_blocking(&collection_name, points, None)
            .await?;
    }
    info!(
        "Added {} documents to qrdant in elapsed time: {:?}",
        num_text_points,
        time_to_add.elapsed(),
    );

    Ok(())
}

// search_documents searches for documents in a collection based on cosine distance of embeddings
pub async fn search_documents(
    client: &QdrantClient,
    base_collection: &str,
    filter_by_collections: Vec<Collection>,
    embeddings: Vec<f32>,
    limit: u64,
) -> Result<Vec<EmbeddedDocument>> {
    // we will limit the search for each collection the same
    let total_collections = filter_by_collections.len();

    let mut results = Vec::new();
    for filter_collection in filter_by_collections.clone() {
        let collection_name = format!("{}_{}", base_collection, filter_collection.to_string());
        if !client.has_collection(&collection_name).await? {
            return Err(anyhow::anyhow!(
                "Collection: {} does not exist",
                collection_name
            ));
        }
        let mut collection_limit = limit;
        if total_collections > 1 {
            // multiply limit by filter_collection ratio
            collection_limit = (limit as f32 * filter_collection.limit_by_collection()) as u64;
            if collection_limit == 0 {
                collection_limit = 1;
            }
        }
        info!(
            "Searching collection: {} with limit: {}",
            collection_name, collection_limit
        );
        let search_text_result = client
            .search_points(&SearchPoints {
                collection_name: collection_name.into(),
                vector: embeddings.clone(),
                filter: None,
                limit: collection_limit,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await?;
        for search_result in search_text_result.result {
            let metadata_json = serde_json::to_value(&search_result.payload)?;
            let metadata: Result<EmbeddedMetadata, serde_json::Error> =
                serde_json::from_value(metadata_json);

            match metadata {
                Ok(metadata) => {
                    let embedded_document = EmbeddedDocument {
                        text_embeddings: vec![],
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
    }
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
