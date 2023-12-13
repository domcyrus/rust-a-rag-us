use crate::data::Collection;
use crate::embedding::EmbeddingProgress;
use crate::ollama;
use crate::progress_tracker::ProgressTracker;
use crate::qdrant::add_documents;
use crate::retriever;
use crate::state::AppState;
use axum::{
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use chrono::Utc;
use log::info;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Instant};
use utoipa::{OpenApi, ToSchema};
use uuid::Uuid;

// Define a serializable structure for your response
#[derive(Serialize)]
pub struct StateResponse {
    // Add fields relevant for your response
    progress_data: HashMap<Uuid, EmbeddingProgress>,
}

#[derive(OpenApi)]
#[openapi(
    paths(get_state, upload),
    components(schemas(UploadParams, Collection))
)]
pub struct ApiDoc;

/// get-state function returns the current progress state
///
/// This route does retrieve the current state.
#[utoipa::path(
    get,
    path = "/get-state",
    responses(
        (status = 200, description = "Success response", body = String),
        (status = 500, description = "Internal Server Error", body = String)
    )
)]
pub async fn get_state(
    state: axum::extract::Extension<Arc<AppState<EmbeddingProgress>>>,
) -> Json<StateResponse> {
    let progress_map = state.get_all_progress();
    let progress_data = progress_map.clone();
    drop(progress_map);
    Json(StateResponse { progress_data })
}

#[derive(Deserialize, Default, ToSchema)]
pub struct UploadParams {
    pub url: String,
    pub ollama_model: Option<String>,
    pub ollama_host: Option<String>,
    pub ollama_port: Option<u16>,
    pub filter_collections: Option<Vec<Collection>>,
    pub base_collection: Option<String>,
}

/// upload function starts an upload task
///
/// This route does start an upload task.
#[utoipa::path(
    post,
    path = "/upload",
    params(
        ("upload_params" = UploadParams, Path, description = "Upload parameters"),
    ),
    responses(
        (status = 200, description = "Success response", body = String),
        (status = 500, description = "Internal Server Error", body = String)
    )
)]
pub async fn upload(
    state: axum::extract::Extension<Arc<AppState<EmbeddingProgress>>>,
    upload_params: Option<Query<UploadParams>>,
) -> (StatusCode, Json<String>) {
    // create uuid
    let id = Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("{}{}", "upload", Utc::now()).as_bytes(),
    );

    let Query(upload_params) = upload_params.unwrap_or(Query::default());
    let ollama_model = upload_params
        .ollama_model
        .unwrap_or(state.app_config.ollama_model.clone());
    info!("Ollama model {}", ollama_model);
    let ollama_host = upload_params
        .ollama_host
        .unwrap_or(state.app_config.ollama_host.clone());
    info!("Ollama host {}", ollama_host);
    let ollama_port = upload_params
        .ollama_port
        .unwrap_or(state.app_config.ollama_port.clone());
    let filter_collections = upload_params
        .filter_collections
        .unwrap_or(state.app_config.filter_collections.clone());
    let base_collection = upload_params
        .base_collection
        .unwrap_or(state.app_config.base_collection.clone());
    info!("Ollama port {}", ollama_port);
    let url = upload_params.url;

    if url.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json("mandatory URL is empty".to_string()),
        );
    }

    info!("Fetching {}", url);
    let start = Instant::now();
    let qdrant_client = state.app_config.qdrant_client.clone();
    let docs = retriever::sitemap(&url.clone()).await;
    let mut docs = match docs {
        Ok(docs) => docs,
        Err(e) => {
            info!("Error fetching documents: {}", e);
            return (StatusCode::BAD_REQUEST, Json(e.to_string()));
        }
    };
    let duration = start.elapsed();
    info!("Fetched {} docs from {} in {:?}", docs.len(), url, duration);

    let tracker = state.progress_map.clone();

    // spawn a background task
    tokio::spawn(async move {
        info!("Creating Ollama client");
        let ollama = ollama_rs::Ollama::new(ollama_host.to_string(), ollama_port);
        let llm = ollama::Llm::new(ollama);

        let total_docs = docs.len();
        info!("Adding {} documents", total_docs);

        let embedding_progress = EmbeddingProgress::new(total_docs);

        {
            let tracker = tracker.lock();
            tracker.unwrap().insert(id, embedding_progress);
        }

        let (_handle, model) = crate::embedding::Model::spawn(tracker, id);
        let make_summary = filter_collections.contains(&Collection::Summary);

        for doc in docs.iter_mut() {
            if make_summary {
                info!("Creating summary document");
                let result = doc.add_summary(&ollama_model, &llm).await;
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        info!("Error adding summary: {}", e);
                    }
                }
                let embeddings = model.encode(doc.clone()).await;
                let embeddings = match embeddings {
                    Ok(embeddings) => embeddings,
                    Err(e) => {
                        info!("Error encoding document: {}", e);
                        continue;
                    }
                };
                let result = add_documents(
                    &qdrant_client,
                    &base_collection,
                    filter_collections.clone(),
                    embeddings,
                )
                .await;
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        info!("Error adding documents: {}", e);
                    }
                }
            }
        }
    });

    (StatusCode::OK, Json(id.to_string()))
}

// AppError is a wrapper around `anyhow::Error` that implements `IntoResponse`.
// Make our own error that wraps `anyhow::Error`.
pub struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
