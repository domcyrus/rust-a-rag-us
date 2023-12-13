use axum::{routing::get, routing::post, Router};
use dotenv::dotenv;
use log::info;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use rust_a_rag_us::api::{get_state, upload, ApiDoc};
use rust_a_rag_us::embedding::EmbeddingProgress;
use rust_a_rag_us::state::{AppConfigInput, AppState};
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[tokio::main]
async fn main() {
    dotenv().ok();
    env_logger::init();

    let qdrant_client_address =
        std::env::var("QDRANT_CLIENT_ADDRESS").unwrap_or("http://localhost:6334".to_string());
    let qdrant_client =
        QdrantClient::new(Some(QdrantClientConfig::from_url(&qdrant_client_address))).unwrap();

    let app_config_input = AppConfigInput {
        address: Some(std::env::var("ADDRESS").unwrap_or("127.0.0.1:3000".to_string())),
        base_collection: Some(
            std::env::var("BASE_COLLECTION").unwrap_or("rura_collection".to_string()),
        ),
        filter_collections: Some(vec![rust_a_rag_us::data::Collection::Basic]),
        ollama_model: Some(
            std::env::var("OLLAMA_MODEL").unwrap_or("openhermes2.5-mistral:7b-q6_K".to_string()),
        ),
        ollama_host: Some(std::env::var("OLLAMA_HOST").unwrap_or("localhost".to_string())),
        ollama_port: Some(
            std::env::var("OLLAMA_PORT")
                .unwrap_or("11434".to_string())
                .parse::<u16>()
                .unwrap(),
        ),
        qdrant_client: Some(qdrant_client),
    };
    let state = Arc::new(AppState::<EmbeddingProgress>::new(app_config_input).unwrap());
    let listener = tokio::net::TcpListener::bind(state.app_config.address.as_str())
        .await
        .unwrap();

    let app = Router::new()
        .route("/get-state", get(get_state))
        .route("/upload", post(upload))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs", ApiDoc::openapi()))
        .layer(axum::Extension(state));

    info!("listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
