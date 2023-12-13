use crate::data::Collection;
use crate::progress_tracker::ProgressTracker;
use anyhow::{Error, Result};
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use uuid::Uuid;

pub struct AppConfig {
    pub address: String,
    pub base_collection: String,
    pub filter_collections: Vec<Collection>,
    pub ollama_model: String,
    pub ollama_host: String,
    pub ollama_port: u16,
    pub qdrant_client: Arc<QdrantClient>,
}

pub struct AppState<T: ProgressTracker> {
    pub progress_map: Arc<Mutex<HashMap<Uuid, T>>>,
    pub app_config: AppConfig,
}

#[derive(Default)]
pub struct AppConfigInput {
    pub address: Option<String>,
    pub base_collection: Option<String>,
    pub filter_collections: Option<Vec<Collection>>,
    pub ollama_model: Option<String>,
    pub ollama_host: Option<String>,
    pub ollama_port: Option<u16>,
    pub qdrant_client: Option<QdrantClient>,
}

impl<T: ProgressTracker> AppState<T> {
    pub fn new(app_config_input: AppConfigInput) -> Result<Self, Error> {
        // TODO: define the default values in one place
        let filter_collection: Vec<Collection> = app_config_input
            .filter_collections
            .unwrap_or(vec![Collection::Basic]);
        let address = "http://localhost:6334";
        let qdrant_config = QdrantClientConfig::from_url(address);
        let qdrant_client = match app_config_input.qdrant_client {
            Some(qdrant_client) => qdrant_client,
            None => QdrantClient::new(Some(qdrant_config))?,
        };
        Ok(AppState {
            progress_map: Arc::new(Mutex::new(HashMap::new())),
            app_config: AppConfig {
                address: app_config_input
                    .address
                    .unwrap_or("127.0.0.1:3000".to_string()),
                base_collection: app_config_input
                    .base_collection
                    .unwrap_or("rura_collection".to_string()),
                filter_collections: filter_collection,
                ollama_model: app_config_input
                    .ollama_model
                    .unwrap_or("openhermes2.5-mistral:7b-q6_K".to_string()),
                ollama_host: app_config_input
                    .ollama_host
                    .unwrap_or("localhost".to_string()),
                ollama_port: app_config_input.ollama_port.unwrap_or(11434),
                qdrant_client: Arc::new(qdrant_client),
            },
        })
    }

    pub fn get_all_progress(&self) -> std::sync::MutexGuard<HashMap<Uuid, T>> {
        self.progress_map.lock().unwrap()
    }
}
