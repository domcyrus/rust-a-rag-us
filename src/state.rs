use crate::data::Collection;
use crate::progress_tracker::ProgressTracker;
use std::{collections::HashMap, sync::Mutex};
use uuid::Uuid;

pub struct AppConfig {
    pub address: String,
    pub base_collection: String,
    pub filter_collections: Vec<Collection>,
    pub ollama_model: String,
    pub ollama_host: String,
    pub ollama_port: u16,
}

pub struct AppState<T: ProgressTracker> {
    pub progress_map: Mutex<HashMap<Uuid, T>>,
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
}

impl<T: ProgressTracker> AppState<T> {
    pub fn new(app_config_input: AppConfigInput) -> Self {
        // TODO: define the default values in one place
        let filter_collection: Vec<Collection> = app_config_input
            .filter_collections
            .unwrap_or(vec![Collection::Basic]);
        AppState {
            progress_map: Mutex::new(HashMap::new()),
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
            },
        }
    }

    pub fn get_all_progress(&self) -> std::sync::MutexGuard<HashMap<Uuid, T>> {
        self.progress_map.lock().unwrap()
    }
}
