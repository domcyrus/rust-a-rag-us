use crate::ollama::Llm;
use anyhow::Error;
use chrono::prelude::*;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::collections::HashMap;
use text_splitter::TextSplitter;
use utoipa::ToSchema;
use uuid::Uuid;

// FRAGMENT_SIZE is the size of a fragment
static FRAGMENT_SIZE: usize = 1512;
// OVERLAP_SIZE is the size of the overlap between fragments
static OVERLAP_SIZE: usize = 256;
// MAX_TITLE_SIZE is the maximum size of a title
static MAX_TITLE_SIZE: usize = 128;
// MAX_URL_SIZE is the maximum size of a url
static MAX_URL_SIZE: usize = 128;
// META_FRAGMENT_SIZE is the size of the meta embedding
pub static META_FRAGMENT_SIZE: usize = 384;

// Collection represents a collection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, ToSchema)]
pub enum Collection {
    Basic,
    Summary,
}

impl Collection {
    // all returns all collections
    pub fn all() -> Vec<Collection> {
        vec![Collection::Basic, Collection::Summary]
    }

    // limit by collection
    pub fn limit_by_collection(&self) -> f32 {
        match self {
            // basic collection is weighted higher
            Collection::Basic => 0.8,
            // summary collection is weighted lower
            Collection::Summary => 0.2,
        }
    }
}

// collection to string
impl ToString for Collection {
    fn to_string(&self) -> String {
        match self {
            Collection::Basic => "basic".to_string(),
            Collection::Summary => "summary".to_string(),
        }
    }
}

// string to collection
impl From<&str> for Collection {
    fn from(s: &str) -> Self {
        match s {
            "basic" => Collection::Basic,
            "summary" => Collection::Summary,
            _ => {
                error!("Error converting collection, unknown collection: {}", s);
                Collection::Basic
            }
        }
    }
}

// EmbeddedMetadata represents metadata embedded in a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedMetadata {
    pub id: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub timestamp: String,
    pub collection: Collection,
}

impl EmbeddedMetadata {
    // from_document returns a new EmbeddedMetadata from a document
    pub fn from_document(
        document: &Document,
        text: String,
        collection: Collection,
    ) -> Result<Self, Error> {
        // get hash from collection map
        // generate id as hash from url and text to avoid duplicates
        let hash_text = format!("{}{}", document.url, text);
        let mut hasher = Sha1::new();
        hasher.update(hash_text);
        let hash = hasher.finalize();
        let hash = format!("{:x}", hash);
        let id: String = Uuid::new_v5(&Uuid::NAMESPACE_OID, hash.as_bytes()).to_string();
        Ok(EmbeddedMetadata {
            id: id,
            title: document.title.clone(),
            url: document.url.clone(),
            text: text,
            timestamp: document.timestamp.to_rfc3339(),
            collection: collection,
        })
    }
}

// EmbeddedDocument represents a document with embeddings
#[derive(Debug, Clone)]
pub struct EmbeddedDocument {
    pub text_embeddings: Vec<f32>,
    pub metadata: EmbeddedMetadata,
}

// Document represents a document
#[derive(Debug, Clone)]
pub struct Document {
    pub title: String,
    pub url: String,
    pub text: HashMap<Collection, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Fragment represents a fragment of a document
#[derive(Debug, Clone)]
pub struct Fragment {
    pub text: String,
    pub collection: Collection,
}

impl Document {
    // new returns a new document from a url, title and text using collection.
    pub fn new(collection: Collection, url: String, title: String, text: String) -> Self {
        let mut text_map = HashMap::new();
        text_map.insert(collection, text);

        Document {
            title: title,
            url: url,
            text: text_map.clone(),
            timestamp: Utc::now(),
        }
    }

    pub fn update_text(&mut self, collection: Collection, text: String) {
        debug!(
            "Updating text {} for collection: {}",
            text,
            collection.to_string()
        );
        self.text.insert(collection, text);
    }

    // to_fragments returns a vector of fragments of the document
    pub fn to_fragments(&self) -> Result<Vec<Fragment>, Error> {
        info!("Splitting text into fragments by collections",);

        // split text into chunks of FRAGMENT_SIZE characters. Overlap by OVERLAP_SIZE characters
        let splitter = TextSplitter::default().with_trim_chunks(true);

        // truncate title to MAX_TITLE_SIZE characters
        let title = splitter.chunks(&self.title, MAX_TITLE_SIZE).next();

        // truncate url to MAX_URL_SIZE characters
        let url = splitter.chunks(&self.url, MAX_URL_SIZE).next();

        let mut result = Vec::new();
        for (collection, text) in &self.text {
            info!("Collection: {}", collection.to_string());
            let text_results = splitter.chunks(&text, FRAGMENT_SIZE..OVERLAP_SIZE + FRAGMENT_SIZE);
            for text_result in text_results {
                let title = title.clone();
                let url = url.clone();
                match (title, url) {
                    (Some(title), Some(url)) => {
                        result.push(Fragment {
                            text: format!("Title: {} URL: {} Content: {}", title, url, text_result),
                            collection: collection.clone(),
                        });
                    }
                    _ => {
                        error!("Error splitting text, title or url not found");
                        Err(anyhow::anyhow!(
                            "Error splitting text, title or url not found"
                        ))?
                    }
                }
            }
        }
        Ok(result)
    }

    pub async fn add_summary(&mut self, model: &str, llm: &Llm) -> Result<(), Error> {
        // retrieve the basic collection text
        let basic_text = self.text.get(&Collection::Basic);
        match basic_text {
            Some(basic_text) => {
                // get summary
                let summary = llm.summarize(model, basic_text).await?;
                // update text with summary
                self.update_text(Collection::Summary, summary);
                Ok(())
            }
            None => {
                error!("Error adding summary, basic text not found");
                Err(anyhow::anyhow!(
                    "Error adding summary, basic text not found"
                ))
            }
        }
    }
}
