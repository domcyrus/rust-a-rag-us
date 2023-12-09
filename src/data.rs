use chrono::prelude::*;
use log::error;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use text_splitter::TextSplitter;
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

// EmbeddedMetadata represents metadata embedded in a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedMetadata {
    pub id: String,
    pub hash: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub meta_text: String,
    pub timestamp: String,
}

impl EmbeddedMetadata {
    // from_document returns a new EmbeddedMetadata from a document
    pub fn from_document(document: &Document, text: String, meta_text: String) -> Self {
        let id: String =
            Uuid::new_v5(&Uuid::NAMESPACE_OID, document.hash.clone().as_bytes()).to_string();
        EmbeddedMetadata {
            id: id,
            hash: document.hash.clone(),
            title: document.title.clone(),
            url: document.url.clone(),
            text: text,
            meta_text: meta_text,
            timestamp: document.timestamp.to_rfc3339(),
        }
    }
}

// EmbeddedDocument represents a document with embeddings
#[derive(Debug, Clone)]
pub struct EmbeddedDocument {
    pub text_embeddings: Vec<f32>,
    pub meta_embeddings: Vec<f32>,
    pub metadata: EmbeddedMetadata,
}

// Document represents a document
#[derive(Debug, Clone)]
pub struct Document {
    pub hash: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Fragments represents fragments of a document
#[derive(Debug, Clone)]
pub struct Fragments {
    pub text: Vec<String>,
    pub meta_text: Vec<String>,
}

impl Document {
    // new returns a new document
    pub fn new(url: String, title: String, text: String) -> Self {
        // generate id as hash from url and text to avoid duplicates
        let hash_text = format!("{}{}", url, text);
        let mut hasher = Sha1::new();
        hasher.update(hash_text);
        let result = hasher.finalize();

        Document {
            hash: format!("{:x}", result),
            title: title,
            url: url,
            text: text,
            timestamp: Utc::now(),
        }
    }

    // to_fragments returns a vector of fragments of the document
    pub fn to_fragments(&self) -> Fragments {
        // split text into chunks of FRAGMENT_SIZE characters. Overlap by OVERLAP_SIZE characters
        let splitter = TextSplitter::default().with_trim_chunks(true);

        // truncate title to MAX_TITLE_SIZE characters
        let title = splitter.chunks(&self.title, MAX_TITLE_SIZE).next();

        // truncate url to MAX_URL_SIZE characters
        let url = splitter.chunks(&self.url, MAX_URL_SIZE).next();

        let text_result = splitter.chunks(&self.text, FRAGMENT_SIZE..OVERLAP_SIZE + FRAGMENT_SIZE);
        let meta_result = splitter.chunks(&self.text, META_FRAGMENT_SIZE);

        match (title, url) {
            (Some(title), Some(url)) => Fragments {
                text: text_result
                    .map(|s| format!("Title: {} URL: {} Content: {}", title, url, s))
                    .collect(),
                meta_text: meta_result
                    .map(|s| format!("Title: {} URL: {} Content: {}", title, url, s))
                    .collect(),
            },
            _ => {
                error!("Error splitting text, title or url not found");
                Fragments {
                    text: text_result.map(|s| format!("Content: {}", s)).collect(),
                    meta_text: meta_result.map(|s| format!("Content: {}", s)).collect(),
                }
            }
        }
    }
}
