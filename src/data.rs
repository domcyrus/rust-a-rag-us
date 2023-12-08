use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use uuid::Uuid;

static FRAGMENT_SIZE: usize = 1024;
static OVERLAP_SIZE: usize = 256;
static MAX_TITLE_SIZE: usize = 128;
static MAX_URL_SIZE: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedMetadata {
    pub id: String,
    pub hash: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub timestamp: String,
}

impl EmbeddedMetadata {
    pub fn from_document(document: &Document, text: String) -> Self {
        let id: String =
            Uuid::new_v5(&Uuid::NAMESPACE_OID, document.hash.clone().as_bytes()).to_string();
        EmbeddedMetadata {
            id: id,
            hash: document.hash.clone(),
            title: document.title.clone(),
            url: document.url.clone(),
            text: text,
            timestamp: document.timestamp.to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddedDocument {
    pub embeddings: Vec<f32>,
    pub metadata: EmbeddedMetadata,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub hash: String,
    pub title: String,
    pub url: String,
    pub text: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Document {
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

    pub fn to_fragments(&self) -> Vec<String> {
        // split text into chunks of FRAGMENT_SIZE characters. Overlap by OVERLAP_SIZE characters
        // TODO(marco): this is a very naive implementation. We should split by sentences or at least by words instead
        let mut document_fragments = Vec::new();
        let mut start = 0;
        let char_vec: Vec<char> = self.text.chars().collect();
        if char_vec.len() == 0 {
            return document_fragments;
        }
        // split by FRAGMENT_SIZE, but overlap by 256 characters
        let mut end = FRAGMENT_SIZE;
        while end < char_vec.len() {
            let fragment = &char_vec[start..end];
            let fragment_string: String = fragment.iter().collect();
            // truncate title to MAX_TITLE_SIZE characters
            let title = if self.title.len() > MAX_TITLE_SIZE {
                &self.title[..MAX_TITLE_SIZE]
            } else {
                &self.title
            };

            // truncate url to MAX_URL_SIZE characters
            let url = if self.url.len() > MAX_URL_SIZE {
                &self.url[..MAX_URL_SIZE]
            } else {
                &self.url
            };
            // prefix the fragment with the title
            let fragment_string =
                format!("Title: {} URL: {} Content: {}", title, url, fragment_string);
            document_fragments.push(fragment_string);
            start += FRAGMENT_SIZE - OVERLAP_SIZE;
            end += FRAGMENT_SIZE - OVERLAP_SIZE;
        }

        return document_fragments;
    }
}
