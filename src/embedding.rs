use crate::data::{Document, EmbeddedDocument, EmbeddedMetadata};
use crate::progress_tracker::ProgressTracker;
use anyhow::{Error, Result};
use log::info;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{
    sync::mpsc,
    thread::{self, JoinHandle},
};
use tch::Device;
use tokio::{sync::oneshot, task};
use uuid::Uuid;

// EMBEDDING_SIZE represents the size of the embedding
pub static EMBEDDING_SIZE: u64 = 384;

// Message represents a message
type Message = (Document, oneshot::Sender<Vec<EmbeddedDocument>>);

// EmbeddingProgress represents the progress of an embedding task
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct EmbeddingProgress {
    total_documents: usize,
    processed_documents: usize,
}

impl ProgressTracker for EmbeddingProgress {
    fn new(total_documents: usize) -> Self {
        EmbeddingProgress {
            total_documents: total_documents,
            processed_documents: 0,
        }
    }

    // increment_total increments the total documents
    fn increment_processed(&mut self) {
        self.processed_documents += 1;
    }

    // progress_status returns the current progress status
    fn progress_status(&self) -> (usize, usize) {
        (self.processed_documents, self.total_documents)
    }
}

// Model represents a model
// based on https://github.com/guillaume-be/rust-bert/blob/main/examples/async-sentiment.rs
pub struct Model {
    sender: mpsc::SyncSender<Message>,
}

impl Model {
    // spawn returns a new model and a handle to the model
    pub fn spawn(
        progress_state: Arc<Mutex<HashMap<Uuid, EmbeddingProgress>>>,
        id: Uuid,
    ) -> (JoinHandle<anyhow::Result<()>>, Model) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(receiver, progress_state, id));
        (handle, Model { sender })
    }

    // runner runs the model
    fn runner(
        receiver: mpsc::Receiver<Message>,
        progress_state: Arc<Mutex<HashMap<Uuid, EmbeddingProgress>>>,
        id: Uuid,
    ) -> anyhow::Result<(), Error> {
        info!("Loading remote embedding model");
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .with_device(Device::cuda_if_available())
            .create_model()
            .expect("Could not load model");

        while let Ok((document, sender)) = receiver.recv() {
            let mut embedded_documents = Vec::new();
            let mut document_average_time = vec![];
            let doc_start = Instant::now();
            let fragments = document.to_fragments()?;
            for fragment in fragments {
                let fragment_start = Instant::now();
                let text_embedding = model
                    .encode(&[fragment.text.clone()])
                    .expect("Could not embed fragment");
                embedded_documents.push(EmbeddedDocument {
                    text_embeddings: text_embedding[0].clone(),
                    metadata: EmbeddedMetadata::from_document(
                        &document,
                        fragment.text.clone(),
                        fragment.collection.clone(),
                    )?,
                });
                document_average_time.push(fragment_start.elapsed());
            }
            document_average_time.push(doc_start.elapsed());
            info!("Documents embedded in {:?}", doc_start.elapsed());

            let mut total_time = 0;
            for time in &document_average_time {
                total_time += time.as_millis();
            }

            let total_items = &document_average_time.len();
            let average_time = total_time / *total_items as u128;
            info!("Average time per document: {}ms", average_time);
            info!("Total Items: {}", total_items);

            sender.send(embedded_documents).expect("sending results");
            let state = progress_state.lock();
            match state {
                Ok(mut state) => {
                    if let Some(s) = state.get_mut(&id) {
                        s.increment_processed();
                    } else {
                        return Err(anyhow::anyhow!("Failed to get state"));
                    }
                }
                Err(_) => {
                    return Err(anyhow::anyhow!("Failed to get state"));
                }
            }
        }

        Ok(())
    }

    // encode returns a vector of embedded documents
    pub async fn encode(&self, document: Document) -> Result<Vec<EmbeddedDocument>, Error> {
        let (sender, receiver) = oneshot::channel();
        task::block_in_place(|| self.sender.send((document, sender)))?;
        Ok(receiver.await?)
    }
}

// text_embedding_async returns a text embedding for a given text in a as
pub async fn text_embedding_async(text: String) -> Vec<f32> {
    let handle = tokio::task::spawn_blocking(move || {
        let embeds = get_text_embedding(&text);
        embeds
    });

    let res = handle.await.unwrap();
    res
}

// get_text_embedding returns a text embedding for a given text
pub fn get_text_embedding(text: &str) -> Vec<f32> {
    let model_start = Instant::now();
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()
        .expect("Could not create model");
    info!("Model started in {:?}", model_start.elapsed());

    let embedding_start = Instant::now();
    let embedding = model
        .encode(&[text.to_string()])
        .expect("Could not embed fragment");
    info!("Embedding generated in {:?}", embedding_start.elapsed());
    embedding[0].clone()
}
