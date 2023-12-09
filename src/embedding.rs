use crate::data::{Document, EmbeddedDocument, EmbeddedMetadata};
use anyhow::Error;
use std::time::Instant;
use std::{
    sync::mpsc,
    thread::{self, JoinHandle},
};
use tch::Device;
use tokio::{sync::oneshot, task};

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

use log::{debug, info};

// EMBEDDING_SIZE represents the size of the embedding
pub static EMBEDDING_SIZE: u64 = 384;

// Message represents a message
type Message = (Document, oneshot::Sender<Vec<EmbeddedDocument>>);

// Model represents a model
// based on https://github.com/guillaume-be/rust-bert/blob/main/examples/async-sentiment.rs
pub struct Model {
    sender: mpsc::SyncSender<Message>,
}

impl Model {
    // spawn returns a new model and a handle to the model
    pub fn spawn() -> (JoinHandle<anyhow::Result<()>>, Model) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(receiver));
        (handle, Model { sender })
    }

    // runner runs the model
    fn runner(receiver: mpsc::Receiver<Message>) -> anyhow::Result<(), Error> {
        debug!("Starting model runner");
        debug!("Loading remote model");
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .with_device(Device::cuda_if_available())
            .create_model()
            .expect("Could not load model");

        while let Ok((document, sender)) = receiver.recv() {
            let mut embedded_documents = Vec::new();
            let mut document_average_time = vec![];
            let doc_start = Instant::now();
            let fragments = document.to_fragments();
            let it = fragments.text.iter().zip(fragments.meta_text.iter());

            for (text_fragmenent, meta_fragment) in it {
                let fragment_start = Instant::now();
                let text_embedding = model
                    .encode(&[text_fragmenent.clone()])
                    .expect("Could not embed fragment");
                let meta_embedding = model
                    .encode(&[meta_fragment.clone()])
                    .expect("Could not embed fragment");
                embedded_documents.push(EmbeddedDocument {
                    text_embeddings: text_embedding[0].clone(),
                    meta_embeddings: meta_embedding[0].clone(),
                    metadata: EmbeddedMetadata::from_document(
                        &document,
                        text_fragmenent.clone(),
                        meta_fragment.clone(),
                    ),
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
