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

pub static EMBEDDING_SIZE: u64 = 384;

type Message = (Document, oneshot::Sender<Vec<EmbeddedDocument>>);

pub struct Model {
    sender: mpsc::SyncSender<Message>,
}

impl Model {
    pub fn spawn() -> (JoinHandle<anyhow::Result<()>>, Model) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(receiver));
        (handle, Model { sender })
    }

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
            for fragmenent in document.to_fragments() {
                let fragment_start = Instant::now();
                let embedding = model
                    .encode(&[fragmenent.clone()])
                    .expect("Could not embed fragment");
                embedded_documents.push(EmbeddedDocument {
                    embeddings: embedding[0].clone(),
                    metadata: EmbeddedMetadata::from_document(&document, fragmenent),
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

    pub async fn encode(&self, document: Document) -> Result<Vec<EmbeddedDocument>, Error> {
        let (sender, receiver) = oneshot::channel();
        task::block_in_place(|| self.sender.send((document, sender)))?;
        Ok(receiver.await?)
    }
}

pub async fn text_embedding_async(text: String) -> Vec<f32> {
    let handle = tokio::task::spawn_blocking(move || {
        let embeds = get_text_embedding(&text);
        embeds
    });

    let res = handle.await.unwrap();
    res
}

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
