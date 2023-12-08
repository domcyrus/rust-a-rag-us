use ollama_rs::{
    generation::completion::{request::GenerationRequest, GenerationResponseStream},
    Ollama,
};
use tokio::io::{stdout, AsyncWriteExt};
use tokio_stream::StreamExt;

pub struct Llm {
    ollama: Ollama,
}

impl Llm {
    pub fn new(ollama: Ollama) -> Self {
        Llm { ollama: ollama }
    }

    pub async fn generate(&self, model: String, prompt: String) -> Result<String, anyhow::Error> {
        let res = self
            .ollama
            .generate(GenerationRequest::new(model, prompt))
            .await;
        match res {
            Ok(res) => {
                return Ok(res.response);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Error generating text: {}", e));
            }
        }
    }
    pub async fn generate_stream(
        &self,
        model: String,
        prompt: String,
    ) -> Result<(), anyhow::Error> {
        let mut stream: GenerationResponseStream = self
            .ollama
            .generate_stream(GenerationRequest::new(model, prompt))
            .await?;
        let mut stdout = stdout();
        while let Some(Ok(res)) = stream.next().await {
            stdout.write_all(res.response.as_bytes()).await?;
            stdout.flush().await?;
        }
        Ok(())
    }
}

pub static PROMPT: &str = r#"You are a customer support agent. You are designed to be as helpful as possible while providing only factual information. You should be friendly, but not overly chatty. Context information is below. Given the context information and not prior knowledge, answer the query.
Context:
{context}

Question: {question}
Helpful answer thats includes a heading derived from the question:"#;

//pub static PROMPT: &str = r#"Answer the question based on the context below. Please provide a detailed and structured and well formated response.
//The output should be structured with a heading derived from the question below, and use a star to designate a list item.
//If the question cannont be answered using the information provider answer with "I dont't Know".
//
//Context:
//{context}
//
//Question: {question}
//Helpful answer thats includes a heading derived from the question:"#;
//
