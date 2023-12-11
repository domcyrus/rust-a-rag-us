use log::debug;
use ollama_rs::{
    generation::completion::{request::GenerationRequest, GenerationResponseStream},
    Ollama,
};
use tokio::io::{stdout, AsyncWriteExt};
use tokio_stream::StreamExt;

// Llm is a wrapper around the Ollama client
pub struct Llm {
    ollama: Ollama,
}

impl Llm {
    // new creates a new Llm
    pub fn new(ollama: Ollama) -> Self {
        Llm { ollama: ollama }
    }

    // generate generates text from a prompt
    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String, anyhow::Error> {
        let res = self
            .ollama
            .generate(GenerationRequest::new(
                model.to_string(),
                prompt.to_string(),
            ))
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
    // generate_stream generates a stream of text currently hardwired to stdout from a prompt
    pub async fn generate_stream(&self, model: &str, prompt: &str) -> Result<(), anyhow::Error> {
        let mut stream: GenerationResponseStream = self
            .ollama
            .generate_stream(GenerationRequest::new(
                model.to_string(),
                prompt.to_string(),
            ))
            .await?;
        let mut stdout = stdout();
        while let Some(Ok(res)) = stream.next().await {
            stdout.write_all(res.response.as_bytes()).await?;
            stdout.flush().await?;
        }
        Ok(())
    }
    pub async fn summarize(&self, model: &str, text: &str) -> Result<String, anyhow::Error> {
        let formatted_prompt = PROMPT_SUMMARY.replace("{context}", text);
        debug!("Formatted summary prompt: {}", formatted_prompt);
        self.generate(model, &formatted_prompt).await
    }
}

pub static PROMPT: &str = r#"You are a customer support agent, programmed to offer highly accurate and helpful assistance. Your responses should be strictly based on factual information, presented in a friendly yet concise manner. Utilize only the context information provided below, without drawing on any prior knowledge. Your goal is to address the query directly and efficiently, ensuring clarity and relevance in your answer.
Context:
{context}

Question: {question}
Helpful answer thats includes a heading derived from the question:"#;

//pub static PROMPT_SUMMARY: &str = r#"You are an advanced summarization agent, your objective is to craft a succinct and precise summary using only the context information given. Your approach should center on extracting and condensing the critical elements and core details into a brief and clear format. Avoid referencing the creation of a summary in your output or stating that it's a summary.
pub static PROMPT_SUMMARY: &str = r#"Your role as an advanced summarization agent involves distilling the provided context information into a concise and precise format. Emphasize extracting and synthesizing the main points and critical details, presenting them in a clear, compact form. In your output, seamlessly integrate these key elements without explicitly labeling the output as a summary or indicating the summarization process.
Context:
{context}
"#;
