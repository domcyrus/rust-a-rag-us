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
    // generate_stream generates a stream of text currently hardwired to stdout from a prompt
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

pub static PROMPT: &str = r#"You are a customer support agent, programmed to offer highly accurate and helpful assistance. Your responses should be strictly based on factual information, presented in a friendly yet concise manner. Utilize only the context information provided below, without drawing on any prior knowledge. Your goal is to address the query directly and efficiently, ensuring clarity and relevance in your answer.
Context:
{context}

Question: {question}
Helpful answer thats includes a heading derived from the question:"#;

pub static PROMPT_SUMMARY: &str = r#"You are an advanced summary agent. Your task is to generate a concise and accurate summary based solely on the context information provided below. Do not rely on prior knowledge. Focus on distilling the key points and essential details into a brief, coherent summary.
Context:
{context}
"#;
