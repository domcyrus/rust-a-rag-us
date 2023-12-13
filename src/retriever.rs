use std::sync::Arc;

use crate::data::{self, Document};
use anyhow::{Error, Result};
use log::info;
use scraper::{Html, Selector};
use tokio::sync::Semaphore;
use tokio::task;

// sitemap returns a vector of documents from a sitemap.xml
pub async fn sitemap(url: &str) -> Result<Vec<Document>, Error> {
    let mut url_with_sitemap: String = url.to_string();
    if !url_with_sitemap.ends_with("sitemap.xml") {
        url_with_sitemap.push_str("/sitemap.xml");
    }
    let resp = match reqwest::get(url_with_sitemap).await {
        Ok(x) => x,
        Err(err) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch sitemap: {}",
                err.to_string()
            ))
        }
    };
    let text = resp.text().await?;
    let document = Html::parse_document(&text);
    let selector =
        Selector::parse(r#"loc"#).or(Err(anyhow::anyhow!("Failed to parse loc selector")))?;

    let mut urls = Vec::new();
    for sitemap_url in document.select(&selector) {
        info!("Fetching {}", sitemap_url.inner_html());
        // TODO(marco): handle recursive sitemaps
        urls.push(sitemap_url.inner_html().to_string());
    }
    let bodies = fetch_bodies(urls).await?;
    let documents = parse_contents(bodies).await?;
    Ok(documents)
}

static CONCURRENT_REQUESTS: usize = 10;

// Body is a struct containing a url and a body
struct Body {
    url: String,
    body: String,
}

async fn fetch_bodies(urls: Vec<String>) -> Result<Vec<Body>, Error> {
    let now = std::time::Instant::now();
    let semaphore = Arc::new(Semaphore::new(CONCURRENT_REQUESTS));
    let mut bodies = Vec::new();
    let mut tasks = Vec::new();
    for url in urls {
        let permit = semaphore.clone().acquire_owned().await?;
        let task = task::spawn(async move {
            let client = reqwest::Client::new();
            let response = client.get(&url).send().await;
            let response = match response {
                Ok(x) => x,
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to fetch url: {} with error: {}",
                        url,
                        err.to_string()
                    ))
                }
            };

            // Read the response body
            let body = Body {
                url: url,
                body: response.text().await?,
            };
            drop(permit); // Release the permit
            Ok(body) // Return the body text
        });
        tasks.push(task);
    }

    for task in tasks {
        let body = task.await?;
        let body = body?;
        bodies.push(body);
    }
    info!(
        "Fetched {} bodies in {:?} seconds",
        bodies.len(),
        now.elapsed(),
    );

    Ok(bodies)
}

async fn parse_contents(bodies: Vec<Body>) -> Result<Vec<Document>, Error> {
    let now = std::time::Instant::now();
    let mut results = Vec::new();
    for body in bodies {
        // Parse the HTML
        let document = Html::parse_document(&body.body);

        // Extract the title
        let title_selector =
            Selector::parse("title").or(Err(anyhow::anyhow!("Failed to parse title selector")))?;

        let title = document
            .select(&title_selector)
            .next()
            .map_or(String::from(""), |n| n.text().collect());

        info!("found title: {}", title);

        // Create a selector for the body element
        let body_selector =
            Selector::parse("body").or(Err(anyhow::anyhow!("Failed to parse body selector")))?;

        // Extract the body element
        if let Some(body_element) = document.select(&body_selector).next() {
            // Remove script and nav elements from the body
            let unwanted_selector = Selector::parse("script, nav")
                .or(Err(anyhow::anyhow!("Failed to parse unwanted selector")))?;
            let cleaned_body_html = body_element
                .select(&unwanted_selector)
                .fold(body_element.html(), |acc, unwanted| {
                    acc.replace(unwanted.html().as_str(), "")
                });

            // Parse the cleaned body HTML
            let cleaned_body_document = Html::parse_fragment(&cleaned_body_html);
            let text_one_liner =
                cleaned_body_document
                    .root_element()
                    .text()
                    .fold(String::from(""), |acc, node| {
                        let text = node.trim();
                        if text.len() > 0 {
                            format!("{} {}", acc, text)
                        } else {
                            acc
                        }
                    });
            results.push(Document::new(
                data::Collection::Basic,
                body.url,
                title,
                text_one_liner,
            ));
        }
    }
    info!(
        "Parsed {} documents in {:?} seconds",
        results.len(),
        now.elapsed()
    );
    Ok(results)
}

// fetch_content returns a document from a url
pub async fn fetch_content(url: &str) -> Result<Document, Error> {
    let resp = reqwest::get(url).await?;
    let body = resp.text().await?;

    let documents = parse_contents(vec![Body {
        url: url.to_string(),
        body: body,
    }])
    .await?;
    if documents.len() != 1 {
        return Err(anyhow::anyhow!(
            "Failed to parse content, expected 1 document, got: {}",
            documents.len()
        ));
    }

    return Ok(documents[0].clone());
}
