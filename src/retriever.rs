use std::sync::Arc;

use crate::data::{self, Document};
use anyhow::{Error, Result};
use log::info;
use scraper::{Html, Selector};
use tokio::sync::Semaphore;
use tokio::task;

// get_urls returns a vector of urls from a sitemap.xml
//
// function needs to be non async because scraper::Html is not Send, grmbl
fn get_urls(body: String) -> Result<Vec<String>, Error> {
    let document = Html::parse_document(&body);
    let selector =
        Selector::parse(r#"loc"#).or(Err(anyhow::anyhow!("Failed to parse loc selector")))?;

    let mut urls = Vec::new();
    for sitemap_url in document.select(&selector) {
        info!("Fetching {}", sitemap_url.inner_html());
        // TODO(marco): handle recursive sitemaps
        urls.push(sitemap_url.inner_html().to_string());
    }
    Ok(urls)
}

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
    let urls = get_urls(text)?;
    let bodies = fetch_bodies(urls).await?;
    let documents = parse_contents(bodies)?;
    Ok(documents)
}

static CONCURRENT_REQUESTS: usize = 10;

// Body is a struct containing a url and a body
struct Body {
    url: String,
    body: String,
}

// fetch_bodies returns a vector of bodies from a vector of urls
async fn fetch_bodies(urls: Vec<String>) -> Result<Vec<Body>, Error> {
    let now = std::time::Instant::now();
    let semaphore = Arc::new(Semaphore::new(CONCURRENT_REQUESTS));
    let mut tasks = Vec::new();

    for url in urls {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = reqwest::Client::new(); // Moved outside the task
        let task = task::spawn(async move {
            let response = match client.get(&url).send().await {
                Ok(resp) => resp,
                Err(err) => return Err(anyhow::anyhow!("Error fetching URL {}: {}", url, err)),
            };

            let body_text = response.text().await?;
            drop(permit);
            Ok(Body {
                url,
                body: body_text,
            })
        });
        tasks.push(task);
    }

    let mut bodies = Vec::new();
    for task in tasks {
        match task.await {
            Ok(result) => bodies.push(result?),
            Err(e) => return Err(anyhow::anyhow!("Task error: {}", e)),
        }
    }
    info!("Fetched {} bodies in {:?}", bodies.len(), now.elapsed());
    Ok(bodies)
}

// parse_contents returns a vector of documents from a vector of bodies
//
// function needs to be non async because scraper::Html is not Send, grmbl
fn parse_contents(bodies: Vec<Body>) -> Result<Vec<Document>, Error> {
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
pub async fn fetch_content(url: String) -> Result<Document, Error> {
    let resp = reqwest::get(url.clone()).await?;
    let body = resp.text().await?;

    let documents = parse_contents(vec![Body {
        url: url,
        body: body,
    }])?;
    if documents.len() != 1 {
        return Err(anyhow::anyhow!(
            "Failed to parse content, expected 1 document, got: {}",
            documents.len()
        ));
    }

    return Ok(documents[0].clone());
}
