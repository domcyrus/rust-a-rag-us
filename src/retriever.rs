use anyhow::{Error, Result};
use log::info;
use scraper::{Html, Selector};

use crate::data::{self, Document};

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
    let selector = Selector::parse(r#"loc"#).unwrap();
    let mut documents = Vec::new();
    for sitemap_url in document.select(&selector) {
        info!("Fetching {}", sitemap_url.inner_html());
        let doc = fetch_content(&sitemap_url.inner_html()).await?;
        documents.push(doc);
    }
    Ok(documents)
}

async fn fetch_content(url: &str) -> Result<data::Document, Error> {
    let resp = reqwest::get(url).await?;

    let body = resp.text().await?;

    // Parse the HTML
    let document = Html::parse_document(&body);

    // Extract the title
    let title_selector = Selector::parse("title").unwrap();
    let title = document
        .select(&title_selector)
        .next()
        .map_or(String::from(""), |n| n.text().collect());

    info!("found title: {} on url: {}", title, url);

    // Create a selector for the body element
    let body_selector = Selector::parse("body").unwrap();

    // Extract the body element
    if let Some(body_element) = document.select(&body_selector).next() {
        // Remove script elements from the body
        let script_selector = Selector::parse("script").unwrap();
        let cleaned_body_html = body_element
            .select(&script_selector)
            .fold(body_element.html(), |acc, script| {
                acc.replace(script.html().as_str(), "")
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
        return Ok(data::Document::new(url.to_string(), title, text_one_liner));
    }
    Err(anyhow::anyhow!("Failed to fetch content"))
}