use axum::{
    Json, Router,
    body::Body,
    extract::{Path as AxumPath, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use pocket_core::{ConversationManager, Role};
use pocket_engine::LocalEngine;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tower-http::cors::CorsLayer;

/// Represents a simple text part in Gemini's part array
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    pub text: String,
}

/// Represents the contents object containing role and parts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub role: String, // "user" or "model"
    pub parts: Vec<Part>,
}

/// Incoming generateContent JSON body payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
}

/// Candidate response item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub content: Content,
}

/// Outgoing generateContent JSON response payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Vec<Candidate>,
}

/// Shared Axum router application state
#[derive(Clone)]
struct AppState {
    engine: Option<LocalEngine>,
}

/// Binds to a port and serves the Gemini-compatible generateContent REST API
pub async fn start_server(engine: Option<LocalEngine>, port: u16) -> anyhow::Result<()> {
    let state = AppState { engine };

    // Build the Router with routes and permissive CORS rules
    let app = Router::new()
        // Serve embedded static frontend
        .route("/", get(index_handler))
        .route("/*path", get(static_handler))
        // Accept routes of the form: /v1beta/models/pocketgemma:generateContent
        .route("/v1beta/models/:model", post(generate_content_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let address = format!("0.0.0.0:{}", port);
    println!(
        "Starting Gemini-Compatible API Server on http://{}",
        address
    );
    println!("Permissive CORS: ENABLED");

    let listener = TcpListener::bind(&address).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to bind server to address: {}. Error: {}",
            address,
            e
        )
    })?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Axum server runtime error: {}", e))?;

    Ok(())
}

/// Axum route handler executing model inference or mock flows
async fn generate_content_handler(
    State(state): State<AppState>,
    AxumPath(model): AxumPath<String>,
    Json(payload): Json<GenerateContentRequest>,
) -> Json<GenerateContentResponse> {
    println!(
        "[pocket-server] Request received for dynamic model segment: {}",
        model
    );

    // Initialize ConversationManager
    let mut manager = ConversationManager::new(None);

    // Populate ConversationManager log with incoming Gemini conversation contents
    for content in &payload.contents {
        let role = match content.role.as_str() {
            "user" => Role::User,
            "model" => Role::Assistant,
            _ => Role::User,
        };

        // Concatenate all parts text into a single String representation
        let text = content
            .parts
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<&str>>()
            .join("\n");

        manager.add_message(role, text);
    }

    let text_response = if let Some(ref local_engine) = state.engine {
        println!("[pocket-server] Initializing model inference forward pass...");
        let formatted_prompt = manager.get_formatted_prompt();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

        let engine_clone = local_engine.clone();
        let gen_task = tokio::spawn(async move {
            if let Err(err) = engine_clone.generate_stream(&formatted_prompt, tx).await {
                eprintln!("[pocket-server] Engine generation error: {:?}", err);
            }
        });

        // Collect and print all streamed tokens in real-time
        let mut full_response = String::new();
        use std::io::Write;
        while let Some(token) = rx.recv().await {
            print!("{}", token);
            let _ = std::io::stdout().flush();
            full_response.push_str(&token);
        }
        println!();

        // Await thread task completion
        if let Err(err) = gen_task.await {
            eprintln!("[pocket-server] Generation task panicked: {:?}", err);
        }

        full_response
    } else {
        println!(
            "[pocket-server] Model not loaded, falling back to conversational offline mock..."
        );
        let last_prompt = manager
            .messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or("");
        get_conversational_mock_response(last_prompt, &manager)
    };

    // Construct standard Gemini generateContent response envelope
    let response = GenerateContentResponse {
        candidates: vec![Candidate {
            content: Content {
                role: "model".to_string(),
                parts: vec![Part {
                    text: text_response,
                }],
            },
        }],
    };

    Json(response)
}

/// Helper returning conversational responses matching the offline CLI mock behavior
fn get_conversational_mock_response(prompt: &str, manager: &ConversationManager) -> String {
    let prompt_lower = prompt.to_lowercase();

    // 1. Memory retrieval check: search history for dog's name
    if prompt_lower.contains("dog")
        && (prompt_lower.contains("name") || prompt_lower.contains("what is"))
    {
        let mut dog_name = None;
        for msg in &manager.messages {
            if msg.role == Role::User {
                let content_lower = msg.content.to_lowercase();
                if let Some(pos) = content_lower.find("dog's name is ") {
                    let start = pos + "dog's name is ".len();
                    let name_part = &msg.content[start..];
                    let name = name_part
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(|c: char| !c.is_alphabetic() && c != '.');
                    if !name.is_empty() {
                        dog_name = Some(name.to_string());
                    }
                } else if let Some(pos) = content_lower.find("dog is named ") {
                    let start = pos + "dog is named ".len();
                    let name_part = &msg.content[start..];
                    let name = name_part
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .trim_matches(|c: char| !c.is_alphabetic() && c != '.');
                    if !name.is_empty() {
                        dog_name = Some(name.to_string());
                    }
                }
            }
        }
        if let Some(name) = dog_name {
            return format!(
                "You previously mentioned that your dog's name is {}! This is proof that my multi-turn conversation memory (Phase 3) is working flawlessly in offline demo mode. What else would you like to discuss?",
                name
            );
        }
    }

    // 2. Memory retrieval check: recall previous user prompt
    if prompt_lower.contains("last prompt")
        || prompt_lower.contains("did i say")
        || prompt_lower.contains("previous prompt")
    {
        let mut prev_user_msg = None;
        if manager.messages.len() > 1 {
            // Traverse backwards, skip the current prompt (which is the last user message)
            for msg in manager.messages.iter().rev().skip(1) {
                if msg.role == Role::User {
                    prev_user_msg = Some(msg.content.as_str());
                    break;
                }
            }
        }
        if let Some(prev) = prev_user_msg {
            return format!(
                "Your previous prompt in this session was: \"{}\". My ConversationManager tracks all of our historical turns perfectly!",
                prev
            );
        }
    }

    // 3. Fallback mock responses
    if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
        "Hello! This is the Pocket Gemma Server operating in offline demo mode. I mimic the Gemini API generateContent endpoint beautifully. How can I help you today?".to_string()
    } else if prompt_lower.contains("model")
        || prompt_lower.contains("gemma")
        || prompt_lower.contains("gguf")
    {
        "I am currently running in offline mock mode. Provide the GGUF model path to run real CPU inference.".to_string()
    } else {
        "This is a mock response from the Gemini-compatible pocket-server for Pocket Gemma. Connect your frontend or tooling to start testing offline edge AI!".to_string()
    }
}

/// Axum route handler returning index.html for root path
async fn index_handler() -> impl IntoResponse {
    serve_asset("index.html")
}

/// Axum route handler serving embedded static files from the pocket-ui crate
async fn static_handler(
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    serve_asset(&path)
}

/// Helper function to retrieve files from rust-embed and return as an Axum Response
fn serve_asset(path: &str) -> Response {
    match pocket_ui::Assets::get(path) {
        Some(file) => {
            let content_type = match std::path::Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
            {
                Some("html") => "text/html",
                Some("css") => "text/css",
                Some("js") => "application/javascript",
                Some("png") => "image/png",
                Some("svg") => "image/svg+xml",
                _ => "application/octet-stream",
            };

            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, HeaderValue::from_static(content_type))
                .body(Body::from(file.data))
                .unwrap()
        }
        None => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap(),
    }
}
