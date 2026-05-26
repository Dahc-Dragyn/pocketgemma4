use anyhow::Context;
use clap::Parser;
use pocket_core::{ConversationManager, Role};
use pocket_engine::LocalEngine;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;

/// Pocket Gemma - A hyper-optimized, single-binary local Gemma 2 LLM runner
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the quantized GGUF model file
    #[arg(short, long)]
    model: Option<String>,

    /// Path to the tokenizer JSON file
    #[arg(short, long)]
    tokenizer: Option<String>,

    /// Override pre-flight physical RAM safety limits
    #[arg(long, default_value_t = false)]
    force_ram: bool,

    /// Launch Gemini-compatible REST API server on port 8080
    #[arg(short, long, default_value_t = false)]
    serve: bool,

    /// Launch Embedded Chat HUD in your default web browser
    #[arg(short, long, default_value_t = false)]
    ui: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    println!("==================================================");
    println!("          Pocket Gemma - Core Controller          ");
    println!("==================================================");

    // Initialize the engine if model path is supplied
    let engine = if let Some(model_path) = args.model.as_ref() {
        println!("Loading model from: {}", model_path);
        if let Some(tok_path) = args.tokenizer.as_ref() {
            println!("Notice: Bypassing external tokenizer '{}'. Dynamic native GGUF vocabulary extraction is active!", tok_path);
        }

        let path = Path::new(model_path);

        // Execute physical RAM safety guardrail
        print!("Pre-flight hardware check...");
        io::stdout().flush().context("Failed to flush stdout")?;

        match pocket_core::verify_ram_safety(path, args.force_ram) {
            Ok(_) => {
                println!(" PASSED");
            }
            Err(err) => {
                println!(" FAILED");
                return Err(anyhow::anyhow!(err))
                    .context("Pre-flight system RAM guardrail safety block");
            }
        }

        print!("Initializing Local Inference Engine (Gemma)...");
        io::stdout().flush().context("Failed to flush stdout")?;

        let start = std::time::Instant::now();
        let loaded_engine = LocalEngine::load(path)
            .context("Failed to load local model weights and extract native vocabulary")?;

        println!(" Done! (Loaded in {:.2?})\n", start.elapsed());
        Some(loaded_engine)
    } else {
        println!("Running in offline demo mode. No GGUF model specified.");
        println!("To load a model, run with: --model <GGUF_PATH>");
        println!("Type 'exit' or 'quit' to close. Press Ctrl-C or Ctrl-D to abort.\n");
        None
    };

    // Calculate if UI launch mode is requested or is zero-argument execution
    let run_ui = args.ui || std::env::args().len() <= 1;

    if run_ui {
        println!("==================================================");
        println!("         Pocket Gemma - Embedded Chat HUD         ");
        println!("==================================================");
        println!("Launching Desktop Appliance Web UI Portal...");

        let url = "http://127.0.0.1:8080";
        if let Err(err) = webbrowser::open(url) {
            eprintln!("Warning: Failed to launch system default browser: {}", err);
            println!("Please navigate directly to {} in your browser.", url);
        } else {
            println!("Default system browser launched successfully at {}", url);
        }

        pocket_server::start_server(engine, 8080).await?;
        return Ok(());
    }

    // Bypasses the terminal REPL loop if launch server mode is requested
    if args.serve {
        println!("==================================================");
        println!("         Pocket Gemma - API Server Mode           ");
        println!("==================================================");
        pocket_server::start_server(engine, 8080).await?;
        return Ok(());
    }

    // Initialize rustyline editor
    let mut rl = DefaultEditor::new().context("Failed to initialize terminal REPL editor")?;

    // Instantiate ConversationManager outside the rustyline loop to maintain state
    let mut manager = ConversationManager::new(None);

    loop {
        // Prompt user for input
        let readline = rl.readline("PocketGemma > ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                // Ignore empty inputs
                if trimmed.is_empty() {
                    continue;
                }

                // Handle explicit exit commands
                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    println!("Exiting pocket-cli gracefully. Goodbye!");
                    break;
                }

                // Add prompt history to rustyline CLI list
                if let Err(err) = rl.add_history_entry(trimmed) {
                    eprintln!("Warning: Failed to add history entry: {}", err);
                }

                // Record user turn in conversational memory
                manager.add_message(Role::User, trimmed.to_string());

                if let Some(ref local_engine) = engine {
                    // --- REAL MODEL STREAMING INFERENCE ---
                    println!("\nPocketGemma:");

                    // Instantiate tokio mpsc channel for streaming
                    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(100);

                    // Spawn the stream generator task, passing the formatted conversational history
                    let engine_clone = local_engine.clone();
                    let prompt_str = manager.get_formatted_prompt();

                    let gen_task = tokio::spawn(async move {
                        if let Err(err) = engine_clone.generate_stream(&prompt_str, tx).await {
                            eprintln!("\nEngine Inference Error: {:?}", err);
                        }
                    });

                    // Build assistant response string dynamically from received tokens
                    let mut full_response = String::new();
                    let mut parser = pocket_core::ReasoningParser::new();
                    let mut current_state = pocket_core::ParserState::Factual;

                    // Consume tokens as they arrive via channel
                    while let Some(token) = rx.recv().await {
                        full_response.push_str(&token);
                        let outputs = parser.process(&token);
                        for out in outputs {
                            match out {
                                pocket_core::ParserOutput::Thinking(text) => {
                                    if current_state == pocket_core::ParserState::Factual {
                                        print!("\x1b[33m\n[Thinking Process]\n\x1b[0m");
                                        current_state = pocket_core::ParserState::Thinking;
                                    }
                                    print!("\x1b[90m{}\x1b[0m", text);
                                    io::stdout().flush().context("Failed to flush stdout token")?;
                                }
                                pocket_core::ParserOutput::Factual(text) => {
                                    if current_state == pocket_core::ParserState::Thinking {
                                        print!("\x1b[36m\n\n[Factual Response]\n\x1b[0m");
                                        current_state = pocket_core::ParserState::Factual;
                                    }
                                    print!("{}", text);
                                    io::stdout().flush().context("Failed to flush stdout token")?;
                                }
                            }
                        }
                    }

                    if let Some(flushed) = parser.flush() {
                        match flushed {
                            pocket_core::ParserOutput::Thinking(text) => {
                                if current_state == pocket_core::ParserState::Factual {
                                    print!("\x1b[33m\n[Thinking Process]\n\x1b[0m");
                                }
                                print!("\x1b[90m{}\x1b[0m", text);
                                io::stdout().flush().context("Failed to flush stdout token")?;
                            }
                            pocket_core::ParserOutput::Factual(text) => {
                                if current_state == pocket_core::ParserState::Thinking {
                                    print!("\x1b[36m\n\n[Factual Response]\n\x1b[0m");
                                }
                                print!("{}", text);
                                io::stdout().flush().context("Failed to flush stdout token")?;
                            }
                        }
                    }
                    println!("\n");

                    // Save the complete assistant response back to conversation history
                    manager.add_message(Role::Assistant, full_response);

                    // Await the inference generation task completion
                    if let Err(err) = gen_task.await {
                        eprintln!("Generation task panicked: {:?}", err);
                    }
                } else {
                    // --- OFFLINE DEMO MODE FALLBACK ---
                    print!("PocketGemma is thinking...");
                    io::stdout().flush().context("Failed to flush stdout")?;

                    // Asynchronous simulated delay of 1.0s (made snappier)
                    sleep(Duration::from_millis(1000)).await;

                    // Clear the "thinking..." line and print response header
                    print!("\r\x1b[K");
                    io::stdout().flush().context("Failed to flush stdout")?;

                    println!("\nPocketGemma:");

                    // Fetch rich context-aware response based on entire conversation state
                    let mock_response = get_conversational_mock_response(trimmed, &manager);

                    // Stream mock response with typewriter delay
                    for token in mock_response.split_whitespace() {
                        print!("{} ", token);
                        io::stdout()
                            .flush()
                            .context("Failed to flush token to stdout")?;
                        sleep(Duration::from_millis(40)).await;
                    }
                    println!("\n");

                    // Save the mock response turn to memory
                    manager.add_message(Role::Assistant, mock_response);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\n[Ctrl-C detected] Exiting pocket-cli gracefully. Goodbye!");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("\n[Ctrl-D detected] Exiting pocket-cli gracefully. Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error reading input line: {:?}", err);
                return Err(err).context("Fatal error in terminal REPL reader");
            }
        }
    }

    Ok(())
}

/// Returns a rich, context-aware mock response that dynamically reflects conversational memory.
fn get_conversational_mock_response(prompt: &str, manager: &ConversationManager) -> String {
    let prompt_lower = prompt.to_lowercase();

    // 1. Dynamic Check: Memory Retrieval for Pet/Dog Names
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

    // 2. Dynamic Check: Last Prompt Recall
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

    // 3. Fallback Standard Responses
    if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
        "Hello! I am Pocket Gemma, now running with Phase 3 active memory and dynamic ChatML templating. How can I help you today?".to_string()
    } else if prompt_lower.contains("model")
        || prompt_lower.contains("gemma")
        || prompt_lower.contains("gguf")
    {
        "With Phase 3 complete, we can format full conversational histories into quantized GGUF CPU models like Gemma 2 using unified templates and stream the answers in real-time.".to_string()
    } else if prompt_lower.contains("help") {
        "You are in the interactive CLI REPL. I now maintain full multi-turn conversational history. You can mention facts (like 'my dog's name is Winnie') and ask me about them later to test my memory!".to_string()
    } else if prompt_lower.contains("ram")
        || prompt_lower.contains("memory")
        || prompt_lower.contains("size")
    {
        "Pocket Gemma monitors system memory using sysinfo. In this Phase 3 REPL, the ConversationManager stores all turns to construct cohesive prompt blocks.".to_string()
    } else {
        "This is a mock conversational response. Once a real GGUF model is loaded with -m, the entire multi-turn conversation history will be compiled and executed using candle-core.".to_string()
    }
}
