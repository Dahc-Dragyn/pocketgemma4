mod quantized_gemma;

use anyhow::Context;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use quantized_gemma::ModelWeights;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct LocalEngine {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Tokenizer>,
}

impl LocalEngine {
    /// Loads GGUF weights onto the CPU using memory-mapped I/O and parses the tokenizer from the given paths
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let file = File::open(model_path)
            .with_context(|| format!("Failed to open GGUF model file: {:?}", model_path))?;

        // Memory-map the model file for maximum sequential page lookup performance
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Apply sequential advice on Unix architectures to prefetch weights
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);

        let mut cursor = std::io::Cursor::new(&mmap[..]);

        // Ingest the GGUF file contents from the cursor
        let content = candle_core::quantized::gguf_file::Content::read(&mut cursor)
            .context("Failed to parse GGUF file metadata")?;

        // Extract GGUF native BPE vocabulary and merge rules
        let tokens_val = content.metadata.get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("Failed to find tokenizer.ggml.tokens in GGUF metadata"))?;
        let merges_val = content.metadata.get("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow::anyhow!("Failed to find tokenizer.ggml.merges in GGUF metadata"))?;

        let tokens_array = match tokens_val {
            candle_core::quantized::gguf_file::Value::Array(arr) => arr,
            _ => anyhow::bail!("tokenizer.ggml.tokens is not an array"),
        };
        let merges_array = match merges_val {
            candle_core::quantized::gguf_file::Value::Array(arr) => arr,
            _ => anyhow::bail!("tokenizer.ggml.merges is not an array"),
        };

        let mut vocab = std::collections::HashMap::new();
        for (idx, token_val) in tokens_array.iter().enumerate() {
            if let candle_core::quantized::gguf_file::Value::String(s) = token_val {
                vocab.insert(s.clone(), idx);
            }
        }

        let mut merges_list = Vec::new();
        for merge_val in merges_array {
            if let candle_core::quantized::gguf_file::Value::String(s) = merge_val {
                merges_list.push(s.clone());
            }
        }

        let metaspace_char = "\u{2581}";

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "special": true,
                    "content": "<pad>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 1,
                    "special": true,
                    "content": "<eos>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 2,
                    "special": true,
                    "content": "<bos>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 3,
                    "special": true,
                    "content": "<unk>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 4,
                    "special": true,
                    "content": "<mask>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 105,
                    "special": true,
                    "content": "<|turn>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                },
                {
                    "id": 106,
                    "special": true,
                    "content": "<turn|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false
                }
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {
                        "type": "Prepend",
                        "prepend": " "
                    },
                    {
                        "type": "Replace",
                        "pattern": {
                            "String": " "
                        },
                        "content": metaspace_char
                    }
                ]
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": metaspace_char,
                "add_prefix_space": true
            },
            "post_processor": null,
            "decoder": {
                "type": "Metaspace",
                "replacement": metaspace_char,
                "add_prefix_space": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "vocab": vocab,
                "merges": merges_list
            }
        });

        let json_str = serde_json::to_string(&tokenizer_json)?;
        let tokenizer = Tokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| anyhow::anyhow!("Tokenizer build error: {}", e))?;

        println!("=== Gemma 4 Native Control Alignment Verification ===");
        let pad_id = vocab.get("<pad>").map(|&id| id as i32).unwrap_or(-1);
        let eos_id = vocab.get("<eos>").map(|&id| id as i32).unwrap_or(-1);
        let bos_id = vocab.get("<bos>").map(|&id| id as i32).unwrap_or(-1);
        let turn_start_id = vocab.get("<|turn>").map(|&id| id as i32).unwrap_or(-1);
        let turn_end_id = vocab.get("<turn|>").map(|&id| id as i32).unwrap_or(-1);
        
        println!("<pad>: {}", pad_id);
        println!("<eos>: {}", eos_id);
        println!("<bos>: {}", bos_id);
        println!("<|turn>: {}", turn_start_id);
        println!("<turn|>: {}", turn_end_id);
        println!("===================================================");

        // Instantiate ModelWeights on CPU (dynamically detects Gemma version from GGUF headers)
        let model_weights = ModelWeights::from_gguf(content, &mut cursor, &Device::Cpu)
            .context("Failed to load quantized Gemma weights from GGUF content")?;

        Ok(Self {
            model: Arc::new(Mutex::new(model_weights)),
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Asynchronously streams generated tokens across a tokio mpsc channel
    pub async fn generate_stream(
        &self,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> anyhow::Result<()> {
        let model_arc = Arc::clone(&self.model);
        let tokenizer_arc = Arc::clone(&self.tokenizer);
        let prompt_str = prompt.to_string();

        // Offload the heavy blocking tensor computations to spawn_blocking
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let mut model_lock = model_arc
                .lock()
                .map_err(|e| anyhow::anyhow!("Model mutex poisoned: {}", e))?;

            // Setup LogitsProcessor with a default seed and conservative parameters (temperature = 0.3, top_p = 0.85)
            let mut logits_processor = LogitsProcessor::new(299792458, Some(0.3), Some(0.85));

            // Format the input prompt into the Gemma 4 Instruct template
            let formatted_prompt = format!("<|turn|>user\n{}<turn|>\n<|turn|>model\n", prompt_str);
            
            // Tokenize the formatted prompt without adding special tokens automatically
            let encoded = tokenizer_arc
                .encode(formatted_prompt, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            
            // Explicitly enforce <bos> at index 0
            let mut input_ids = vec![2u32];
            input_ids.extend(encoded.get_ids());

            if input_ids.len() <= 1 {
                anyhow::bail!("Input prompt generated zero tokens");
            }

            // Gemma 4 EOS token ID is 1 (<eos>)
            let eos_token_id = 1;

            let mut tokens_list = input_ids.to_vec();
            let mut pos = 0;
            let mut consecutive_whitespace = 0;
            let max_tokens = 512;

            for i in 0..max_tokens {
                let input = if i == 0 {
                    // Prefill phase: feed the entire prompt
                    Tensor::new(tokens_list.as_slice(), &Device::Cpu)
                        .context("Failed to construct prefill tensor")?
                        .unsqueeze(0)
                        .context("Failed to unsqueeze prefill tensor")?
                } else {
                    // Autoregressive phase: feed only the single last generated token
                    let last_token = *tokens_list
                        .last()
                        .ok_or_else(|| anyhow::anyhow!("Token sequence is empty"))?;
                    Tensor::new(&[last_token], &Device::Cpu)
                        .context("Failed to construct autoregressive tensor")?
                        .unsqueeze(0)
                        .context("Failed to unsqueeze autoregressive tensor")?
                };

                // Compute logits using forward pass on CPU GGUF model
                let logits = model_lock
                    .forward(&input, pos)
                    .context("Model forward pass failed")?;

                // Advance prompt sequence context index
                pos += input.dim(1).context("Failed to read sequence dimension")?;

                // Squeeze to extract logits for the last token.
                // In quantized_gemma, forward already slices internally so the shape is (1, vocab_size)
                let last_logits = logits.squeeze(0).context("Failed to squeeze logits")?;

                // Retrieve raw logits as a CPU-accessible 1D vector to apply penalty
                let mut logits_vec = last_logits.to_vec1::<f32>().context("Failed to read logits into 1D vec")?;

                // Apply repetition penalty over the last 512 tokens (repeat_last_n = 512, repeat_penalty = 1.25)
                // Exclude the initial prompt tokens to prevent extreme probability degradation of prompt context
                let start_idx = input_ids.len().max(tokens_list.len().saturating_sub(512));
                for &prev_token in &tokens_list[start_idx..] {
                    // Strict stop token bypass: do not penalize standard/native EOS, BOS, and turn signals
                    if prev_token == 1 || prev_token == 2 || prev_token == 105 || prev_token == 106 {
                        continue;
                    }
                    let prev_token_usize = prev_token as usize;
                    if prev_token_usize < logits_vec.len() {
                        let logit = logits_vec[prev_token_usize];
                        if logit >= 0.0 {
                            logits_vec[prev_token_usize] = logit / 1.25;
                        } else {
                            logits_vec[prev_token_usize] = logit * 1.25;
                        }
                    }
                }

                // Construct a new penalized logits tensor
                let penalized_logits = Tensor::new(logits_vec.as_slice(), &Device::Cpu)
                    .context("Failed to construct penalized logits tensor")?;

                // Sample next token ID
                let next_token = logits_processor
                    .sample(&penalized_logits)
                    .context("Logits sampling failed")?;

                println!("Selected Token ID: {}", next_token);

                // Check for end of sequence signal
                if next_token == eos_token_id || next_token == 1 || next_token == 105 || next_token == 106 {
                    break;
                }

                // Append token ID to context list
                tokens_list.push(next_token);

                // Decode token ID to String slice
                if let Ok(token_str) = tokenizer_arc.decode(&[next_token], true) {
                    // Text-fallback safety check for stop tokens
                    if token_str.contains("<end_of_turn>") || token_str.contains("<eos>") {
                        break;
                    }

                    // Trim & Flush safety check to prevent endless whitespace loops
                    if token_str.trim().is_empty() {
                        consecutive_whitespace += 1;
                        if consecutive_whitespace >= 10 { // safety threshold of 10 consecutive empty/whitespace pieces
                            break;
                        }
                    } else {
                        consecutive_whitespace = 0;
                    }

                    // Check if channel is still open before sending
                    if tx.blocking_send(token_str).is_err() {
                        // Receiver dropped, stop generation gracefully
                        break;
                    }
                }
            }

            drop(tx);
            Ok(())
        })
        .await
        .context("Tokio spawn_blocking task join failed")?
    }
}
