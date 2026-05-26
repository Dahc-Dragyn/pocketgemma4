# 💎 PocketGemma4--- Is not working atm! Pocketgemma https://github.com/Dahc-Dragyn/pocketgemma.git  is running =) if the end of the world has happend and you need a LLM... its slow use the best computer you can get ahold of. The model is currently "thinking" and outputting random multilingual noise because the Q4_K_M quantization headers are being destroyed by manual memory-slicing and pagefile thrashing, forcing the engine to dequantize corrupted weight data.

### *Hyper-Optimized, Self-Contained Edge AI & Chat HUD for Gemma 4 *google_gemma-4-E2B-it-Q4_K_M.gguf

**PocketGemma4** is a high-performance, single-binary local LLM runner optimized exclusively for **Gemma 2 2B IT**. Built entirely in Rust, it bundles a customized quantized tensor parser, an active conversational context manager, a physical RAM protection guardrail, a Gemini-compatible REST server, and a beautiful embedded web-based HUD into one self-contained, lightning-fast execution file.

---

## 🚀 Key Features

*   **Custom quantized_gemma2 Engine**: Leverages custom RMSNorm layer mapping designed specifically for Gemma 2 (handling models without Query/Key Normalization cleanly).
*   **RoPE Cache Pre-computation Cap**: Safely caps memory allocations for Rotary Position Embeddings (RoPE) at 8,192 tokens, reducing cache footprint from **1.6 GB to ~104 MB** (a **16x memory footprint reduction**).
*   **Real-Time Terminal Telemetry**: Streams token outputs to the command line instantly as they generate.
*   **Unified stop token enforcement**: Hardened to immediately break on Gemma 2 turn-boundaries (`<eos>` / ID `1` and `<end_of_turn>` / ID `107`) to avoid running redundant CPU autoregressive computations.
*   **Physical RAM Interlock**: Uses `sysinfo` to verify the model size does not exceed 80% of system RAM, protecting against OS thrashing.
*   **Air-Gapped Embedded UI**: Launches an in-memory chat HUD directly in the system browser using raw HTML/CSS/JS embedded into the executable with `rust-embed`.
*   **Gemini-Compatible API Server**: Emulates the Google Gemini API `generateContent` REST interface on port `8080` for easy integration with standard tools.

---

## 📂 H: Drive Storage Layout

The correct model weights and tokenizer dictionary files have been consolidated onto the `H:` drive:

```
H:\pocketgemma\
├── gemma-2-2b-it-Q4_K_M.gguf  <-- Gemma 2 Quantized Weights (~1.70 GB)
└── tokenizer.json             <-- Correct Gemma 2 Tokenizer Dictionary (~17.5 MB)
```

---

## 🛠️ Quickstart Commands (PowerShell)

### 1. Launch Interactive CLI REPL
Run the command-line chat session with real-time token telemetry:
```powershell
.\target\release\pocketgemma.exe --model "H:\pocketgemma\gemma-2-2b-it-Q4_K_M.gguf" --tokenizer "H:\pocketgemma\tokenizer.json"
```

### 2. Launch Embedded Chat HUD (Web UI)
Launch the embedded air-gapped web frontend (automatically opens `http://127.0.0.1:8080`):
```powershell
.\target\release\pocketgemma.exe --ui --model "H:\pocketgemma\gemma-2-2b-it-Q4_K_M.gguf" --tokenizer "H:\pocketgemma\tokenizer.json"
```

### 3. Launch API Server
Start the Gemini-compatible REST server (port `8080`):
```powershell
.\target\release\pocketgemma.exe --serve --model "H:\pocketgemma\gemma-2-2b-it-Q4_K_M.gguf" --tokenizer "H:\pocketgemma\tokenizer.json"
```

#### Test API Server with `curl`:
```powershell
curl -X POST http://127.0.0.1:8080/v1beta/models/pocketgemma:generateContent `
  -H "Content-Type: application/json" `
  -d '{ "contents": [{"role": "user", "parts": [{"text": "Hello, are you running entirely offline?"}]}] }'
```

---

## ⚙️ Architecture & Crate Taxonomy

*   [**`pocket-core`**](file:///c:/Antigravity%20projects/Rust/pocketgemma/pocket-core) - ConversationManager, Role definition, template builder, and the physical RAM verification guardrail.
*   [**`pocket-engine`**](file:///c:/Antigravity%20projects/Rust/pocketgemma/pocket-engine) - Heavy blocking model loading, quantized gemma2 parser (`quantized_gemma.rs`), real-time console streaming, and stop token mappings.
*   [**`pocket-server`**](file:///c:/Antigravity%20projects/Rust/pocketgemma/pocket-server) - Axum web server routing, Gemini REST payload parser, and asset delivery middleware.
*   [**`pocket-ui`**](file:///c:/Antigravity%20projects/Rust/pocketgemma/pocket-ui) - Glassmorphic, dark-themed responsive chat frontend compiled directly into the binary.
*   [**`pocket-cli`**](file:///c:/Antigravity%20projects/Rust/pocketgemma/pocket-cli) - Core CLI orchestrator parsing arguments, initiating rustyline REPL, and serving UI/REST commands.
