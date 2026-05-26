pub mod thinking;

pub use thinking::{ParserOutput, ParserState, ReasoningParser};

use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PocketError {
    #[error("I/O Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Configuration Error: {0}")]
    ConfigError(String),

    #[error("Engine Inference Error: {0}")]
    EngineError(String),

    #[error("Insufficient System RAM: {0}")]
    MemoryInsufficient(String),
}

pub type Result<T> = std::result::Result<T, PocketError>;

/// Represents the role of the speaker in a conversation turn
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A single turn of a multi-turn conversation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Manages rolling conversation context history and chat template formatting
#[derive(Debug, Clone)]
pub struct ConversationManager {
    pub system_prompt: String,
    pub messages: Vec<Message>,
}

impl ConversationManager {
    /// Instantiates a new ConversationManager with an optional system prompt override
    pub fn new(system_prompt: Option<String>) -> Self {
        Self {
            system_prompt: system_prompt
                .unwrap_or_else(|| "You are a helpful, highly capable AI assistant.".to_string()),
            messages: Vec::new(),
        }
    }

    /// Appends a new turn to the message log and hard-clamps history to the last 2 turns
    pub fn add_message(&mut self, role: Role, content: String) {
        self.messages.push(Message { role, content });

        let max_msg = if self.messages.last().map(|m| m.role) == Some(Role::Assistant) { 4 } else { 3 };
        if self.messages.len() > max_msg {
            let drain_count = self.messages.len() - max_msg;
            self.messages.drain(0..drain_count);
        }
    }

    /// Resets the conversation logs
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Stitches system prompt and history into the Gemma 2 prompt template:
    /// `<start_of_turn>user\n[System Instructions]\n\n[Prompt]<end_of_turn>\n<start_of_turn>model\n...`
    pub fn get_formatted_prompt(&self) -> String {
        let mut prompt = String::new();
        let mut system_prepended = false;

        // If there are no messages, format just the system prompt inside a user turn
        if self.messages.is_empty() {
            if !self.system_prompt.is_empty() {
                prompt.push_str("<start_of_turn>user\n");
                prompt.push_str(&self.system_prompt);
                prompt.push_str("<end_of_turn>\n");
            }
            prompt.push_str("<start_of_turn>model\n");
            return prompt;
        }

        let messages_len = self.messages.len();
        // Iterate through messages and format
        for (idx, message) in self.messages.iter().enumerate() {
            match message.role {
                Role::System => {
                    prompt.push_str("<start_of_turn>user\n");
                    prompt.push_str(&message.content);
                    prompt.push_str("<end_of_turn>\n");
                }
                Role::User => {
                    prompt.push_str("<start_of_turn>user\n");
                    if idx == messages_len - 1 {
                        // Prepend simplified strict factual prompt frame
                        prompt.push_str("[Provide a single, direct, factual answer without filler.]\n\n");
                    } else if !system_prepended && !self.system_prompt.is_empty() {
                        prompt.push_str(&self.system_prompt);
                        prompt.push_str("\n\n");
                        system_prepended = true;
                    }
                    prompt.push_str(&message.content);
                    prompt.push_str("<end_of_turn>\n");
                }
                Role::Assistant => {
                    prompt.push_str("<start_of_turn>model\n");
                    prompt.push_str(&message.content);
                    prompt.push_str("<end_of_turn>\n");
                }
            }
        }

        // Open assistant response turn
        prompt.push_str("<start_of_turn>model\n");

        prompt
    }
}

/// Pre-flight hardware guardrail (Req 3.1.2, 3.1.3):
/// Validates that model file size is less than 80% of total physical memory.
pub fn verify_ram_safety(model_path: &Path, force_ram: bool) -> Result<()> {
    if force_ram {
        return Ok(());
    }

    let file_metadata = std::fs::metadata(model_path).map_err(|e| {
        PocketError::ConfigError(format!(
            "Failed to read model file size from {:?}. Error: {}",
            model_path, e
        ))
    })?;
    let file_size = file_metadata.len();

    let mut sys = sysinfo::System::new_all();
    sys.refresh_memory();
    let total_ram = sys.total_memory();

    let safety_threshold = (total_ram as f64 * 0.8) as u64;

    if file_size > safety_threshold {
        return Err(PocketError::MemoryInsufficient(format!(
            "CRITICAL: Quantized model size ({:.2} GB) exceeds 80% of physical system RAM ({:.2} GB).\n\
            Loading this model may freeze or destabilize your operating system.\n\
            Use the '--force-ram' flag to bypass this hardware safety interlock.",
            file_size as f64 / 1_073_741_824.0,
            total_ram as f64 / 1_073_741_824.0
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma_conversation_manager_formatting() {
        let mut manager =
            ConversationManager::new(Some("You are a helpful assistant.".to_string()));

        // Assert initial formatting contains only system prompt inside user block
        let initial_prompt = manager.get_formatted_prompt();
        assert_eq!(
            initial_prompt,
            "<start_of_turn>user\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>model\n"
        );

        // Append user turn and verify strict factual prepending inside the final user turn
        manager.add_message(Role::User, "Hello!".to_string());
        let after_user = manager.get_formatted_prompt();
        assert_eq!(
            after_user,
            "<start_of_turn>user\n[Provide a single, direct, factual answer without filler.]\n\nHello!<end_of_turn>\n<start_of_turn>model\n"
        );

        // Append assistant turn and verify
        manager.add_message(Role::Assistant, "Hi there!".to_string());
        manager.add_message(Role::User, "Tell me a story.".to_string());
        let final_prompt = manager.get_formatted_prompt();
        assert_eq!(
            final_prompt,
            "<start_of_turn>user\nYou are a helpful assistant.\n\nHello!<end_of_turn>\n<start_of_turn>model\nHi there!<end_of_turn>\n<start_of_turn>user\n[Provide a single, direct, factual answer without filler.]\n\nTell me a story.<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn test_sliding_context_window() {
        let mut manager = ConversationManager::new(None);

        // Add 1st turn
        manager.add_message(Role::User, "Prompt 1".to_string());
        manager.add_message(Role::Assistant, "Response 1".to_string());
        assert_eq!(manager.messages.len(), 2);

        // Add 2nd turn
        manager.add_message(Role::User, "Prompt 2".to_string());
        assert_eq!(manager.messages.len(), 3);

        manager.add_message(Role::Assistant, "Response 2".to_string());
        assert_eq!(manager.messages.len(), 4);

        // Add 3rd turn (should trigger slide context clamping)
        manager.add_message(Role::User, "Prompt 3".to_string());
        assert_eq!(manager.messages.len(), 3);
        assert_eq!(manager.messages[0].content, "Prompt 2");
        assert_eq!(manager.messages[1].content, "Response 2");
        assert_eq!(manager.messages[2].content, "Prompt 3");
    }
}
