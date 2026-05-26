//! Stateful streaming Reasoning Tag Parser to identify and segregate `<|think|>` blocks.
//! Supports partial token splits across stream chunks via a sliding prefix/suffix matching window.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserState {
    Factual,
    Thinking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserOutput {
    Factual(String),
    Thinking(String),
}

pub struct ReasoningParser {
    state: ParserState,
    pending_buffer: String,
}

impl ReasoningParser {
    /// Creates a new `ReasoningParser` initialized in the Factual state.
    pub fn new() -> Self {
        Self {
            state: ParserState::Factual,
            pending_buffer: String::new(),
        }
    }

    /// Retrieves the current parser state.
    pub fn state(&self) -> ParserState {
        self.state
    }

    /// Processes an incoming text chunk and yields any finalized factual/thinking outputs.
    /// Safely buffers partial tag matches so they are not leaked prematurely.
    pub fn process(&mut self, chunk: &str) -> Vec<ParserOutput> {
        self.pending_buffer.push_str(chunk);
        let mut outputs = Vec::new();

        loop {
            match self.state {
                ParserState::Factual => {
                    // Check for a complete <|think|> tag
                    let tag = "<|think|>";
                    if let Some(idx) = self.pending_buffer.find(tag) {
                        let before = self.pending_buffer[..idx].to_string();
                        if !before.is_empty() {
                            outputs.push(ParserOutput::Factual(before));
                        }
                        self.state = ParserState::Thinking;
                        self.pending_buffer = self.pending_buffer[idx + tag.len()..].to_string();
                    } else {
                        // Suffix match check for "<|think|>" prefix to prevent leaks
                        let mut longest_partial_len = 0;
                        for i in 1..tag.len() {
                            let prefix = &tag[..i];
                            if self.pending_buffer.ends_with(prefix) {
                                longest_partial_len = i;
                            }
                        }

                        let emit_len = self.pending_buffer.len() - longest_partial_len;
                        if emit_len > 0 {
                            let to_emit = self.pending_buffer[..emit_len].to_string();
                            outputs.push(ParserOutput::Factual(to_emit));
                            self.pending_buffer = self.pending_buffer[emit_len..].to_string();
                        }
                        break;
                    }
                }
                ParserState::Thinking => {
                    // Check for a complete </|think|> tag
                    let tag = "</|think|>";
                    if let Some(idx) = self.pending_buffer.find(tag) {
                        let before = self.pending_buffer[..idx].to_string();
                        if !before.is_empty() {
                            outputs.push(ParserOutput::Thinking(before));
                        }
                        self.state = ParserState::Factual;
                        self.pending_buffer = self.pending_buffer[idx + tag.len()..].to_string();
                    } else {
                        // Suffix match check for "</|think|>" prefix to prevent leaks
                        let mut longest_partial_len = 0;
                        for i in 1..tag.len() {
                            let prefix = &tag[..i];
                            if self.pending_buffer.ends_with(prefix) {
                                longest_partial_len = i;
                            }
                        }

                        let emit_len = self.pending_buffer.len() - longest_partial_len;
                        if emit_len > 0 {
                            let to_emit = self.pending_buffer[..emit_len].to_string();
                            outputs.push(ParserOutput::Thinking(to_emit));
                            self.pending_buffer = self.pending_buffer[emit_len..].to_string();
                        }
                        break;
                    }
                }
            }
        }
        outputs
    }

    /// Flushes any remaining buffered text at the end of token generation.
    pub fn flush(&mut self) -> Option<ParserOutput> {
        if self.pending_buffer.is_empty() {
            None
        } else {
            let content = std::mem::take(&mut self.pending_buffer);
            match self.state {
                ParserState::Factual => Some(ParserOutput::Factual(content)),
                ParserState::Thinking => Some(ParserOutput::Thinking(content)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_tag_isolation() {
        let mut parser = ReasoningParser::new();

        // 1. Send clean factual prefix
        let out1 = parser.process("Hello world! ");
        assert_eq!(out1, vec![ParserOutput::Factual("Hello world! ".to_string())]);

        // 2. Start thinking block
        let out2 = parser.process("<|think|>I am thinking hard.");
        assert_eq!(out2, vec![ParserOutput::Thinking("I am thinking hard.".to_string())]);

        // 3. End thinking block
        let out3 = parser.process("</|think|> Here is the fact.");
        assert_eq!(out3, vec![ParserOutput::Factual(" Here is the fact.".to_string())]);

        assert_eq!(parser.state(), ParserState::Factual);
    }

    #[test]
    fn test_chunked_tag_boundaries() {
        let mut parser = ReasoningParser::new();

        // Send a factual statement and start of thinking tag split across chunks
        let out1 = parser.process("The answer is <|");
        assert_eq!(out1, vec![ParserOutput::Factual("The answer is ".to_string())]);

        let out2 = parser.process("th");
        assert_eq!(out2, vec![]); // withheld prefix match

        let out3 = parser.process("ink|>2 + 2 is ");
        assert_eq!(out3, vec![ParserOutput::Thinking("2 + 2 is ".to_string())]);

        // Split end tag
        let out4 = parser.process("4. </|th");
        assert_eq!(out4, vec![ParserOutput::Thinking("4. ".to_string())]);

        let out5 = parser.process("ink|>Done!");
        assert_eq!(out5, vec![ParserOutput::Factual("Done!".to_string())]);

        assert_eq!(parser.flush(), None);
    }

    #[test]
    fn test_flush_withheld_text() {
        let mut parser = ReasoningParser::new();

        // Suffix matches prefix of tag but never completes
        let out = parser.process("This is <|th");
        assert_eq!(out, vec![ParserOutput::Factual("This is ".to_string())]);

        // Flush should yield the withheld text
        let flushed = parser.flush();
        assert_eq!(flushed, Some(ParserOutput::Factual("<|th".to_string())));
    }
}
