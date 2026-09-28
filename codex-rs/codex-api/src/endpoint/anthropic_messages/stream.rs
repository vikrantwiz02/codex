//! Translation from Anthropic Messages streaming events into Codex [`ResponseEvent`]s.
//!
//! Text content blocks stream as an assistant message (`OutputItemAdded`, text deltas, then
//! `OutputItemDone`). Tool-use blocks are buffered until they finish and are then emitted as a
//! single function call, or as a custom tool call for freeform tools. Thinking blocks are
//! ignored because requests do not enable extended thinking.

use super::tools::CUSTOM_TOOL_INPUT_FIELD;
use super::tools::ToolKind;
use super::tools::ToolRegistry;
use crate::common::ResponseEvent;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: u64,
        content_block: Value,
    },
    ContentBlockDelta {
        index: u64,
        delta: Value,
    },
    ContentBlockStop {
        index: u64,
    },
    MessageDelta {
        #[serde(default)]
        usage: Option<Usage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct MessageStart {
    id: String,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Default, Deserialize)]
struct Usage {
    input_tokens: Option<i64>,
    cache_creation_input_tokens: Option<i64>,
    cache_read_input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamError {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: String,
}

impl StreamError {
    pub(crate) fn into_api_error(self) -> ApiError {
        match self.kind.as_str() {
            "overloaded_error" => ApiError::ServerOverloaded { retry_after: None },
            "rate_limit_error" => ApiError::RateLimitExceeded {
                message: self.message,
                retry_after: None,
            },
            "invalid_request_error" if self.message.contains("prompt is too long") => {
                ApiError::ContextWindowExceeded
            }
            "invalid_request_error" => ApiError::InvalidRequest {
                message: self.message,
            },
            _ => ApiError::Stream(format!("{}: {}", self.kind, self.message)),
        }
    }
}

enum Block {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
    Ignored,
}

/// Per-response translation state.
pub(crate) struct MessagesStreamState {
    tools: ToolRegistry,
    response_id: Option<String>,
    usage: Usage,
    blocks: HashMap<u64, Block>,
}

impl MessagesStreamState {
    pub(crate) fn new(tools: ToolRegistry) -> Self {
        Self {
            tools,
            response_id: None,
            usage: Usage::default(),
            blocks: HashMap::new(),
        }
    }

    /// Handles one decoded SSE payload.
    ///
    /// Returns the events to forward and whether the response is complete.
    pub(crate) fn handle(&mut self, data: &str) -> Result<(Vec<ResponseEvent>, bool), ApiError> {
        let event = match serde_json::from_str::<StreamEvent>(data) {
            Ok(event) => event,
            Err(err) => {
                debug!("failed to parse Anthropic SSE event: {err}");
                return Ok((Vec::new(), false));
            }
        };
        let mut events = Vec::new();
        match event {
            StreamEvent::MessageStart { message } => {
                if let Some(usage) = message.usage {
                    self.merge_usage(usage);
                }
                events.push(ResponseEvent::Created {
                    response_id: Some(message.id.clone()),
                });
                self.response_id = Some(message.id);
            }
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                let block = match content_block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        events.push(ResponseEvent::OutputItemAdded(assistant_message(
                            String::new(),
                        )));
                        let text = string_field(&content_block, "text");
                        if !text.is_empty() {
                            events.push(ResponseEvent::OutputTextDelta(text.clone()));
                        }
                        Block::Text(text)
                    }
                    Some("tool_use") => Block::ToolUse {
                        id: string_field(&content_block, "id"),
                        name: string_field(&content_block, "name"),
                        input_json: String::new(),
                    },
                    _ => Block::Ignored,
                };
                self.blocks.insert(index, block);
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                match (
                    self.blocks.get_mut(&index),
                    delta.get("type").and_then(Value::as_str),
                ) {
                    (Some(Block::Text(text)), Some("text_delta")) => {
                        let delta = string_field(&delta, "text");
                        text.push_str(&delta);
                        events.push(ResponseEvent::OutputTextDelta(delta));
                    }
                    (Some(Block::ToolUse { input_json, .. }), Some("input_json_delta")) => {
                        input_json.push_str(&string_field(&delta, "partial_json"));
                    }
                    _ => {}
                }
            }
            StreamEvent::ContentBlockStop { index } => {
                if let Some(item) = self
                    .blocks
                    .remove(&index)
                    .and_then(|block| self.finish_block(block))
                {
                    events.push(ResponseEvent::OutputItemDone(item));
                }
            }
            StreamEvent::MessageDelta { usage } => {
                if let Some(usage) = usage {
                    self.merge_usage(usage);
                }
            }
            StreamEvent::MessageStop => {
                events.push(ResponseEvent::Completed {
                    response_id: self.response_id.clone().unwrap_or_default(),
                    token_usage: Some(self.token_usage()),
                    usage_metadata: None,
                    end_turn: None,
                });
                return Ok((events, true));
            }
            StreamEvent::Error { error } => return Err(error.into_api_error()),
            StreamEvent::Ping | StreamEvent::Unknown => {}
        }
        Ok((events, false))
    }

    fn finish_block(&self, block: Block) -> Option<ResponseItem> {
        match block {
            Block::Text(text) => Some(assistant_message(text)),
            Block::ToolUse {
                id,
                name,
                input_json,
            } => {
                let input_json = if input_json.trim().is_empty() {
                    "{}".to_string()
                } else {
                    input_json
                };
                let Some(target) = self.tools.target(&name) else {
                    // Unknown tools are still surfaced so Codex can report the error to the model.
                    return Some(function_call(id, name, /*namespace*/ None, input_json));
                };
                match target.kind {
                    ToolKind::Function => Some(function_call(
                        id,
                        target.name.clone(),
                        target.namespace.clone(),
                        input_json,
                    )),
                    ToolKind::Custom => {
                        let input = serde_json::from_str::<Value>(&input_json)
                            .ok()
                            .and_then(|value| {
                                value
                                    .get(CUSTOM_TOOL_INPUT_FIELD)
                                    .and_then(Value::as_str)
                                    .map(str::to_string)
                            })
                            .unwrap_or(input_json);
                        Some(ResponseItem::CustomToolCall {
                            id: None,
                            status: None,
                            call_id: id,
                            name: target.name.clone(),
                            namespace: target.namespace.clone(),
                            input,
                            internal_chat_message_metadata_passthrough: None,
                        })
                    }
                }
            }
            Block::Ignored => None,
        }
    }

    fn merge_usage(&mut self, usage: Usage) {
        let Usage {
            input_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            output_tokens,
        } = usage;
        // Later events report cumulative counts, so newer values replace older ones.
        self.usage.input_tokens = input_tokens.or(self.usage.input_tokens);
        self.usage.cache_creation_input_tokens =
            cache_creation_input_tokens.or(self.usage.cache_creation_input_tokens);
        self.usage.cache_read_input_tokens =
            cache_read_input_tokens.or(self.usage.cache_read_input_tokens);
        self.usage.output_tokens = output_tokens.or(self.usage.output_tokens);
    }

    fn token_usage(&self) -> TokenUsage {
        let cache_write = self.usage.cache_creation_input_tokens.unwrap_or(0);
        let cache_read = self.usage.cache_read_input_tokens.unwrap_or(0);
        // Anthropic reports cache reads and writes separately from `input_tokens`, while Codex
        // counts them as part of the input.
        let input_tokens = self.usage.input_tokens.unwrap_or(0) + cache_write + cache_read;
        let output_tokens = self.usage.output_tokens.unwrap_or(0);
        TokenUsage {
            input_tokens,
            cached_input_tokens: cache_read,
            cache_write_input_tokens: cache_write,
            output_tokens,
            reasoning_output_tokens: 0,
            total_tokens: input_tokens + output_tokens,
            codex_rollout_budget_units: None,
        }
    }
}

fn string_field(value: &Value, field: &str) -> String {
    value
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn assistant_message(text: String) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText { text }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_call(
    call_id: String,
    name: String,
    namespace: Option<String>,
    arguments: String,
) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name,
        namespace,
        arguments,
        encrypted_function_args: None,
        call_id,
        internal_chat_message_metadata_passthrough: None,
    }
}

/// Reads an Anthropic Messages SSE body and forwards translated events until completion.
pub(crate) async fn process_messages_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    mut state: MessagesStreamState,
) {
    let mut stream = stream.eventsource();
    loop {
        let start = Instant::now();
        let response = tokio::select! {
            biased;
            _ = tx_event.closed() => return,
            response = timeout(idle_timeout, stream.next()) => response,
        };
        if let Some(telemetry) = telemetry.as_ref() {
            telemetry.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(err))) => {
                debug!("Anthropic SSE error: {err:#}");
                let error = match err {
                    eventsource_stream::EventStreamError::Transport(
                        error @ codex_client::TransportError::Policy(_),
                    ) => ApiError::Transport(error),
                    error => ApiError::Stream(error.to_string()),
                };
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Ok(None) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "stream closed before message_stop".to_string(),
                    )))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for SSE".to_string(),
                    )))
                    .await;
                return;
            }
        };
        trace!("Anthropic SSE event: {}", &sse.data);

        match state.handle(&sse.data) {
            Ok((events, completed)) => {
                for event in events {
                    if tx_event.send(Ok(event)).await.is_err() {
                        return;
                    }
                }
                if completed {
                    return;
                }
            }
            Err(error) => {
                let _ = tx_event.send(Err(error)).await;
                return;
            }
        }
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
