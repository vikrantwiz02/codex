//! Translation from Codex's Responses-shaped request into an Anthropic Messages request.
//!
//! Codex builds every model request as a [`ResponsesApiRequest`]. Providers that speak the
//! Anthropic Messages API receive the same request, translated here:
//!
//! - `instructions` becomes the top-level `system` prompt.
//! - Conversation items become alternating `user` / `assistant` messages. Developer and system
//!   messages are sent as user text because the Messages API only has one system prompt.
//! - Function and freeform (custom) tools become Anthropic tools. Freeform tools take a single
//!   `input` string argument. Namespaced tools are flattened into one wire name, and the
//!   [`ToolRegistry`] maps the wire name back when the model calls the tool.
//! - Reasoning, hosted tools (web search, tool search, image generation), and compaction items
//!   have no Messages API equivalent and are omitted.

use super::tools::CUSTOM_TOOL_INPUT_FIELD;
use super::tools::Tool;
use super::tools::ToolRegistry;
use super::tools::translate_tools;
use crate::common::ResponsesApiRequest;
use crate::error::ApiError;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ImageReference;
use codex_protocol::models::ResponseItem;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;

/// Output token budget sent with every request; the Messages API requires one.
const DEFAULT_MAX_TOKENS: u32 = 32_000;
const MISSING_TOOL_OUTPUT: &str = "Tool call was interrupted before producing output.";

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct MessagesRequest {
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) system: Vec<ContentBlock>,
    pub(crate) messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) stream: bool,
}

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    User,
    Assistant,
}

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct ContentBlock {
    #[serde(flatten)]
    pub(crate) kind: BlockKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cache_control: Option<CacheControl>,
}

impl From<BlockKind> for ContentBlock {
    fn from(kind: BlockKind) -> Self {
        Self {
            kind,
            cache_control: None,
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum BlockKind {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum CacheControl {
    Ephemeral,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ToolChoice {
    Auto { disable_parallel_tool_use: bool },
}

/// Translates a Responses-shaped request into an Anthropic Messages request.
pub(crate) fn build_messages_request(
    request: &ResponsesApiRequest,
) -> Result<(MessagesRequest, ToolRegistry), ApiError> {
    let (tools, registry) = match &request.tools {
        Some(tools) => translate_tools(tools.as_raw_value().get())?,
        None => (Vec::new(), ToolRegistry::default()),
    };

    let mut builder = MessagesBuilder::default();
    for item in &request.input {
        builder.push_item(item, &registry);
    }
    let mut messages = builder.finish();
    if let Some(last_block) = messages
        .last_mut()
        .and_then(|message| message.content.last_mut())
    {
        last_block.cache_control = Some(CacheControl::Ephemeral);
    }

    let system = if request.instructions.trim().is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock {
            kind: BlockKind::Text {
                text: request.instructions.clone(),
            },
            cache_control: Some(CacheControl::Ephemeral),
        }]
    };
    let tool_choice = (!tools.is_empty()).then_some(ToolChoice::Auto {
        disable_parallel_tool_use: !request.parallel_tool_calls,
    });

    Ok((
        MessagesRequest {
            model: request.model.clone(),
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages,
            tools,
            tool_choice,
            stream: true,
        },
        registry,
    ))
}

/// Accumulates Messages API turns while keeping tool_use / tool_result pairing valid.
#[derive(Default)]
struct MessagesBuilder {
    messages: Vec<Message>,
    /// Tool calls whose results have not been sent yet, in call order.
    pending_tool_uses: Vec<String>,
    seen_tool_uses: HashSet<String>,
}

impl MessagesBuilder {
    fn push_item(&mut self, item: &ResponseItem, registry: &ToolRegistry) {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let role = if role == "assistant" {
                    Role::Assistant
                } else {
                    Role::User
                };
                for content_item in content {
                    if let Some(block) = content_item_block(content_item) {
                        self.push(role, block);
                    }
                }
            }
            ResponseItem::AgentMessage {
                author, content, ..
            } => {
                for part in content {
                    match part {
                        AgentMessageInputContent::InputText { text } => {
                            self.push_text(Role::User, format!("Message from {author}:\n{text}"));
                        }
                        AgentMessageInputContent::EncryptedContent { .. } => {}
                    }
                }
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                let input = serde_json::from_str::<Value>(arguments)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({ "arguments": arguments }));
                self.push_tool_use(
                    call_id,
                    registry.wire_name(namespace.as_deref(), name),
                    input,
                );
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                namespace,
                input,
                ..
            } => {
                self.push_tool_use(
                    call_id,
                    registry.wire_name(namespace.as_deref(), name),
                    json!({ CUSTOM_TOOL_INPUT_FIELD: input }),
                );
            }
            ResponseItem::FunctionCallOutput {
                call_id: Some(call_id),
                output,
                ..
            }
            | ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => self.push_tool_result(call_id, output),
            ResponseItem::FunctionCallOutput { call_id: None, .. }
            | ResponseItem::AdditionalTools { .. }
            | ResponseItem::Reasoning { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::ConfigurationUpdate { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }

    fn push_text(&mut self, role: Role, text: String) {
        if !text.trim().is_empty() {
            self.push(role, BlockKind::Text { text }.into());
        }
    }

    fn push_tool_use(&mut self, call_id: &str, name: String, input: Value) {
        if !self.seen_tool_uses.insert(call_id.to_string()) {
            return;
        }
        self.push(
            Role::Assistant,
            BlockKind::ToolUse {
                id: call_id.to_string(),
                name,
                input,
            }
            .into(),
        );
        self.pending_tool_uses.push(call_id.to_string());
    }

    fn push_tool_result(&mut self, call_id: &str, output: &FunctionCallOutputPayload) {
        let Some(position) = self.pending_tool_uses.iter().position(|id| id == call_id) else {
            // The matching call is missing from history (or was already answered), so the
            // Messages API would reject a tool_result block. Keep the output as plain text.
            let text = output.body.to_text().unwrap_or_default();
            self.push_text(
                Role::User,
                format!("Output of tool call {call_id}:\n{text}"),
            );
            return;
        };
        self.pending_tool_uses.remove(position);
        let mut content = tool_result_content(&output.body);
        if content.is_empty() {
            content.push(
                BlockKind::Text {
                    text: "(no output)".to_string(),
                }
                .into(),
            );
        }
        self.push(
            Role::User,
            BlockKind::ToolResult {
                tool_use_id: call_id.to_string(),
                content,
                is_error: output.success == Some(false),
            }
            .into(),
        );
    }

    fn push(&mut self, role: Role, block: ContentBlock) {
        // Parallel tool calls may keep extending the assistant turn, but once the user turn
        // carries anything besides tool results, or the next assistant turn starts, every
        // outstanding call must already be answered.
        let closes_pending = match role {
            Role::User => !matches!(block.kind, BlockKind::ToolResult { .. }),
            Role::Assistant => self
                .messages
                .last()
                .is_some_and(|message| message.role == Role::User),
        };
        if closes_pending && !self.pending_tool_uses.is_empty() {
            self.answer_pending_tool_uses();
        }
        match self.messages.last_mut() {
            Some(message) if message.role == role => message.content.push(block),
            _ => self.messages.push(Message {
                role,
                content: vec![block],
            }),
        }
    }

    /// Every tool_use must be answered by a tool_result in the next user turn.
    fn answer_pending_tool_uses(&mut self) {
        for call_id in std::mem::take(&mut self.pending_tool_uses) {
            self.push(
                Role::User,
                BlockKind::ToolResult {
                    tool_use_id: call_id,
                    content: vec![
                        BlockKind::Text {
                            text: MISSING_TOOL_OUTPUT.to_string(),
                        }
                        .into(),
                    ],
                    is_error: true,
                }
                .into(),
            );
        }
    }

    fn finish(mut self) -> Vec<Message> {
        self.answer_pending_tool_uses();
        self.messages
    }
}

fn content_item_block(item: &ContentItem) -> Option<ContentBlock> {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => {
            (!text.trim().is_empty()).then(|| BlockKind::Text { text: text.clone() }.into())
        }
        ContentItem::InputImage { image, .. } => image_block(image),
        ContentItem::InputAudio { .. } => None,
    }
}

fn tool_result_content(body: &FunctionCallOutputBody) -> Vec<ContentBlock> {
    match body {
        FunctionCallOutputBody::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![BlockKind::Text { text: text.clone() }.into()]
            }
        }
        FunctionCallOutputBody::ContentItems(items) => items
            .iter()
            .filter_map(|item| match item {
                FunctionCallOutputContentItem::InputText { text } => {
                    (!text.is_empty()).then(|| BlockKind::Text { text: text.clone() }.into())
                }
                FunctionCallOutputContentItem::InputImage { image, .. } => image_block(image),
                FunctionCallOutputContentItem::InputAudio { .. }
                | FunctionCallOutputContentItem::EncryptedContent { .. } => None,
            })
            .collect(),
    }
}

fn image_block(image: &ImageReference) -> Option<ContentBlock> {
    let source = match image {
        ImageReference::Inline { image_url } => match image_url.strip_prefix("data:") {
            Some(data_url) => {
                let (metadata, data) = data_url.split_once(',')?;
                let media_type = metadata.strip_suffix(";base64")?;
                ImageSource::Base64 {
                    media_type: media_type.to_string(),
                    data: data.to_string(),
                }
            }
            None => ImageSource::Url {
                url: image_url.clone(),
            },
        },
        // Uploaded OpenAI file references cannot be resolved by other providers.
        ImageReference::File { .. } => return None,
    };
    Some(BlockKind::Image { source }.into())
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
