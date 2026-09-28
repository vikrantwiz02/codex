//! Translation of Codex tool definitions into Anthropic tools.
//!
//! Function tools map directly. Freeform (custom) tools take a single `input` string argument,
//! with their grammar appended to the description. Tools inside a namespace (for example MCP
//! servers) are flattened into one valid wire name, and the [`ToolRegistry`] maps wire names back
//! to the Codex tool when the model calls one. Hosted tools such as web search are skipped.

use crate::error::ApiError;
use codex_protocol::DEFAULT_FUNCTION_NAMESPACE;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// Anthropic tool names must match `^[a-zA-Z0-9_-]{1,64}$`.
pub(crate) const MAX_TOOL_NAME_LEN: usize = 64;
/// Argument name used to carry a freeform tool's raw input.
pub(crate) const CUSTOM_TOOL_INPUT_FIELD: &str = "input";

#[derive(Debug, Serialize, PartialEq)]
pub(crate) struct Tool {
    pub(crate) name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(crate) description: String,
    pub(crate) input_schema: Value,
}

/// How a model-visible tool call must be surfaced back to Codex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolKind {
    Function,
    Custom,
}

/// The Codex tool behind an Anthropic wire tool name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolTarget {
    pub(crate) name: String,
    pub(crate) namespace: Option<String>,
    pub(crate) kind: ToolKind,
}

/// Bidirectional mapping between Codex tool names and Anthropic wire tool names.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ToolRegistry {
    by_wire_name: HashMap<String, ToolTarget>,
    wire_names: HashMap<(Option<String>, String), String>,
}

impl ToolRegistry {
    pub(crate) fn target(&self, wire_name: &str) -> Option<&ToolTarget> {
        self.by_wire_name.get(wire_name)
    }

    pub(crate) fn wire_name(&self, namespace: Option<&str>, name: &str) -> String {
        self.wire_names
            .get(&(namespace.map(str::to_string), name.to_string()))
            .cloned()
            .unwrap_or_else(|| encode_tool_name(namespace, name))
    }

    fn register(&mut self, target: ToolTarget) -> String {
        let base = encode_tool_name(target.namespace.as_deref(), &target.name);
        let mut wire_name = base.clone();
        let mut suffix = 2;
        while self.by_wire_name.contains_key(&wire_name) {
            let tail = format!("_{suffix}");
            let keep = base.len().min(MAX_TOOL_NAME_LEN - tail.len());
            wire_name = format!("{}{tail}", &base[..keep]);
            suffix += 1;
        }
        self.wire_names.insert(
            (target.namespace.clone(), target.name.clone()),
            wire_name.clone(),
        );
        self.by_wire_name.insert(wire_name.clone(), target);
        wire_name
    }
}

/// Flattens a possibly namespaced Codex tool name into a valid Anthropic tool name.
fn encode_tool_name(namespace: Option<&str>, name: &str) -> String {
    let joined = match namespace {
        None => name.to_string(),
        Some(namespace) if namespace.is_empty() || namespace == DEFAULT_FUNCTION_NAMESPACE => {
            name.to_string()
        }
        Some(namespace) if namespace.ends_with('_') => format!("{namespace}{name}"),
        Some(namespace) => format!("{namespace}__{name}"),
    };
    let mut sanitized: String = joined
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    sanitized.truncate(MAX_TOOL_NAME_LEN);
    if sanitized.is_empty() {
        sanitized.push('_');
    }
    sanitized
}

pub(crate) fn translate_tools(raw_tools: &str) -> Result<(Vec<Tool>, ToolRegistry), ApiError> {
    let specs: Vec<Value> = serde_json::from_str(raw_tools)
        .map_err(|err| ApiError::Stream(format!("failed to decode tool definitions: {err}")))?;
    let mut registry = ToolRegistry::default();
    let mut tools = Vec::new();
    for spec in &specs {
        match spec.get("type").and_then(Value::as_str) {
            Some("namespace") => {
                let namespace = spec.get("name").and_then(Value::as_str).unwrap_or_default();
                for inner in spec
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(tool) = translate_tool(inner, Some(namespace), &mut registry) {
                        tools.push(tool);
                    }
                }
            }
            _ => {
                if let Some(tool) = translate_tool(spec, /*namespace*/ None, &mut registry) {
                    tools.push(tool);
                }
            }
        }
    }
    Ok((tools, registry))
}

fn translate_tool(
    spec: &Value,
    namespace: Option<&str>,
    registry: &mut ToolRegistry,
) -> Option<Tool> {
    let kind = match spec.get("type").and_then(Value::as_str) {
        Some("function") => ToolKind::Function,
        Some("custom") => ToolKind::Custom,
        // Hosted tools such as web_search and tool_search have no Messages API equivalent.
        _ => return None,
    };
    let name = spec.get("name").and_then(Value::as_str)?.to_string();
    let mut description = spec
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let input_schema = match kind {
        ToolKind::Function => spec
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
        ToolKind::Custom => {
            if let Some(definition) = spec
                .pointer("/format/definition")
                .and_then(Value::as_str)
                .filter(|definition| !definition.trim().is_empty())
            {
                let syntax = spec
                    .pointer("/format/syntax")
                    .and_then(Value::as_str)
                    .unwrap_or("grammar");
                description.push_str(&format!(
                    "\n\nThe `{CUSTOM_TOOL_INPUT_FIELD}` argument must follow this {syntax} grammar:\n{definition}"
                ));
            }
            json!({
                "type": "object",
                "properties": {
                    CUSTOM_TOOL_INPUT_FIELD: {
                        "type": "string",
                        "description": "Raw tool input.",
                    },
                },
                "required": [CUSTOM_TOOL_INPUT_FIELD],
            })
        }
    };
    let wire_name = registry.register(ToolTarget {
        name,
        namespace: namespace.map(str::to_string),
        kind,
    });
    Some(Tool {
        name: wire_name,
        description,
        input_schema,
    })
}
