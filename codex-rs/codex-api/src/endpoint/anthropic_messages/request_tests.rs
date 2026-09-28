use super::*;
use crate::common::ResponsesApiTools;
use crate::endpoint::anthropic_messages::tools::MAX_TOOL_NAME_LEN;
use crate::endpoint::anthropic_messages::tools::ToolKind;
use crate::endpoint::anthropic_messages::tools::ToolTarget;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use pretty_assertions::assert_eq;
use serde_json::value::RawValue;
use std::sync::Arc;

fn request(input: Vec<ResponseItem>, tools: Option<Value>) -> ResponsesApiRequest {
    ResponsesApiRequest {
        model: "claude-sonnet-4-5".to_string(),
        instructions: "You are Codex.".to_string(),
        input,
        tools: tools.map(|tools| {
            let raw: Arc<RawValue> =
                Arc::from(RawValue::from_string(tools.to_string()).expect("valid tools json"));
            ResponsesApiTools::from(raw)
        }),
        tool_choice: "auto".to_string(),
        parallel_tool_calls: true,
        reasoning: None,
        store: false,
        stream: true,
        stream_options: None,
        include: Vec::new(),
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
        access_programs: None,
    }
}

fn message(role: &str, text: &str) -> ResponseItem {
    let content = if role == "assistant" {
        ContentItem::OutputText {
            text: text.to_string(),
        }
    } else {
        ContentItem::InputText {
            text: text.to_string(),
        }
    };
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![content],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_call(
    call_id: &str,
    name: &str,
    namespace: Option<&str>,
    arguments: &str,
) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: name.to_string(),
        namespace: namespace.map(str::to_string),
        arguments: arguments.to_string(),
        encrypted_function_args: None,
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn function_output(call_id: &str, text: &str, success: Option<bool>) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: Some(call_id.to_string()),
        name: None,
        namespace: None,
        output: FunctionCallOutputPayload {
            body: FunctionCallOutputBody::Text(text.to_string()),
            success,
        },
        internal_chat_message_metadata_passthrough: None,
    }
}

fn to_json(request: &MessagesRequest) -> Value {
    serde_json::to_value(request).expect("serialize messages request")
}

#[test]
fn translates_conversation_and_tools() {
    let input = vec![
        message("developer", "Sandbox: workspace-write"),
        message("user", "List the files."),
        ResponseItem::Reasoning {
            id: None,
            summary: Vec::new(),
            content: None,
            encrypted_content: Some("opaque".to_string()),
            internal_chat_message_metadata_passthrough: None,
        },
        message("assistant", "Listing files."),
        function_call("call_1", "shell", None, r#"{"command":["ls"]}"#),
        function_call(
            "call_2",
            "read_file",
            Some("mcp__fs__"),
            r#"{"path":"a.txt"}"#,
        ),
        function_output("call_1", "a.txt", Some(true)),
        function_output("call_2", "missing", Some(false)),
        ResponseItem::CustomToolCall {
            id: None,
            status: None,
            call_id: "call_3".to_string(),
            name: "apply_patch".to_string(),
            namespace: None,
            input: "*** Begin Patch".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::CustomToolCallOutput {
            id: None,
            call_id: "call_3".to_string(),
            name: None,
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("Done".to_string()),
                success: None,
            },
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let tools = json!([
        {
            "type": "function",
            "name": "shell",
            "description": "Run a command.",
            "strict": false,
            "parameters": {"type": "object", "properties": {"command": {"type": "array"}}},
        },
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch.",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: patch"},
        },
        {
            "type": "namespace",
            "name": "mcp__fs__",
            "description": "Filesystem server.",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "strict": false,
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            }],
        },
        {"type": "web_search"},
    ]);

    let (messages_request, registry) =
        build_messages_request(&request(input, Some(tools))).expect("translate request");

    assert_eq!(
        to_json(&messages_request),
        json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": [{
                "type": "text",
                "text": "You are Codex.",
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Sandbox: workspace-write"},
                        {"type": "text", "text": "List the files."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Listing files."},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "shell",
                            "input": {"command": ["ls"]},
                        },
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "mcp__fs__read_file",
                            "input": {"path": "a.txt"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": [{"type": "text", "text": "a.txt"}],
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_2",
                            "content": [{"type": "text", "text": "missing"}],
                            "is_error": true,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "call_3",
                        "name": "apply_patch",
                        "input": {"input": "*** Begin Patch"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_3",
                        "content": [{"type": "text", "text": "Done"}],
                        "cache_control": {"type": "ephemeral"},
                    }],
                },
            ],
            "tools": [
                {
                    "name": "shell",
                    "description": "Run a command.",
                    "input_schema": {"type": "object", "properties": {"command": {"type": "array"}}},
                },
                {
                    "name": "apply_patch",
                    "description": "Apply a patch.\n\nThe `input` argument must follow this lark grammar:\nstart: patch",
                    "input_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Raw tool input."}},
                        "required": ["input"],
                    },
                },
                {
                    "name": "mcp__fs__read_file",
                    "description": "Read a file.",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": false},
            "stream": true,
        })
    );
    assert_eq!(
        registry.target("mcp__fs__read_file"),
        Some(&ToolTarget {
            name: "read_file".to_string(),
            namespace: Some("mcp__fs__".to_string()),
            kind: ToolKind::Function,
        })
    );
}

#[test]
fn repairs_unpaired_tool_calls_and_outputs() {
    let input = vec![
        message("user", "Start."),
        function_call("call_1", "shell", None, "not json"),
        message("user", "Interrupted, try again."),
        function_output("call_orphan", "stale output", None),
    ];

    let (messages_request, _) =
        build_messages_request(&request(input, /*tools*/ None)).expect("translate request");

    assert_eq!(
        to_json(&messages_request)["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "Start."}]},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "shell",
                    "input": {"arguments": "not json"},
                }],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": MISSING_TOOL_OUTPUT}],
                        "is_error": true,
                    },
                    {"type": "text", "text": "Interrupted, try again."},
                    {
                        "type": "text",
                        "text": "Output of tool call call_orphan:\nstale output",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ])
    );
}

#[test]
fn sanitizes_and_disambiguates_tool_names() {
    let long_name = "x".repeat(80);
    let tools = json!([
        {
            "type": "namespace",
            "name": "mcp__my server__",
            "description": "",
            "tools": [
                {"type": "function", "name": "look.up", "description": "", "strict": false, "parameters": {}},
                {"type": "function", "name": &long_name, "description": "", "strict": false, "parameters": {}},
            ],
        },
        {"type": "function", "name": format!("mcp__my_server__{long_name}"), "description": "", "strict": false, "parameters": {}},
    ]);

    let (messages_request, registry) =
        build_messages_request(&request(Vec::new(), Some(tools))).expect("translate request");

    let names: Vec<&str> = messages_request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    let truncated = format!("mcp__my_server__{long_name}")[..MAX_TOOL_NAME_LEN].to_string();
    let disambiguated = format!("{}_2", &truncated[..MAX_TOOL_NAME_LEN - 2]);
    assert_eq!(
        names,
        vec![
            "mcp__my_server__look_up",
            truncated.as_str(),
            disambiguated.as_str()
        ]
    );
    assert_eq!(
        registry.target(&disambiguated),
        Some(&ToolTarget {
            name: format!("mcp__my_server__{long_name}"),
            namespace: None,
            kind: ToolKind::Function,
        })
    );
}

#[test]
fn converts_inline_images() {
    let input = vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![
            ContentItem::InputImage {
                image: ImageReference::Inline {
                    image_url: "data:image/png;base64,AAAA".to_string(),
                },
                detail: None,
            },
            ContentItem::InputImage {
                image: ImageReference::Inline {
                    image_url: "https://example.com/cat.png".to_string(),
                },
                detail: None,
            },
        ],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }];

    let (messages_request, _) =
        build_messages_request(&request(input, /*tools*/ None)).expect("translate request");

    assert_eq!(
        to_json(&messages_request)["messages"],
        json!([{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                },
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/cat.png"},
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }])
    );
}
