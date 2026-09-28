use super::super::tools::ToolTarget;
use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn registry() -> ToolRegistry {
    let tools = json!([
        {"type": "function", "name": "shell", "description": "", "strict": false, "parameters": {}},
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "",
            "format": {"type": "grammar", "syntax": "lark", "definition": ""},
        },
        {
            "type": "namespace",
            "name": "mcp__fs__",
            "description": "",
            "tools": [{"type": "function", "name": "read_file", "description": "", "strict": false, "parameters": {}}],
        },
    ]);
    let request = crate::common::ResponsesApiRequest {
        model: "claude".to_string(),
        instructions: String::new(),
        input: Vec::new(),
        tools: Some(crate::common::ResponsesApiTools::from(
            std::sync::Arc::from(
                serde_json::value::RawValue::from_string(tools.to_string()).expect("valid tools"),
            ),
        )),
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
    };
    let (_, registry) =
        super::super::request::build_messages_request(&request).expect("translate request");
    assert_eq!(
        registry.target("apply_patch"),
        Some(&ToolTarget {
            name: "apply_patch".to_string(),
            namespace: None,
            kind: ToolKind::Custom,
        })
    );
    registry
}

fn run(events: Vec<Value>) -> Vec<String> {
    let mut state = MessagesStreamState::new(registry());
    let mut output = Vec::new();
    for event in events {
        let (translated, completed) = state
            .handle(&event.to_string())
            .expect("event should translate");
        output.extend(translated.into_iter().map(|event| format!("{event:?}")));
        if completed {
            break;
        }
    }
    output
}

fn debug(event: ResponseEvent) -> String {
    format!("{event:?}")
}

#[test]
fn translates_text_tool_calls_and_usage() {
    let events = vec![
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 1,
                },
            },
        }),
        json!({"type": "ping"}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "mcp__fs__read_file", "input": {}}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"a.txt\"}"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "apply_patch", "input": {}}}),
        json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"input\":\"*** Begin Patch\"}"}}),
        json!({"type": "content_block_stop", "index": 2}),
        json!({"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_3", "name": "shell", "input": {}}}),
        json!({"type": "content_block_stop", "index": 3}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}),
        json!({"type": "message_stop"}),
    ];

    assert_eq!(
        run(events),
        vec![
            debug(ResponseEvent::Created {
                response_id: Some("msg_1".to_string()),
            }),
            debug(ResponseEvent::OutputItemAdded(assistant_message(
                String::new()
            ))),
            debug(ResponseEvent::OutputTextDelta("Hel".to_string())),
            debug(ResponseEvent::OutputTextDelta("lo".to_string())),
            debug(ResponseEvent::OutputItemDone(assistant_message(
                "Hello".to_string()
            ))),
            debug(ResponseEvent::OutputItemDone(function_call(
                "toolu_1".to_string(),
                "read_file".to_string(),
                Some("mcp__fs__".to_string()),
                r#"{"path":"a.txt"}"#.to_string(),
            ))),
            debug(ResponseEvent::OutputItemDone(
                ResponseItem::CustomToolCall {
                    id: None,
                    status: None,
                    call_id: "toolu_2".to_string(),
                    name: "apply_patch".to_string(),
                    namespace: None,
                    input: "*** Begin Patch".to_string(),
                    internal_chat_message_metadata_passthrough: None,
                }
            )),
            debug(ResponseEvent::OutputItemDone(function_call(
                "toolu_3".to_string(),
                "shell".to_string(),
                /*namespace*/ None,
                "{}".to_string(),
            ))),
            debug(ResponseEvent::Completed {
                response_id: "msg_1".to_string(),
                token_usage: Some(TokenUsage {
                    input_tokens: 60,
                    cached_input_tokens: 30,
                    cache_write_input_tokens: 20,
                    output_tokens: 42,
                    reasoning_output_tokens: 0,
                    total_tokens: 102,
                    codex_rollout_budget_units: None,
                }),
                usage_metadata: None,
                end_turn: None,
            }),
        ]
    );
}

#[test]
fn maps_stream_errors() {
    let mut state = MessagesStreamState::new(ToolRegistry::default());
    let error = state
        .handle(
            &json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            })
            .to_string(),
        )
        .expect_err("error event should fail the stream");
    assert!(matches!(
        error,
        ApiError::ServerOverloaded { retry_after: None }
    ));

    let error = state
        .handle(
            &json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "prompt is too long: 250000 tokens > 200000 maximum"},
            })
            .to_string(),
        )
        .expect_err("error event should fail the stream");
    assert!(matches!(error, ApiError::ContextWindowExceeded));
}
