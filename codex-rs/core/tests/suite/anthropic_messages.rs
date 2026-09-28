//! End-to-end coverage for providers that speak the Anthropic Messages API.

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event_match;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

const MODEL: &str = "claude-sonnet-4-5";

fn messages_sse(events: &[Value]) -> ResponseTemplate {
    let body: String = events
        .iter()
        .map(|event| {
            let kind = event["type"].as_str().unwrap_or_default();
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect();
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_provider_runs_tool_calls_and_returns_results() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let plan_arguments = json!({
        "plan": [{"step": "Say hello", "status": "in_progress"}],
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(messages_sse(&[
            json!({"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 12, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "update_plan", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": plan_arguments.to_string()}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
            json!({"type": "message_stop"}),
        ]))
        .up_to_n_times(1)
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(messages_sse(&[
            json!({"type": "message_start", "message": {"id": "msg_2", "usage": {"input_tokens": 40, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello from Claude"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            json!({"type": "message_stop"}),
        ]))
        .with_priority(2)
        .expect(1)
        .mount(&server)
        .await;

    let base_url = format!("{}/v1", server.uri());
    let test = test_codex()
        .with_model(MODEL)
        .with_config(move |config| {
            config.update_plan_enabled = true;
            config.model_provider = ModelProviderInfo {
                base_url: Some(base_url),
                // Any variable that is always set works; the mock does not check the key.
                env_key: Some("PATH".to_string()),
                ..ModelProviderInfo::create_anthropic_provider()
            };
        })
        .build_with_auto_env(&server)
        .await?;

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Make a plan, then say hello.".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;

    let last_agent_message = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::TurnComplete(event) => Some(event.last_agent_message.clone()),
        _ => None,
    })
    .await;
    assert_eq!(last_agent_message.as_deref(), Some("Hello from Claude"));

    let requests: Vec<_> = server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|request| request.url.path() == "/v1/messages")
        .collect();
    assert_eq!(requests.len(), 2);

    let first: Value = serde_json::from_slice(&requests[0].body)?;
    assert_eq!(first["model"], MODEL);
    assert_eq!(first["stream"], true);
    assert!(
        first["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().any(|tool| tool["name"] == "update_plan")),
        "update_plan should be offered as an Anthropic tool: {first}"
    );
    assert_eq!(
        requests[0]
            .headers
            .get("anthropic-version")
            .and_then(|value| value.to_str().ok()),
        Some("2023-06-01")
    );
    assert!(requests[0].headers.get("x-api-key").is_some());
    assert!(requests[0].headers.get("authorization").is_none());

    let second: Value = serde_json::from_slice(&requests[1].body)?;
    let messages = second["messages"].as_array().cloned().unwrap_or_default();
    let tool_use_index = messages
        .iter()
        .position(|message| {
            message["role"] == "assistant"
                && message["content"]
                    .as_array()
                    .is_some_and(|content| content.iter().any(|block| block["type"] == "tool_use"))
        })
        .expect("second request should replay the tool call");
    assert_eq!(
        messages[tool_use_index]["content"],
        json!([{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "update_plan",
            "input": plan_arguments,
        }])
    );
    let tool_result = messages[tool_use_index + 1]["content"][0].clone();
    assert_eq!(
        tool_result,
        json!({
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [{"type": "text", "text": "Plan updated"}],
            "cache_control": {"type": "ephemeral"},
        })
    );

    Ok(())
}
