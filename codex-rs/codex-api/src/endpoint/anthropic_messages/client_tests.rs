use super::*;
use crate::common::ResponseEvent;
use bytes::Bytes;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Clone)]
struct CapturingTransport {
    body: String,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl HttpTransport for CapturingTransport {
    async fn execute(&self, _req: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.requests.lock().expect("requests lock").push(req);
        let stream = futures::stream::iter(vec![Ok::<Bytes, TransportError>(Bytes::from(
            self.body.clone(),
        ))]);
        let mut headers = HeaderMap::new();
        headers.insert(REQUEST_ID_HEADER, HeaderValue::from_static("req_123"));
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers,
            bytes: Box::pin(stream),
        })
    }
}

struct BearerAuth;

impl AuthProvider for BearerAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-ant-test"),
        );
    }
}

fn provider() -> Provider {
    Provider {
        name: "Anthropic".to_string(),
        base_url: "https://api.anthropic.com/v1".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: crate::provider::RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(5),
    }
}

fn sse_body(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| {
            let kind = event["type"].as_str().expect("event type");
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect()
}

#[tokio::test]
async fn streams_messages_with_api_key_header() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let transport = CapturingTransport {
        body: sse_body(&[
            json!({"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            json!({"type": "message_stop"}),
        ]),
        requests: Arc::clone(&requests),
    };
    let client = AnthropicMessagesClient::new(transport, provider(), Arc::new(BearerAuth));
    let request = ResponsesApiRequest {
        model: "claude-sonnet-4-5".to_string(),
        instructions: String::new(),
        input: vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Hello".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }],
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: false,
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

    let mut stream = client
        .stream_request(request, ResponsesOptions::default())
        .await
        .expect("stream should start");
    assert_eq!(stream.upstream_request_id.as_deref(), Some("req_123"));
    let mut texts = Vec::new();
    let mut completed = false;
    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
                texts.push(content);
            }
            ResponseEvent::Completed { response_id, .. } => {
                assert_eq!(response_id, "msg_1");
                completed = true;
            }
            _ => {}
        }
    }
    assert!(completed);
    assert_eq!(
        texts,
        vec![vec![ContentItem::OutputText {
            text: "Hi".to_string()
        }]]
    );

    let requests = requests.lock().expect("requests lock");
    let [sent] = requests.as_slice() else {
        panic!("expected exactly one request, got {}", requests.len());
    };
    assert_eq!(sent.url, "https://api.anthropic.com/v1/messages");
    assert_eq!(sent.headers.get(AUTHORIZATION), None);
    assert_eq!(
        sent.headers.get(API_KEY_HEADER),
        Some(&HeaderValue::from_static("sk-ant-test"))
    );
    assert_eq!(
        sent.headers.get(ANTHROPIC_VERSION_HEADER),
        Some(&HeaderValue::from_static(ANTHROPIC_VERSION))
    );
    let body: Value = match sent.body.as_ref() {
        Some(RequestBody::EncodedJson(body)) => {
            serde_json::from_slice(body.as_bytes()).expect("json body")
        }
        Some(RequestBody::Raw(bytes)) => serde_json::from_slice(bytes).expect("json body"),
        Some(RequestBody::Json(value)) => value.clone(),
        None => panic!("request body missing"),
    };
    assert_eq!(
        body,
        json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 32000,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}],
            }],
            "stream": true,
        })
    );
}

#[test]
fn maps_context_window_http_errors() {
    let error = map_http_error(ApiError::Transport(TransportError::Http {
        status: StatusCode::BAD_REQUEST,
        url: None,
        headers: None,
        body: Some(
            json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "prompt is too long: 1 tokens > 0 maximum"},
            })
            .to_string(),
        ),
        retry_after: None,
    }));
    assert!(matches!(error, ApiError::ContextWindowExceeded));
}
