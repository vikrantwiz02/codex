//! Streaming client for providers that speak the Anthropic Messages API (`POST /messages`).
//!
//! The client accepts the same [`ResponsesApiRequest`] and returns the same [`ResponseStream`]
//! as [`crate::ResponsesClient`], so callers can pick a client based on the provider's wire API
//! without changing how they build requests or consume events.

mod request;
mod stream;
mod tools;

use crate::auth::AuthProvider;
use crate::auth::AuthProviderFuture;
use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::ResponsesOptions;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_client::TransportError;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use http::header::AUTHORIZATION;
use request::build_messages_request;
use std::sync::Arc;
use stream::MessagesStreamState;
use stream::StreamError;
use stream::process_messages_sse;
use tokio::sync::mpsc;
use tracing::instrument;

const MESSAGES_PATH: &str = "/messages";
const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const API_KEY_HEADER: &str = "x-api-key";
const REQUEST_ID_HEADER: &str = "request-id";

pub struct AnthropicMessagesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> AnthropicMessagesClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        let auth: SharedAuthProvider = Arc::new(AnthropicApiKeyAuth { inner: auth });
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    #[instrument(
        name = "anthropic_messages.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "anthropic_messages_http",
            http.method = "POST",
            api.path = MESSAGES_PATH
        )
    )]
    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        options: ResponsesOptions,
    ) -> Result<ResponseStream, ApiError> {
        let (messages_request, tools) = build_messages_request(&request)?;
        let body = EncodedJsonBody::encode(&messages_request)
            .map_err(|e| ApiError::Stream(format!("failed to encode messages request: {e}")))?;

        let mut headers = options.extra_headers;
        if !headers.contains_key(ANTHROPIC_VERSION_HEADER) {
            headers.insert(
                ANTHROPIC_VERSION_HEADER,
                HeaderValue::from_static(ANTHROPIC_VERSION),
            );
        }

        let stream_response = self
            .session
            .stream_encoded_json_with(Method::POST, MESSAGES_PATH, headers, Some(body), |req| {
                req.headers.insert(
                    http::header::ACCEPT,
                    HeaderValue::from_static("text/event-stream"),
                );
            })
            .await
            .map_err(map_http_error)?;

        let upstream_request_id = stream_response
            .headers
            .get(REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let (tx_event, rx_event) = mpsc::channel(1600);
        tokio::spawn(process_messages_sse(
            stream_response.bytes,
            tx_event,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            MessagesStreamState::new(tools),
        ));

        Ok(ResponseStream {
            rx_event,
            upstream_request_id,
            interrupt: None,
        })
    }
}

/// Maps Anthropic error bodies onto Codex error kinds; auth failures stay transport errors so
/// callers can still run their unauthorized-recovery path.
fn map_http_error(error: ApiError) -> ApiError {
    let ApiError::Transport(TransportError::Http {
        status,
        body: Some(body),
        ..
    }) = &error
    else {
        return error;
    };
    if *status == http::StatusCode::UNAUTHORIZED || *status == http::StatusCode::FORBIDDEN {
        return error;
    }
    #[derive(serde::Deserialize)]
    struct ErrorBody {
        error: StreamError,
    }
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(ErrorBody { error: parsed }) => match parsed.into_api_error() {
            mapped @ (ApiError::ContextWindowExceeded | ApiError::ServerOverloaded { .. }) => {
                mapped
            }
            _ => error,
        },
        Err(_) => error,
    }
}

/// Sends API keys in Anthropic's `x-api-key` header instead of `Authorization: Bearer`.
///
/// Codex resolves provider API keys (for example from `env_key`) as bearer tokens. The Messages
/// API only accepts API keys through `x-api-key`, so bearer credentials are moved there unless
/// the request already carries an explicit `x-api-key` header.
struct AnthropicApiKeyAuth {
    inner: SharedAuthProvider,
}

impl AnthropicApiKeyAuth {
    fn move_bearer_to_api_key(headers: &mut HeaderMap) {
        if headers.contains_key(API_KEY_HEADER) {
            return;
        }
        let Some(api_key) = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .and_then(|api_key| HeaderValue::from_str(api_key).ok())
        else {
            return;
        };
        headers.remove(AUTHORIZATION);
        headers.insert(API_KEY_HEADER, api_key);
    }
}

impl AuthProvider for AnthropicApiKeyAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        self.inner.add_auth_headers(headers);
        Self::move_bearer_to_api_key(headers);
    }

    fn apply_auth(&self, request: codex_client::Request) -> AuthProviderFuture<'_> {
        Box::pin(async move {
            let mut request = self.inner.apply_auth(request).await?;
            Self::move_bearer_to_api_key(&mut request.headers);
            Ok(request)
        })
    }
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
