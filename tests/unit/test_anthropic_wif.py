"""Exercise AWS identity exchange and refresh through the real Anthropic SDK."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import anthropic
import boto3
import httpx
import pytest
from anthropic.lib.credentials import TokenCache
from botocore.stub import Stubber

from benchmarking import runtime_clients

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("workspace", ["wrkspc_test", "wrkspc_other"])
def test_wif_exchanges_caches_and_refreshes_with_fresh_aws_tokens(
    monkeypatch, anthropic_wif_env, workspace
):
    # Existing Batch jobs can still inject legacy keys. Explicit WIF must win.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "unused-auth-token")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", workspace)
    sts = boto3.client(
        "sts",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    sts_factory = Mock(return_value=sts)
    monkeypatch.setattr(runtime_clients.boto3, "client", sts_factory)
    exchanges = []
    message_headers = []
    clock = [time.time()]

    def respond(request):
        if request.url.path == "/v1/oauth/token":
            exchanges.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "access_token": f"access-token-{len(exchanges)}",
                    "token_type": "Bearer",
                    "expires_in": 600,
                },
            )
        assert request.url.path == "/v1/messages"
        message_headers.append(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
        Stubber(sts) as stub,
    ):
        for token in ("aws-jwt-1", "aws-jwt-2"):
            stub.add_response(
                "get_web_identity_token",
                {
                    "WebIdentityToken": token,
                    "Expiration": datetime.now(timezone.utc) + timedelta(seconds=900),
                },
                {
                    "Audience": ["https://api.anthropic.com"],
                    "SigningAlgorithm": "RS256",
                    "DurationSeconds": 900,
                },
            )

        monkeypatch.setattr(
            runtime_clients,
            "WorkloadIdentityCredentials",
            lambda **kwargs: anthropic.WorkloadIdentityCredentials(
                **kwargs, http_client=http_client
            ),
        )
        monkeypatch.setattr(
            runtime_clients,
            "AnthropicClient",
            lambda **kwargs: anthropic.Anthropic(
                **kwargs,
                http_client=http_client,
                max_retries=0,
                _token_cache=TokenCache(
                    kwargs["credentials"], time_source=lambda: clock[0]
                ),
            ),
        )
        client = runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "anthropic-python"},
            client_config={},
            config_id="anthropic-config",
        )
        sts_factory.assert_not_called()

        for advance in (0, 0, 601):
            clock[0] += advance
            response = client.messages.create(
                model="test-model",
                max_tokens=16,
                messages=[{"role": "user", "content": "test"}],
            )
            assert response.content[0].text == "ok"

        stub.assert_no_pending_responses()
        assert sts_factory.call_count == 2
        sts_factory.assert_called_with("sts", region_name="us-east-1")
        assert [exchange["assertion"] for exchange in exchanges] == [
            "aws-jwt-1",
            "aws-jwt-2",
        ]
        for exchange in exchanges:
            assert (
                exchange["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
            )
            assert exchange["federation_rule_id"] == "fdrl_test"
            assert exchange["organization_id"] == "00000000-0000-0000-0000-000000000001"
            assert exchange["service_account_id"] == "svac_test"
            assert exchange["workspace_id"] == workspace
        assert [headers["authorization"] for headers in message_headers] == [
            "Bearer access-token-1",
            "Bearer access-token-1",
            "Bearer access-token-2",
        ]
        assert all("x-api-key" not in headers for headers in message_headers)


def test_aws_failure_does_not_fall_back_to_api_key(monkeypatch, anthropic_wif_env):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-api-key")
    monkeypatch.setattr(
        runtime_clients.boto3,
        "client",
        Mock(side_effect=RuntimeError("AWS unavailable")),
    )
    send = Mock(side_effect=AssertionError("No HTTP request expected"))
    with httpx.Client(transport=httpx.MockTransport(send)) as http_client:
        monkeypatch.setattr(
            runtime_clients,
            "AnthropicClient",
            lambda **kwargs: anthropic.Anthropic(
                **kwargs, http_client=http_client, max_retries=0
            ),
        )
        with runtime_clients.build_model_runtime_client(
            runtime_config={"sdk": "anthropic-python"},
            client_config={},
            config_id="anthropic-config",
        ) as client:
            with pytest.raises(anthropic.APIConnectionError) as exc_info:
                client.messages.create(
                    model="test-model",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "test"}],
                )
            assert isinstance(exc_info.value.__cause__, RuntimeError)
            assert str(exc_info.value.__cause__) == "AWS unavailable"
        send.assert_not_called()
