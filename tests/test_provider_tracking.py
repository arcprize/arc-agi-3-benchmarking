from arcagi3.adapters.provider import ProviderAdapter
from arcagi3.utils.context import SessionContext


class DummyTrackingProvider(ProviderAdapter):
    """
    Minimal ProviderAdapter implementation to test call_with_tracking().

    We patch read_models_config in the test to avoid real models.yml access.
    """

    def init_client(self):
        return object()

    def make_prediction(self, prompt: str, task_id=None, test_id=None, pair_index=None):
        raise NotImplementedError()

    def extract_json_from_response(self, input_response: str):
        raise NotImplementedError()

    def extract_usage(self, response):
        # 10 prompt tokens, 5 completion tokens, 2 reasoning tokens
        return 10, 5, 2

    def extract_content(self, response):
        return "ok"

    def call_provider(self, messages):
        return {"ok": True}


def test_call_with_tracking_updates_session_context(monkeypatch):
    from arcagi3.adapters import provider as provider_module
    from arcagi3.utils import task_utils

    # Avoid config lookup / provider mismatch validation
    monkeypatch.setattr(
        task_utils,
        "read_models_config",
        lambda config: type(
            "ModelConfig",
            (),
            {
                "provider": "dummytrackingprovider",
                "pricing": type("Pricing", (), {"input": 1_000_000.0, "output": 2_000_000.0})(),
                "kwargs": {"memory_word_limit": 100},
                "is_multimodal": False,
            },
        )(),
    )
    # ProviderAdapter imports read_models_config into its module at import time
    monkeypatch.setattr(provider_module, "read_models_config", task_utils.read_models_config)

    provider = DummyTrackingProvider("dummy-config")
    ctx = SessionContext()

    provider.call_with_tracking(ctx, [{"role": "user", "content": "hi"}])

    assert ctx.metrics.total_usage.prompt_tokens == 10
    assert ctx.metrics.total_usage.completion_tokens == 5
    # reasoning tokens are stored in completion_tokens_details
    assert ctx.metrics.total_usage.completion_tokens_details is not None
    assert ctx.metrics.total_usage.completion_tokens_details.reasoning_tokens == 2

    # pricing is per 1M tokens, and we set pricing so that:
    # prompt_cost = 10 * 1.0, completion_cost = 5 * 2.0, reasoning_cost = 2 * 2.0
    assert ctx.metrics.total_cost.prompt_cost == 10.0
    assert ctx.metrics.total_cost.completion_cost == 10.0
    assert (ctx.metrics.total_cost.reasoning_cost or 0.0) == 4.0
    assert ctx.metrics.total_cost.total_cost == 24.0
