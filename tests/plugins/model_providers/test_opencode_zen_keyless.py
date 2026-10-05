"""Behavior contract for OpenCode Zen's explicitly keyless model route."""

from __future__ import annotations

import pytest

from hermes_cli.auth import AuthError
from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.model_switch import switch_model
from hermes_cli.runtime_provider import resolve_runtime_provider
from providers import get_provider_profile
from run_agent import AIAgent


def test_space_bunny_is_keyless_only_on_the_official_zen_route(monkeypatch):
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)
    profile = get_provider_profile("opencode-zen")

    assert profile.keyless_model_ids == frozenset({"space-bunny-free"})
    assert profile.supports_anonymous_access(
        model="opencode-zen/space-bunny-free",
        base_url="https://opencode.ai/zen/v1",
    )
    assert not profile.supports_anonymous_access(
        model="mimo-v2.5-free", base_url="https://opencode.ai/zen/v1"
    )
    assert not profile.supports_anonymous_access(
        model="space-bunny-free", base_url="https://proxy.example/v1"
    )

    runtime = resolve_runtime_provider(
        requested="opencode-zen", target_model="space-bunny-free"
    )
    assert runtime["provider"] == "opencode-zen"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == "https://opencode.ai/zen/v1"
    assert runtime["api_key"] == ""
    assert runtime["source"] == "anonymous-model"

    with pytest.raises(AuthError) as paid:
        resolve_runtime_provider(requested="opencode-zen", target_model="gpt-5.5")
    assert paid.value.code == "missing_api_key"

    selected = switch_model(
        "space-bunny-free",
        current_provider="custom",
        current_model="stealth/union-alpha",
        current_base_url="https://openrouter.ai/api/v1",
        explicit_provider="opencode-zen",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    assert selected.success is True
    # The resolved id is the auth-registry id, not the label the user typed: Zen's aliases
    # ("opencode", "zen") normalize onto one registry entry. Assert the route it resolved to
    # rather than the spelling, so an alias change is not a test failure.
    assert selected.target_provider in {"opencode-zen", "opencode"}
    assert selected.base_url == "https://opencode.ai/zen/v1"
    assert selected.api_key == ""

    paid_switch = switch_model(
        "gpt-5.5",
        current_provider="opencode-zen",
        current_model="space-bunny-free",
        current_base_url="https://opencode.ai/zen/v1",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    assert paid_switch.success is False
    assert "requires an API key" in paid_switch.error_message


def test_keyless_zen_is_explicit_in_the_ui_and_omits_sdk_authorization():
    ctx = ConfigContext(
        current_provider="custom",
        current_model="stealth/union-alpha",
        current_base_url="https://openrouter.ai/api/v1",
        user_providers={"opencode-zen": {"models": {"space-bunny-free": {}}}},
        custom_providers=[],
    )
    payload = build_models_payload(
        ctx,
        explicit_only=True,
        picker_hints=True,
        non_blocking_catalogs=True,
        probe_custom_providers=False,
    )
    row = next(r for r in payload["providers"] if r.get("slug") == "opencode-zen")
    assert row["models"] == ["space-bunny-free"]
    assert row["authenticated"] is True

    agent = AIAgent(
        api_key="",
        base_url="https://opencode.ai/zen/v1",
        model="space-bunny-free",
        provider="opencode-zen",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-test",
    )
    try:
        assert agent.api_key == ""
        assert agent.client.api_key == ""
        assert agent.client.auth_headers == {}
        assert agent.client.default_headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
        assert agent.client.default_headers["X-Title"] == "Hermes Agent"
    finally:
        agent.client.close()


def test_switching_onto_the_keyless_route_drops_the_previous_providers_key():
    """A switch onto an anonymous route must not carry the previous provider's credential to the
    new endpoint. `agent.switch_model` historically treated a falsy `api_key` as "caller said
    nothing" and kept `agent.api_key`, which paired the old provider's bearer token with the new
    provider's base_url — so an OpenRouter key reached opencode.ai. `None` still means "no
    decision" (best-effort credential refresh); `""` is a decision to send no Authorization."""
    agent = AIAgent(
        api_key="sk-or-v1-previous-provider-key",
        base_url="https://openrouter.ai/api/v1",
        model="stealth/union-alpha",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-switch-test",
    )
    try:
        agent.switch_model(
            new_model="space-bunny-free",
            new_provider="opencode-zen",
            api_key="",
            base_url="https://opencode.ai/zen/v1",
        )

        assert agent.base_url == "https://opencode.ai/zen/v1"
        assert agent.api_key == ""
        assert agent._client_kwargs["api_key"] == ""
        assert agent._client_kwargs["base_url"] == "https://opencode.ai/zen/v1"
        # The rebuilt client must carry no Authorization header at all, not a stale bearer.
        assert agent.client.auth_headers == {}
    finally:
        agent.client.close()


def test_api_key_none_still_means_no_decision_on_a_plain_provider_switch():
    """The guard above must not break the best-effort credential refresh: `None` keeps the
    session's existing key so a same-provider re-select still authenticates."""
    agent = AIAgent(
        api_key="existing-key",
        base_url="https://openrouter.ai/api/v1",
        model="stealth/union-alpha",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id="opencode-zen-keyless-none-test",
    )
    try:
        agent.switch_model(
            new_model="stealth/union-alpha",
            new_provider="custom",
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
        )

        assert agent.api_key == "existing-key"
        assert agent._client_kwargs["api_key"] == "existing-key"
    finally:
        agent.client.close()
