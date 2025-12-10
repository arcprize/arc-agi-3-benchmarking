import textwrap

from arcagi3.prompts import PromptManager, PromptName


def test_default_prompts_load_and_have_expected_content():
    manager = PromptManager()
    system_prompt = manager.render(PromptName.SYSTEM)
    assert "abstract reasoning agent" in system_prompt

    analyze_prompt = manager.render(
        PromptName.ANALYZE_INSTRUCT, {"memory_limit": 500}
    )
    assert "memory scratchpad" in analyze_prompt
    assert "500" in analyze_prompt


def test_placeholder_extraction_and_validation_rejects_unknown():
    # Create a manager and then try to render with an override containing an
    # unknown placeholder; this should raise a ValueError.
    overrides = {
        "analyze_instruct": "Test with {{unknown_placeholder}}",
    }
    manager = PromptManager(overrides=overrides)
    try:
        manager.render(PromptName.ANALYZE_INSTRUCT, {"memory_limit": 10})
    except ValueError as e:
        assert "unknown_placeholder" in str(e)
    else:
        raise AssertionError("Expected ValueError for unknown placeholder")


def test_string_override_and_file_override_and_callable_override(tmp_path):
    # String override (literal template)
    overrides = {
        "compress_memory": "Size={{current_word_count}}, Limit={{memory_limit}}, Text={{memory_text}}"
    }
    manager = PromptManager(overrides=overrides)
    rendered = manager.render(
        PromptName.COMPRESS_MEMORY,
        {
            "current_word_count": 123,
            "memory_limit": 456,
            "memory_text": "hello",
        },
    )
    assert "Size=123" in rendered
    assert "Limit=456" in rendered
    assert "Text=hello" in rendered

    # File-path override
    file_template = tmp_path / "analyze.prompt"
    file_template.write_text(
        textwrap.dedent(
            """
            File based prompt with limit {{memory_limit}}.
            """
        ).strip(),
        encoding="utf-8",
    )
    manager = PromptManager(
        overrides={
            "analyze_instruct": str(file_template),
        }
    )
    rendered_file = manager.render(
        PromptName.ANALYZE_INSTRUCT, {"memory_limit": 42}
    )
    assert "limit 42" in rendered_file

    # Callable override
    def dynamic_prompt(vars):
        return f"Dynamic limit is {vars['memory_limit']}"

    manager = PromptManager(
        overrides={
            "analyze_instruct": dynamic_prompt,
        }
    )
    rendered_callable = manager.render(
        PromptName.ANALYZE_INSTRUCT, {"memory_limit": 99}
    )
    assert rendered_callable == "Dynamic limit is 99"


