import importlib.util
from pathlib import Path


def _load_module_from_path(module_path: Path):
    spec = importlib.util.spec_from_file_location("tmp_prompt_module", str(module_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_prompt_manager_loads_prompts_relative_to_caller(tmp_path):
    # Arrange a fake module on disk with its own ./prompts directory.
    mod_dir = tmp_path / "foo" / "bar"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "myprompt.prompt").write_text("Hello {{x}}", encoding="utf-8")
    (prompts_dir / "noext").write_text("NoExt {{y}}", encoding="utf-8")

    module_path = mod_dir / "file.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    a = mgr.render('myprompt', {'x': 123})",
                "    b = mgr.render('noext', {'y': 'ok'})",
                "    return a, b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    a, b = mod.run()
    assert a == "Hello 123"
    assert b == "NoExt ok"


def test_prompt_manager_validates_template_variables(tmp_path):
    """Test that render() validates template variables strictly."""
    from arcagi3.prompts import PromptManager

    # Create a test module with prompts
    mod_dir = tmp_path / "test_mod"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "test.prompt").write_text("Hello {{x}} and {{y}}", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    # Missing variable should raise",
                "    try:",
                "        mgr.render('test', {'x': 1})",
                "        return 'FAIL'",
                "    except ValueError as e:",
                "        if 'Missing variable' in str(e):",
                "            pass  # expected",
                "    # Extra variable should raise",
                "    try:",
                "        mgr.render('test', {'x': 1, 'y': 2, 'z': 3})",
                "        return 'FAIL'",
                "    except ValueError as e:",
                "        if 'Extra variable' in str(e):",
                "            pass  # expected",
                "    # Correct variables should work",
                "    return mgr.render('test', {'x': 1, 'y': 2})",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "Hello 1 and 2"


