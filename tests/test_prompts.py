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


def test_prompt_manager_validates_missing_variables(tmp_path):
    """Test that render() raises error for missing template variables."""

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
                "            return 'OK'",
                "        return 'FAIL'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "OK"


def test_prompt_manager_allows_extra_variables(tmp_path):
    """Test that render() allows extra variables (useful for conditionals)."""

    mod_dir = tmp_path / "test_extra"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "test.prompt").write_text("Hello {{x}}", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    # Extra variables should be allowed",
                "    result = mgr.render('test', {'x': 1, 'y': 2, 'z': 3})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "Hello 1"


def test_prompt_manager_supports_jinja2_conditionals(tmp_path):
    """Test that render() supports jinja2 style conditionals."""

    # Create a test module with prompts
    mod_dir = tmp_path / "test_jinja"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """Hello {{name}}!
{% if use_vision %}
You will see images of the game state.
{% else %}
You will see text grids representing the game state.
{% endif %}
Good luck!"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    # With vision enabled",
                "    result1 = mgr.render('test', {'name': 'Agent', 'use_vision': True})",
                "    # With vision disabled",
                "    result2 = mgr.render('test', {'name': 'Agent', 'use_vision': False})",
                "    return result1, result2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result1, result2 = mod.run()
    assert "images" in result1
    assert "images" not in result2
    assert "text grids" in result2
    assert "text grids" not in result1


def test_prompt_manager_jinja2_loops(tmp_path):
    """Test jinja2 for loops in templates."""

    mod_dir = tmp_path / "test_loops"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """Items:
{% for item in items %}
- {{ item }}
{% endfor %}
Done."""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {'items': ['apple', 'banana', 'cherry']})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "apple" in result
    assert "banana" in result
    assert "cherry" in result
    assert "Done" in result


def test_prompt_manager_jinja2_filters(tmp_path):
    """Test jinja2 filters in templates."""

    mod_dir = tmp_path / "test_filters"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """Name: {{ name|upper }}
Count: {{ count|default(0) }}
Length: {{ items|length }}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {'name': 'hello', 'items': [1, 2, 3]})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "HELLO" in result
    assert "0" in result  # default for missing count
    assert "3" in result  # length of items


def test_prompt_manager_jinja2_nested_conditionals(tmp_path):
    """Test nested jinja2 conditionals."""

    mod_dir = tmp_path / "test_nested"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """{% if enabled %}
Enabled!
{% if debug %}
Debug mode is on.
{% else %}
Debug mode is off.
{% endif %}
{% else %}
Disabled!
{% endif %}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result1 = mgr.render('test', {'enabled': True, 'debug': True})",
                "    result2 = mgr.render('test', {'enabled': True, 'debug': False})",
                "    result3 = mgr.render('test', {'enabled': False})",
                "    return result1, result2, result3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result1, result2, result3 = mod.run()
    assert "Enabled!" in result1
    assert "Debug mode is on" in result1
    assert "Enabled!" in result2
    assert "Debug mode is off" in result2
    assert "Disabled!" in result3


def test_prompt_manager_elif_conditional(tmp_path):
    """Test jinja2 elif conditionals."""

    mod_dir = tmp_path / "test_elif"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """{% if status == 'active' %}
Status is active
{% elif status == 'inactive' %}
Status is inactive
{% else %}
Status is unknown
{% endif %}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    r1 = mgr.render('test', {'status': 'active'})",
                "    r2 = mgr.render('test', {'status': 'inactive'})",
                "    r3 = mgr.render('test', {'status': 'other'})",
                "    return r1, r2, r3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    r1, r2, r3 = mod.run()
    assert "active" in r1
    assert "inactive" in r2
    assert "unknown" in r3


def test_prompt_manager_template_caching(tmp_path):
    """Test that templates are cached and reused."""

    mod_dir = tmp_path / "test_cache"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "test.prompt").write_text("Hello {{name}}", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    # First render",
                "    r1 = mgr.render('test', {'name': 'Alice'})",
                "    # Second render (should use cached template)",
                "    r2 = mgr.render('test', {'name': 'Bob'})",
                "    return r1, r2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    r1, r2 = mod.run()
    assert r1 == "Hello Alice"
    assert r2 == "Hello Bob"


def test_prompt_manager_empty_template(tmp_path):
    """Test rendering empty or minimal templates."""

    mod_dir = tmp_path / "test_empty"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "empty.prompt").write_text("", encoding="utf-8")
    (prompts_dir / "minimal.prompt").write_text("Just text", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    r1 = mgr.render('empty')",
                "    r2 = mgr.render('minimal')",
                "    return r1, r2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    r1, r2 = mod.run()
    assert r1 == ""
    assert r2 == "Just text"


def test_prompt_manager_no_variables(tmp_path):
    """Test rendering template with no variables passed."""

    mod_dir = tmp_path / "test_no_vars"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "test.prompt").write_text("Static text only", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test')",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "Static text only"


def test_prompt_manager_whitespace_trimming(tmp_path):
    """Test that jinja2 trim_blocks and lstrip_blocks work correctly."""

    mod_dir = tmp_path / "test_whitespace"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    # Template with blocks that should be trimmed
    template = """Start
{% if condition %}
Conditional text
{% endif %}
End"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {'condition': True})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    # Should not have extra blank lines from the {% if %} block
    lines = [line for line in result.split("\n") if line.strip()]
    assert "Start" in lines
    assert "Conditional text" in lines
    assert "End" in lines


def test_prompt_manager_complex_template(tmp_path):
    """Test a complex template with multiple jinja2 features."""

    mod_dir = tmp_path / "test_complex"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """Welcome {{ user|upper }}!

{% if items|length > 0 %}
You have {{ items|length }} item(s):
{% for item in items %}
  {{ loop.index }}. {{ item.name }} ({{ item.value }})
{% endfor %}
{% else %}
You have no items.
{% endif %}

{% if admin %}
You are an administrator.
{% endif %}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {",
                "        'user': 'alice',",
                "        'items': [{'name': 'apple', 'value': 5}, {'name': 'banana', 'value': 3}],",
                "        'admin': True",
                "    })",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "ALICE" in result
    assert "2 item(s)" in result
    assert "apple" in result
    assert "banana" in result
    assert "administrator" in result


def test_prompt_manager_file_not_found(tmp_path):
    """Test error handling when prompt file doesn't exist."""

    mod_dir = tmp_path / "test_notfound"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    try:",
                "        mgr.render('nonexistent')",
                "        return 'FAIL'",
                "    except FileNotFoundError as e:",
                "        if 'nonexistent' in str(e):",
                "            return 'OK'",
                "        return 'FAIL'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "OK"


def test_prompt_manager_file_extension_preference(tmp_path):
    """Test that .prompt extension is preferred over no extension."""

    mod_dir = tmp_path / "test_ext"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    # Create both files - .prompt should be preferred
    (prompts_dir / "test.prompt").write_text("With extension", encoding="utf-8")
    (prompts_dir / "test").write_text("Without extension", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test')",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "With extension"


def test_prompt_manager_numeric_and_boolean_values(tmp_path):
    """Test rendering with numeric and boolean values."""

    mod_dir = tmp_path / "test_types"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """Number: {{ num }}
Boolean: {{ flag }}
Float: {{ fval }}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {'num': 42, 'flag': True, 'fval': 3.14})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "42" in result
    assert "True" in result
    assert "3.14" in result


def test_prompt_manager_list_and_dict_access(tmp_path):
    """Test accessing list and dict elements in templates."""

    mod_dir = tmp_path / "test_structures"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """First item: {{ items[0] }}
Name: {{ user.name }}
Age: {{ user.age }}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {",
                "        'items': ['first', 'second'],",
                "        'user': {'name': 'Alice', 'age': 30}",
                "    })",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "first" in result
    assert "Alice" in result
    assert "30" in result


def test_prompt_manager_backward_compatibility_simple_vars(tmp_path):
    """Test backward compatibility with simple {{ var }} syntax."""

    mod_dir = tmp_path / "test_backward"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    # Simple template without jinja2 features
    (prompts_dir / "test.prompt").write_text(
        "Hello {{name}}, you have {{count}} items.", encoding="utf-8"
    )

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    result = mgr.render('test', {'name': 'Bob', 'count': 5})",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert result == "Hello Bob, you have 5 items."


def test_prompt_manager_multiple_renders_different_vars(tmp_path):
    """Test multiple renders of same template with different variables."""

    mod_dir = tmp_path / "test_multi"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    (prompts_dir / "test.prompt").write_text("Value: {{value}}", encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    results = []",
                "    for i in range(5):",
                "        results.append(mgr.render('test', {'value': i}))",
                "    return results",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    results = mod.run()
    assert len(results) == 5
    assert results[0] == "Value: 0"
    assert results[4] == "Value: 4"


def test_prompt_manager_conditional_with_extra_vars(tmp_path):
    """Test that conditionals work correctly with extra unused variables."""

    mod_dir = tmp_path / "test_conditional_extra"
    prompts_dir = mod_dir / "prompts"
    prompts_dir.mkdir(parents=True)

    template = """{% if use_vision %}
Vision enabled
{% else %}
Vision disabled
{% endif %}"""

    (prompts_dir / "test.prompt").write_text(template, encoding="utf-8")

    module_path = mod_dir / "test.py"
    module_path.write_text(
        "\n".join(
            [
                "from arcagi3.prompts import PromptManager",
                "",
                "def run():",
                "    mgr = PromptManager()",
                "    # Pass extra variables that aren't used",
                "    result = mgr.render('test', {",
                "        'use_vision': True,",
                "        'unused1': 'ignored',",
                "        'unused2': 999",
                "    })",
                "    return result",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mod = _load_module_from_path(module_path)
    result = mod.run()
    assert "Vision enabled" in result
    assert "ignored" not in result
    assert "999" not in result
