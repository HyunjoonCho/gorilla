import importlib
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


NO_TOOL_SELECTION = "__bfcl_final_answer__"
INTEGER_PATTERN = r"-?\d+"
NUMBER_PATTERN = r"-?(?:\d+(?:\.\d+)?|\.\d+)"


class GuidanceConstraintError(RuntimeError):
    """Base error for Guidance-constrained generation."""


class GuidanceUnavailableError(GuidanceConstraintError):
    """Raised when Guidance is unavailable in the current environment."""


class GuidanceGenerationError(GuidanceConstraintError):
    """Raised when Guidance constraint generation fails."""


@dataclass
class GuidanceConstraintConfig:
    repair_attempts: int = 2
    max_calls_per_step: int = 4
    max_json_depth: int = 3


class GuidanceConstraintEngine:
    """
    Guidance-backed constrained tool generation for prompting models.

    The engine enforces tool-name selection with `guidance.select` and generates
    arguments with schema-aware strategies, followed by post-validation and repair.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: GuidanceConstraintConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._guidance_module = None
        self._guidance_model = None

    def generate(
        self,
        formatted_prompt: str,
        tools: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        if not tools:
            return "[]", {"constraint_engine": "guidance", "selected_tools": []}

        self._build_guidance_model()

        tool_map = {tool.get("name"): tool for tool in tools if tool.get("name")}
        if not tool_map:
            return "[]", {"constraint_engine": "guidance", "selected_tools": []}

        tool_names = list(tool_map.keys())
        selected_calls: list[tuple[str, dict[str, Any]]] = []

        for _ in range(self.config.max_calls_per_step):
            selected_name = self._select_tool_name(
                formatted_prompt=formatted_prompt,
                tool_names=tool_names,
                selected_calls=selected_calls,
            )
            if selected_name == NO_TOOL_SELECTION:
                break
            if selected_name not in tool_map:
                raise GuidanceGenerationError(
                    f"Guidance selected unknown tool '{selected_name}'."
                )

            tool_definition = tool_map[selected_name]
            arguments = self._generate_and_validate_arguments(
                formatted_prompt=formatted_prompt,
                tool_name=selected_name,
                tool_definition=tool_definition,
                selected_calls=selected_calls,
                max_new_tokens=max_new_tokens,
            )
            selected_calls.append((selected_name, arguments))

        output_text = self._render_python_tool_calls(selected_calls)
        metadata = {
            "constraint_engine": "guidance",
            "selected_tools": [name for name, _ in selected_calls],
            "selected_tool_count": len(selected_calls),
        }
        return output_text, metadata

    def _select_tool_name(
        self,
        formatted_prompt: str,
        tool_names: list[str],
        selected_calls: list[tuple[str, dict[str, Any]]],
    ) -> str:
        options = [NO_TOOL_SELECTION, *tool_names]
        prompt = (
            "You are selecting the next tool call for a function-calling task.\n"
            "Choose exactly one option from the allowed tool names.\n"
            f"Conversation context:\n{formatted_prompt}\n\n"
            f"Previously selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
            "If no more tool call is needed, choose the final-answer option.\n"
            "Output selection:\n"
        )
        return self._run_select(prompt=prompt, options=options, key="tool_name")

    def _generate_and_validate_arguments(
        self,
        formatted_prompt: str,
        tool_name: str,
        tool_definition: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        schema = self._normalize_object_schema(tool_definition.get("parameters", {}))
        arguments = self._generate_object_value(
            formatted_prompt=formatted_prompt,
            tool_name=tool_name,
            schema=schema,
            selected_calls=selected_calls,
            depth=0,
            max_new_tokens=max_new_tokens,
        )

        last_errors = []
        for _ in range(self.config.repair_attempts + 1):
            is_valid, errors = self.validate_arguments(arguments, schema)
            if is_valid:
                return arguments
            last_errors = errors
            arguments = self._repair_arguments(
                formatted_prompt=formatted_prompt,
                tool_name=tool_name,
                schema=schema,
                selected_calls=selected_calls,
                current_arguments=arguments,
                errors=errors,
                max_new_tokens=max_new_tokens,
            )

        raise GuidanceGenerationError(
            f"Failed to validate constrained arguments for tool '{tool_name}': {last_errors}"
        )

    def _repair_arguments(
        self,
        formatted_prompt: str,
        tool_name: str,
        schema: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        current_arguments: dict[str, Any],
        errors: list[dict[str, str]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        repaired = dict(current_arguments)
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        for error in errors:
            error_kind = error.get("kind", "")
            field = self._extract_root_field(error.get("path", ""))
            if not field:
                continue

            if error_kind == "unexpected_field":
                repaired.pop(field, None)
                continue

            field_schema = properties.get(field, {})
            if error_kind in {"missing_required", "type_mismatch", "enum_mismatch", "depth_exceeded"}:
                regenerated = self._generate_value_for_schema(
                    formatted_prompt=formatted_prompt,
                    tool_name=tool_name,
                    field_name=field,
                    field_schema=field_schema,
                    selected_calls=selected_calls,
                    depth=0,
                    max_new_tokens=max_new_tokens,
                )
                repaired[field] = regenerated

        for field in list(repaired.keys()):
            if field not in properties:
                continue
            repaired[field] = self._coerce_value_to_schema(
                repaired[field], properties[field], depth=2
            )

        for field in required_fields:
            if field in repaired:
                continue
            regenerated = self._generate_value_for_schema(
                formatted_prompt=formatted_prompt,
                tool_name=tool_name,
                field_name=field,
                field_schema=properties.get(field, {}),
                selected_calls=selected_calls,
                depth=0,
                max_new_tokens=max_new_tokens,
            )
            repaired[field] = regenerated

        return repaired

    def validate_arguments(
        self, arguments: dict[str, Any], schema: dict
    ) -> tuple[bool, list[dict[str, str]]]:
        errors: list[dict[str, str]] = []

        schema_type = self._normalize_type(schema.get("type", "dict"))
        if schema_type not in {"dict", "object"}:
            errors.append(
                {
                    "kind": "schema_shape",
                    "path": "$",
                    "message": "Tool parameter schema root must be object/dict.",
                }
            )
            return False, errors

        if not isinstance(arguments, dict):
            errors.append(
                {
                    "kind": "type_mismatch",
                    "path": "$",
                    "message": "Tool arguments must be a dictionary.",
                }
            )
            return False, errors

        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        for field in required_fields:
            if field not in arguments:
                errors.append(
                    {
                        "kind": "missing_required",
                        "path": field,
                        "message": f"Missing required field '{field}'.",
                    }
                )

        for field in arguments:
            if field not in properties:
                errors.append(
                    {
                        "kind": "unexpected_field",
                        "path": field,
                        "message": f"Unexpected field '{field}'.",
                    }
                )

        for field_name, field_value in arguments.items():
            if field_name not in properties:
                continue
            self._validate_value_against_schema(
                value=field_value,
                schema=properties[field_name],
                path=field_name,
                depth=1,
                errors=errors,
            )

        return len(errors) == 0, errors

    def _validate_value_against_schema(
        self,
        value: Any,
        schema: dict,
        path: str,
        depth: int,
        errors: list[dict[str, str]],
    ) -> None:
        if depth > self.config.max_json_depth:
            errors.append(
                {
                    "kind": "depth_exceeded",
                    "path": path,
                    "message": f"JSON depth exceeded the configured max depth ({self.config.max_json_depth}).",
                }
            )
            return

        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            if value not in enum_values:
                errors.append(
                    {
                        "kind": "enum_mismatch",
                        "path": path,
                        "message": f"Value '{value}' is not in enum list.",
                    }
                )
            return

        normalized_type = self._normalize_type(schema.get("type"))

        if normalized_type in {"dict", "object"}:
            if not isinstance(value, dict):
                errors.append(
                    {
                        "kind": "type_mismatch",
                        "path": path,
                        "message": f"Expected object/dict at '{path}'.",
                    }
                )
                return
            nested_properties = schema.get("properties", {})
            nested_required = set(schema.get("required", []))
            for field in nested_required:
                if field not in value:
                    nested_path = f"{path}.{field}"
                    errors.append(
                        {
                            "kind": "missing_required",
                            "path": nested_path,
                            "message": f"Missing required field '{nested_path}'.",
                        }
                    )
            for nested_key, nested_value in value.items():
                nested_path = f"{path}.{nested_key}"
                if nested_key not in nested_properties:
                    errors.append(
                        {
                            "kind": "unexpected_field",
                            "path": nested_path,
                            "message": f"Unexpected field '{nested_path}'.",
                        }
                    )
                    continue
                self._validate_value_against_schema(
                    value=nested_value,
                    schema=nested_properties[nested_key],
                    path=nested_path,
                    depth=depth + 1,
                    errors=errors,
                )
            return

        if normalized_type in {"array", "list"}:
            if not isinstance(value, list):
                errors.append(
                    {
                        "kind": "type_mismatch",
                        "path": path,
                        "message": f"Expected array/list at '{path}'.",
                    }
                )
                return
            item_schema = schema.get("items", {})
            for idx, item in enumerate(value):
                item_path = f"{path}[{idx}]"
                self._validate_value_against_schema(
                    value=item,
                    schema=item_schema,
                    path=item_path,
                    depth=depth + 1,
                    errors=errors,
                )
            return

        if normalized_type in {"boolean", "bool"}:
            if not isinstance(value, bool):
                errors.append(
                    {
                        "kind": "type_mismatch",
                        "path": path,
                        "message": f"Expected boolean at '{path}'.",
                    }
                )
            return

        if normalized_type in {"integer", "int"}:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(
                    {
                        "kind": "type_mismatch",
                        "path": path,
                        "message": f"Expected integer at '{path}'.",
                    }
                )
            return

        if normalized_type in {"float", "number", "double"}:
            if (not isinstance(value, (int, float))) or isinstance(value, bool):
                errors.append(
                    {
                        "kind": "type_mismatch",
                        "path": path,
                        "message": f"Expected numeric value at '{path}'.",
                    }
                )
            return

        if normalized_type in {"string", "str"} and not isinstance(value, str):
            errors.append(
                {
                    "kind": "type_mismatch",
                    "path": path,
                    "message": f"Expected string at '{path}'.",
                }
            )

    def _generate_object_value(
        self,
        formatted_prompt: str,
        tool_name: str,
        schema: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        depth: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if depth >= self.config.max_json_depth:
            return {}

        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))
        generated: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            include_field = field_name in required_fields
            if not include_field:
                include_field = (
                    self._run_select(
                        prompt=(
                            "Decide whether to include an optional tool argument field.\n"
                            f"Tool: {tool_name}\n"
                            f"Field: {field_name}\n"
                            f"Conversation context:\n{formatted_prompt}\n\n"
                            f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                            "Choose yes or no:\n"
                        ),
                        options=["yes", "no"],
                        key=f"include_{field_name}",
                    )
                    == "yes"
                )
            if not include_field:
                continue

            generated[field_name] = self._generate_value_for_schema(
                formatted_prompt=formatted_prompt,
                tool_name=tool_name,
                field_name=field_name,
                field_schema=field_schema,
                selected_calls=selected_calls,
                depth=depth,
                max_new_tokens=max_new_tokens,
            )

        return generated

    def _generate_value_for_schema(
        self,
        formatted_prompt: str,
        tool_name: str,
        field_name: str,
        field_schema: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        depth: int,
        max_new_tokens: int,
    ) -> Any:
        enum_values = field_schema.get("enum")
        if isinstance(enum_values, list) and len(enum_values) > 0:
            return self._run_enum_select(
                prompt=(
                    "Select a valid enum value for a tool argument.\n"
                    f"Tool: {tool_name}\n"
                    f"Field: {field_name}\n"
                    f"Conversation context:\n{formatted_prompt}\n\n"
                    f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                    "Choose one enum value:\n"
                ),
                enum_values=enum_values,
                key=f"enum_{field_name}",
            )

        normalized_type = self._normalize_type(field_schema.get("type"))

        if normalized_type in {"boolean", "bool"}:
            bool_text = self._run_select(
                prompt=(
                    "Select a boolean value for a tool argument.\n"
                    f"Tool: {tool_name}\n"
                    f"Field: {field_name}\n"
                    f"Conversation context:\n{formatted_prompt}\n\n"
                    f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                    "Choose true or false:\n"
                ),
                options=["true", "false"],
                key=f"bool_{field_name}",
            )
            return bool_text == "true"

        if normalized_type in {"integer", "int"}:
            generated_text = self._run_gen(
                prompt=(
                    "Generate an integer tool argument value.\n"
                    f"Tool: {tool_name}\n"
                    f"Field: {field_name}\n"
                    f"Conversation context:\n{formatted_prompt}\n\n"
                    f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                    "Output integer:\n"
                ),
                key=f"int_{field_name}",
                max_tokens=min(16, max_new_tokens),
                regex=INTEGER_PATTERN,
            )
            return self._coerce_value_to_schema(generated_text, field_schema, depth=depth)

        if normalized_type in {"float", "number", "double"}:
            generated_text = self._run_gen(
                prompt=(
                    "Generate a numeric tool argument value.\n"
                    f"Tool: {tool_name}\n"
                    f"Field: {field_name}\n"
                    f"Conversation context:\n{formatted_prompt}\n\n"
                    f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                    "Output numeric value:\n"
                ),
                key=f"num_{field_name}",
                max_tokens=min(20, max_new_tokens),
                regex=NUMBER_PATTERN,
            )
            return self._coerce_value_to_schema(generated_text, field_schema, depth=depth)

        if normalized_type in {"array", "list"}:
            if depth >= self.config.max_json_depth:
                return []
            min_items = int(field_schema.get("minItems", 0) or 0)
            max_items = int(field_schema.get("maxItems", min_items + 2) or (min_items + 2))
            bounded_max = max(min_items, min(3, max_items))
            count_options = [str(i) for i in range(min_items, bounded_max + 1)]
            item_count = int(
                self._run_select(
                    prompt=(
                        "Select how many array elements to generate for a tool argument.\n"
                        f"Tool: {tool_name}\n"
                        f"Field: {field_name}\n"
                        f"Conversation context:\n{formatted_prompt}\n"
                        "Choose array length:\n"
                    ),
                    options=count_options,
                    key=f"arr_count_{field_name}",
                )
            )
            item_schema = field_schema.get("items", {})
            values = []
            for idx in range(item_count):
                item_field_name = f"{field_name}_{idx}"
                values.append(
                    self._generate_value_for_schema(
                        formatted_prompt=formatted_prompt,
                        tool_name=tool_name,
                        field_name=item_field_name,
                        field_schema=item_schema,
                        selected_calls=selected_calls,
                        depth=depth + 1,
                        max_new_tokens=max_new_tokens,
                    )
                )
            return values

        if normalized_type in {"dict", "object"}:
            if depth >= self.config.max_json_depth:
                return {}
            nested_schema = self._normalize_object_schema(field_schema)
            return self._generate_object_value(
                formatted_prompt=formatted_prompt,
                tool_name=tool_name,
                schema=nested_schema,
                selected_calls=selected_calls,
                depth=depth + 1,
                max_new_tokens=max_new_tokens,
            )

        # Free-form string fallback.
        generated_text = self._run_gen(
            prompt=(
                "Generate a free-form string tool argument value.\n"
                f"Tool: {tool_name}\n"
                f"Field: {field_name}\n"
                f"Conversation context:\n{formatted_prompt}\n\n"
                f"Existing selected tool calls: {self._render_python_tool_calls(selected_calls)}\n"
                "Output string value:\n"
            ),
            key=f"str_{field_name}",
            max_tokens=min(64, max_new_tokens),
            regex=None,
        )
        return self._coerce_value_to_schema(generated_text, field_schema, depth=depth)

    def _coerce_value_to_schema(self, value: Any, schema: dict, depth: int) -> Any:
        if depth > self.config.max_json_depth:
            return self._default_for_schema(schema)

        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            if value in enum_values:
                return value
            if isinstance(value, str):
                for enum_value in enum_values:
                    if str(enum_value) == value:
                        return enum_value
            return enum_values[0] if enum_values else value

        normalized_type = self._normalize_type(schema.get("type"))

        if normalized_type in {"boolean", "bool"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return bool(value)

        if normalized_type in {"integer", "int"}:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                match = re.search(INTEGER_PATTERN, value.strip())
                if match:
                    return int(match.group(0))
            default_value = schema.get("default")
            if isinstance(default_value, int):
                return default_value
            return 0

        if normalized_type in {"float", "number", "double"}:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                match = re.search(NUMBER_PATTERN, value.strip())
                if match:
                    return float(match.group(0))
            default_value = schema.get("default")
            if isinstance(default_value, (int, float)):
                return float(default_value)
            return 0.0

        if normalized_type in {"array", "list"}:
            if isinstance(value, list):
                item_schema = schema.get("items", {})
                return [
                    self._coerce_value_to_schema(item, item_schema, depth=depth + 1)
                    for item in value
                ]
            if isinstance(value, str):
                parsed = self._safe_json_load(value)
                if isinstance(parsed, list):
                    item_schema = schema.get("items", {})
                    return [
                        self._coerce_value_to_schema(item, item_schema, depth=depth + 1)
                        for item in parsed
                    ]
            return []

        if normalized_type in {"dict", "object"}:
            object_schema = self._normalize_object_schema(schema)
            if isinstance(value, str):
                parsed = self._safe_json_load(value)
                if isinstance(parsed, dict):
                    value = parsed
            if isinstance(value, dict):
                properties = object_schema.get("properties", {})
                coerced = {}
                for field_name, field_schema in properties.items():
                    if field_name not in value:
                        continue
                    coerced[field_name] = self._coerce_value_to_schema(
                        value[field_name], field_schema, depth=depth + 1
                    )
                return coerced
            return {}

        if value is None:
            default_value = schema.get("default")
            if default_value is not None:
                return default_value
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value)

    def _normalize_object_schema(self, schema: dict) -> dict:
        if not isinstance(schema, dict):
            return {"type": "dict", "properties": {}, "required": []}

        normalized = dict(schema)
        normalized_type = self._normalize_type(normalized.get("type", "dict"))
        if normalized_type not in {"dict", "object"}:
            normalized["type"] = "dict"
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
        return normalized

    def _default_for_schema(self, schema: dict) -> Any:
        if isinstance(schema, dict) and "default" in schema:
            return schema["default"]

        normalized_type = self._normalize_type(
            schema.get("type") if isinstance(schema, dict) else None
        )
        if normalized_type in {"dict", "object"}:
            return {}
        if normalized_type in {"array", "list"}:
            return []
        if normalized_type in {"boolean", "bool"}:
            return False
        if normalized_type in {"integer", "int"}:
            return 0
        if normalized_type in {"float", "number", "double"}:
            return 0.0
        return ""

    def _normalize_type(self, raw_type: Any) -> str:
        if isinstance(raw_type, list) and raw_type:
            # JSON-schema style type unions: pick a non-null type if available.
            filtered = [item for item in raw_type if str(item).lower() != "null"]
            if filtered:
                raw_type = filtered[0]
            else:
                raw_type = raw_type[0]
        raw = str(raw_type or "string").strip().lower()
        mapping = {
            "dict": "dict",
            "dictionary": "dict",
            "object": "object",
            "map": "dict",
            "array": "array",
            "list": "array",
            "tuple": "array",
            "str": "string",
            "string": "string",
            "any": "string",
            "int": "integer",
            "integer": "integer",
            "long": "integer",
            "float": "float",
            "double": "float",
            "number": "float",
            "decimal": "float",
            "bool": "boolean",
            "boolean": "boolean",
        }
        return mapping.get(raw, "string")

    def _extract_root_field(self, path: str) -> str:
        if not path or path == "$":
            return ""
        cleaned = path.replace("$.", "")
        cleaned = cleaned.split(".", 1)[0]
        cleaned = cleaned.split("[", 1)[0]
        return cleaned

    def _render_python_tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> str:
        if not calls:
            return "[]"
        rendered_calls = []
        for name, arguments in calls:
            arguments_text = ", ".join(
                f"{arg_name}={self._python_repr(arg_value)}"
                for arg_name, arg_value in arguments.items()
            )
            rendered_calls.append(f"{name}({arguments_text})")
        return ", ".join(rendered_calls)

    def _python_repr(self, value: Any) -> str:
        if isinstance(value, dict):
            inner = ", ".join(
                f"{self._python_repr(key)}: {self._python_repr(val)}"
                for key, val in value.items()
            )
            return "{" + inner + "}"
        if isinstance(value, list):
            inner = ", ".join(self._python_repr(item) for item in value)
            return "[" + inner + "]"
        return repr(value)

    def _run_enum_select(
        self,
        prompt: str,
        enum_values: list[Any],
        key: str,
    ) -> Any:
        encoded_options = [json.dumps(item, ensure_ascii=False) for item in enum_values]
        chosen = self._run_select(prompt=prompt, options=encoded_options, key=key)
        try:
            return json.loads(chosen)
        except Exception:
            for enum_value in enum_values:
                if str(enum_value) == chosen:
                    return enum_value
        return enum_values[0]

    def _run_select(self, prompt: str, options: list[str], key: str) -> str:
        if not options:
            raise GuidanceGenerationError("Cannot run select with empty options.")
        selection_op = self._build_select_op(options=options, key=key)
        value = self._run_guidance_op(prompt=prompt, op=selection_op, key=key)
        value = str(value).strip()
        if value not in options:
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if value not in options:
                raise GuidanceGenerationError(
                    f"Guidance select returned invalid option '{value}'."
                )
        return value

    def _run_gen(
        self,
        prompt: str,
        key: str,
        max_tokens: int,
        regex: Optional[str],
    ) -> str:
        gen_op = self._build_gen_op(key=key, max_tokens=max_tokens, regex=regex)
        value = self._run_guidance_op(prompt=prompt, op=gen_op, key=key)
        return str(value).strip()

    def _run_guidance_op(self, prompt: str, op: Any, key: str) -> Any:
        if self._guidance_model is None:
            raise GuidanceGenerationError("Guidance model is not initialized.")
        try:
            lm = self._guidance_model + prompt
            lm += op
            return lm[key]
        except Exception as exc:
            raise GuidanceGenerationError(
                f"Guidance operation failed for key '{key}': {exc}"
            ) from exc

    def _build_select_op(self, options: list[str], key: str) -> Any:
        guidance_module = self._import_guidance_module()
        select_fn = getattr(guidance_module, "select", None)
        if select_fn is None:
            raise GuidanceGenerationError("`guidance.select` is unavailable.")

        candidates = (
            {"options": options, "name": key},
            {"name": key, "options": options},
            {"name": key, "choices": options},
        )
        for kwargs in candidates:
            try:
                return select_fn(**kwargs)
            except TypeError:
                continue
        try:
            return select_fn(options, name=key)
        except TypeError as exc:
            raise GuidanceGenerationError(
                "Unable to construct guidance select op with the installed Guidance version."
            ) from exc

    def _build_gen_op(self, key: str, max_tokens: int, regex: Optional[str]) -> Any:
        guidance_module = self._import_guidance_module()
        gen_fn = getattr(guidance_module, "gen", None)
        if gen_fn is None:
            raise GuidanceGenerationError("`guidance.gen` is unavailable.")

        kwargs = {"name": key, "max_tokens": max_tokens}
        if regex is not None:
            for regex_key in ("regex", "pattern"):
                try:
                    return gen_fn(**{**kwargs, regex_key: regex})
                except TypeError:
                    continue
        try:
            return gen_fn(**kwargs)
        except TypeError as exc:
            raise GuidanceGenerationError(
                "Unable to construct guidance gen op with the installed Guidance version."
            ) from exc

    def _build_guidance_model(self) -> Any:
        if self._guidance_model is not None:
            return self._guidance_model

        guidance_module = self._import_guidance_module()
        models_module = getattr(guidance_module, "models", None)
        if models_module is None or not hasattr(models_module, "Transformers"):
            raise GuidanceGenerationError(
                "Installed Guidance package does not expose guidance.models.Transformers."
            )

        transformers_model_cls = getattr(models_module, "Transformers")

        constructor_candidates = [
            {"model": self.model, "tokenizer": self.tokenizer, "echo": False},
            {"model": self.model, "tokenizer": self.tokenizer},
            {"model": self.model, "tokenizer": self.tokenizer, "silent": True},
            {"model": self.model},
        ]
        for kwargs in constructor_candidates:
            try:
                self._guidance_model = transformers_model_cls(**kwargs)
                return self._guidance_model
            except TypeError:
                continue
            except Exception as exc:
                raise GuidanceGenerationError(
                    f"Failed to initialize Guidance Transformers model: {exc}"
                ) from exc

        positional_candidates = [
            (self.model, self.tokenizer),
            (self.model,),
        ]
        for args in positional_candidates:
            try:
                self._guidance_model = transformers_model_cls(*args)
                return self._guidance_model
            except TypeError:
                continue
            except Exception as exc:
                raise GuidanceGenerationError(
                    f"Failed to initialize Guidance Transformers model: {exc}"
                ) from exc

        raise GuidanceGenerationError(
            "Unable to construct Guidance Transformers model with known signatures."
        )

    def _import_guidance_module(self):
        if self._guidance_module is not None:
            return self._guidance_module
        try:
            self._guidance_module = importlib.import_module("guidance")
            return self._guidance_module
        except Exception as exc:
            raise GuidanceUnavailableError(
                "Guidance is not installed. Install BFCL with Guidance extras to enable constrained generation."
            ) from exc

    def _safe_json_load(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None
