import guidance
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


NO_TOOL_SELECTION = "__bfcl_final_answer__"
INTEGER_PATTERN = r"-?\d+"
NUMBER_PATTERN = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
TYPE_MAP = {
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


class GuidanceConstraintError(RuntimeError):
    pass


class GuidanceGenerationError(GuidanceConstraintError):
    pass


@dataclass
class GuidanceConstraintConfig:
    repair_attempts: int = 2
    max_calls_per_step: int = 4
    max_json_depth: int = 3


class PinnedGuidanceRuntime:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._guidance = None
        self._lm = None

    def select(self, prompt: str, options: list[str], key: str) -> str:
        if not options:
            raise GuidanceGenerationError("Cannot run select with empty options.")
        guidance = self._ensure_loaded()
        try:
            lm = self._lm + prompt
            lm += guidance.select(options, name=key)
            return str(lm[key]).strip()
        except Exception as exc:
            raise GuidanceGenerationError(f"Guidance select failed for '{key}': {exc}") from exc

    def gen(
        self,
        prompt: str,
        key: str,
        max_tokens: int,
        regex: Optional[str] = None,
    ) -> str:
        guidance = self._ensure_loaded()
        kwargs = {"name": key, "max_tokens": max_tokens}
        if regex is not None:
            kwargs["regex"] = regex
        try:
            lm = self._lm + prompt
            lm += guidance.gen(**kwargs)
            return str(lm[key]).strip()
        except Exception as exc:
            raise GuidanceGenerationError(f"Guidance generation failed for '{key}': {exc}") from exc

    def _ensure_loaded(self):
        if self._guidance is not None:
            return self._guidance

        try:
            self._lm = guidance.models.Transformers(
                model=self.model,
                tokenizer=self.tokenizer,
                echo=False,
            )
        except Exception as exc:
            raise GuidanceGenerationError(
                "Failed to initialize guidance.models.Transformers."
            ) from exc
        self._guidance = guidance
        return guidance


class GuidanceConstraintEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: GuidanceConstraintConfig,
        runtime: Optional[PinnedGuidanceRuntime] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._runtime = runtime

    def generate(
        self,
        formatted_prompt: str,
        tools: list[dict],
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        tool_map = {tool.get("name"): tool for tool in tools if tool.get("name")}
        if not tool_map:
            return "[]", {"constraint_engine": "guidance", "selected_tools": []}

        selected_calls: list[tuple[str, dict[str, Any]]] = []
        options = [NO_TOOL_SELECTION, *tool_map]
        for _ in range(self.config.max_calls_per_step):
            name = self._select(
                self._prompt(
                    formatted_prompt,
                    selected_calls,
                    "Choose the next tool name or the final-answer sentinel.",
                    f"Options: {json.dumps(options, ensure_ascii=False)}",
                ),
                options,
                "tool_name",
            )
            if name == NO_TOOL_SELECTION:
                break
            selected_calls.append(
                (
                    name,
                    self._generate_arguments(
                        formatted_prompt,
                        name,
                        tool_map[name].get("parameters", {}),
                        selected_calls,
                        max_new_tokens,
                    ),
                )
            )

        return self._render_calls(selected_calls), {
            "constraint_engine": "guidance",
            "selected_tools": [name for name, _ in selected_calls],
            "selected_tool_count": len(selected_calls),
        }

    def _generate_arguments(
        self,
        formatted_prompt: str,
        tool_name: str,
        schema: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        schema = self._object_schema(schema)
        errors = []
        for _ in range(self.config.repair_attempts + 1):
            arguments = self._generate_object(
                formatted_prompt,
                tool_name,
                schema,
                selected_calls,
                0,
                max_new_tokens,
            )
            valid, errors = self.validate_arguments(arguments, schema)
            if valid:
                return arguments
        raise GuidanceGenerationError(
            f"Failed to validate constrained arguments for tool '{tool_name}': {errors}"
        )

    def validate_arguments(
        self,
        arguments: dict[str, Any],
        schema: dict,
    ) -> tuple[bool, list[dict[str, str]]]:
        errors: list[dict[str, str]] = []
        if self._type(schema.get("type", "dict")) not in {"dict", "object"}:
            return False, [self._error("schema_shape", "$", "Tool parameter schema root must be object/dict.")]
        if not isinstance(arguments, dict):
            return False, [self._error("type_mismatch", "$", "Tool arguments must be a dictionary.")]

        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in arguments:
                errors.append(self._error("missing_required", field, f"Missing required field '{field}'."))
        for field, value in arguments.items():
            if field not in properties:
                errors.append(self._error("unexpected_field", field, f"Unexpected field '{field}'."))
                continue
            self._validate(value, properties[field], field, 1, errors)
        return len(errors) == 0, errors

    def _validate(
        self,
        value: Any,
        schema: dict,
        path: str,
        depth: int,
        errors: list[dict[str, str]],
    ) -> None:
        if depth > self.config.max_json_depth:
            errors.append(
                self._error(
                    "depth_exceeded",
                    path,
                    f"JSON depth exceeded the configured max depth ({self.config.max_json_depth}).",
                )
            )
            return

        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            if value not in enum_values:
                errors.append(self._error("enum_mismatch", path, f"Value '{value}' is not in enum list."))
            return

        schema_type = self._type(schema.get("type"))
        if schema_type in {"dict", "object"}:
            if not isinstance(value, dict):
                errors.append(self._error("type_mismatch", path, f"Expected object/dict at '{path}'."))
                return
            properties = schema.get("properties", {})
            for field in schema.get("required", []):
                if field not in value:
                    nested = f"{path}.{field}"
                    errors.append(self._error("missing_required", nested, f"Missing required field '{nested}'."))
            for key, nested_value in value.items():
                nested = f"{path}.{key}"
                if key not in properties:
                    errors.append(self._error("unexpected_field", nested, f"Unexpected field '{nested}'."))
                    continue
                self._validate(nested_value, properties[key], nested, depth + 1, errors)
            return

        if schema_type == "array":
            if not isinstance(value, list):
                errors.append(self._error("type_mismatch", path, f"Expected array/list at '{path}'."))
                return
            item_schema = schema.get("items", {})
            for index, item in enumerate(value):
                self._validate(item, item_schema, f"{path}[{index}]", depth + 1, errors)
            return

        checks = {
            "boolean": lambda x: isinstance(x, bool),
            "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "float": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "string": lambda x: isinstance(x, str),
        }
        if schema_type in checks and not checks[schema_type](value):
            label = {
                "boolean": "boolean",
                "integer": "integer",
                "float": "numeric value",
                "string": "string",
            }[schema_type]
            errors.append(self._error("type_mismatch", path, f"Expected {label} at '{path}'."))

    def _generate_object(
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

        generated = {}
        required = set(schema.get("required", []))
        for field_name, field_schema in schema.get("properties", {}).items():
            if field_name not in required:
                include = self._select(
                    self._prompt(
                        formatted_prompt,
                        selected_calls,
                        f"Tool: {tool_name}",
                        f"Optional field: {field_name}",
                        "Answer yes or no.",
                    ),
                    ["yes", "no"],
                    f"include_{field_name}",
                )
                if include != "yes":
                    continue
            generated[field_name] = self._generate_value(
                formatted_prompt,
                tool_name,
                field_name,
                field_schema,
                selected_calls,
                depth,
                max_new_tokens,
            )
        return generated

    def _generate_value(
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
        if isinstance(enum_values, list) and enum_values:
            options = [json.dumps(item, ensure_ascii=False) for item in enum_values]
            chosen = self._select(
                self._prompt(
                    formatted_prompt,
                    selected_calls,
                    f"Tool: {tool_name}",
                    f"Field: {field_name}",
                    f"Allowed values: {json.dumps(options, ensure_ascii=False)}",
                    "Answer with one allowed value only.",
                ),
                options,
                f"enum_{field_name}",
            )
            try:
                return json.loads(chosen)
            except Exception:
                return next((item for item in enum_values if str(item) == chosen), enum_values[0])

        schema_type = self._type(field_schema.get("type"))
        if schema_type == "boolean":
            return self._select(
                self._prompt(
                    formatted_prompt,
                    selected_calls,
                    f"Tool: {tool_name}",
                    f"Field: {field_name}",
                    "Type: boolean",
                    "Answer with the value only.",
                ),
                ["true", "false"],
                f"bool_{field_name}",
            ) == "true"

        if schema_type in {"integer", "float", "string"}:
            type_hint = "number" if schema_type == "float" else schema_type
            regex = {"integer": INTEGER_PATTERN, "float": NUMBER_PATTERN}.get(schema_type)
            value = self._runtime_adapter.gen(
                self._prompt(
                    formatted_prompt,
                    selected_calls,
                    f"Tool: {tool_name}",
                    f"Field: {field_name}",
                    f"Type: {type_hint}",
                    "Answer with the value only.",
                ),
                key=f"{type_hint}_{field_name}",
                max_tokens=min({"integer": 16, "float": 20}.get(schema_type, 64), max_new_tokens),
                regex=regex,
            )
            return self._decode_generated_value(value, field_schema)

        if schema_type == "array":
            if depth >= self.config.max_json_depth:
                return []
            min_items = int(field_schema.get("minItems", 0) or 0)
            max_items = int(field_schema.get("maxItems", min_items + 2) or (min_items + 2))
            options = [str(i) for i in range(min_items, max(min_items, min(3, max_items)) + 1)]
            count = int(
                self._select(
                    self._prompt(
                        formatted_prompt,
                        selected_calls,
                        f"Tool: {tool_name}",
                        f"Field: {field_name}",
                        "Type: array_length",
                        "Answer with the value only.",
                    ),
                    options,
                    f"arr_count_{field_name}",
                )
            )
            item_schema = field_schema.get("items", {})
            return [
                self._generate_value(
                    formatted_prompt,
                    tool_name,
                    f"{field_name}_{index}",
                    item_schema,
                    selected_calls,
                    depth + 1,
                    max_new_tokens,
                )
                for index in range(count)
            ]

        if schema_type in {"dict", "object"}:
            return {} if depth >= self.config.max_json_depth else self._generate_object(
                formatted_prompt,
                tool_name,
                self._object_schema(field_schema),
                selected_calls,
                depth + 1,
                max_new_tokens,
            )

        value = self._runtime_adapter.gen(
            self._prompt(
                formatted_prompt,
                selected_calls,
                f"Tool: {tool_name}",
                f"Field: {field_name}",
                "Type: string",
                "Answer with the value only.",
            ),
            key=f"str_{field_name}",
            max_tokens=min(64, max_new_tokens),
            regex=None,
        )
        return self._decode_generated_value(value, field_schema)

    def _select(self, prompt: str, options: list[str], key: str) -> str:
        value = self._runtime_adapter.select(prompt, options, key)
        if value not in options and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if value not in options:
            for option in options:
                parsed = self._safe_json_load(option)
                if parsed == value:
                    value = option
                    break
        if value not in options:
            raise GuidanceGenerationError(f"Guidance select returned invalid option '{value}'.")
        return value

    def _prompt(
        self,
        formatted_prompt: str,
        selected_calls: list[tuple[str, dict[str, Any]]],
        *lines: str,
    ) -> str:
        suffix = []
        if selected_calls:
            suffix.append(f"Previously selected calls: {self._render_calls(selected_calls)}")
        suffix.extend(lines)
        return formatted_prompt.rstrip() + "\n\n" + "\n".join(suffix) + "\n"

    @property
    def _runtime_adapter(self) -> PinnedGuidanceRuntime:
        if self._runtime is None:
            self._runtime = PinnedGuidanceRuntime(self.model, self.tokenizer)
        return self._runtime

    def _decode_generated_value(self, value: Any, schema: dict) -> Any:
        schema_type = self._type(schema.get("type"))
        if schema_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if re.fullmatch(INTEGER_PATTERN, stripped):
                    return int(stripped)
            return value
        if schema_type == "float":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if re.fullmatch(NUMBER_PATTERN, stripped):
                    return float(stripped)
            return value
        if schema_type == "string" and isinstance(value, str):
            return value.strip()
        return value

    def _object_schema(self, schema: dict) -> dict:
        schema = dict(schema) if isinstance(schema, dict) else {}
        if self._type(schema.get("type", "dict")) not in {"dict", "object"}:
            schema["type"] = "dict"
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        return schema

    def _type(self, raw_type: Any) -> str:
        if isinstance(raw_type, list) and raw_type:
            raw_type = next((item for item in raw_type if str(item).lower() != "null"), raw_type[0])
        return TYPE_MAP.get(str(raw_type or "string").strip().lower(), "string")

    def _render_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> str:
        if not calls:
            return "[]"
        rendered = []
        for name, arguments in calls:
            args = ", ".join(f"{key}={self._python_repr(value)}" for key, value in arguments.items())
            rendered.append(f"{name}({args})")
        return ", ".join(rendered)

    def _python_repr(self, value: Any) -> str:
        if isinstance(value, dict):
            return "{" + ", ".join(f"{self._python_repr(k)}: {self._python_repr(v)}" for k, v in value.items()) + "}"
        if isinstance(value, list):
            return "[" + ", ".join(self._python_repr(item) for item in value) + "]"
        return repr(value)

    def _safe_json_load(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _error(self, kind: str, path: str, message: str) -> dict[str, str]:
        return {"kind": kind, "path": path, "message": message}
