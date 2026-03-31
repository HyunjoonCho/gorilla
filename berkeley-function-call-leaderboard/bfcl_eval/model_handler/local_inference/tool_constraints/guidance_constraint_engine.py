import guidance
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


NO_TOOL_SELECTION = "DONE"
TERMINAL_ANSWER_SELECTOR_HINT = (
    f"If the assistant should stop calling tools and answer the user directly now, choose {NO_TOOL_SELECTION}."
)
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
    mode: str = "guidance"
    repair_attempts: int = 2
    max_calls_per_step: int = 1
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
        stop: Optional[str] = None,
    ) -> str:
        guidance = self._ensure_loaded()
        kwargs = {"name": key, "max_tokens": max_tokens}
        if regex is not None:
            kwargs["regex"] = regex
        if stop is not None:
            kwargs["stop"] = stop
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
        allow_answer: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        tool_map = {tool.get("name"): tool for tool in tools if tool.get("name")}
        if not tool_map:
            if allow_answer:
                return self._generate_terminal_answer(formatted_prompt, max_new_tokens)
            return "[]", {
                "constraint_engine": self.config.mode,
                "selected_tools": [],
                "selected_tool_count": 0,
                "terminated_with_answer": False,
            }

        if self.config.mode == "guidance_tool_only":
            name = self._select(
                self._prompt(formatted_prompt, [], allow_answer_hint=allow_answer),
                [*tool_map, *([NO_TOOL_SELECTION] if allow_answer else [])],
                "tool_name",
            )
            if name == NO_TOOL_SELECTION:
                return self._generate_terminal_answer(formatted_prompt, max_new_tokens)
            prefix = f"{name}("
            tail = self._runtime_adapter.gen(
                formatted_prompt.rstrip() + prefix,
                key="tool_tail",
                max_tokens=max_new_tokens,
            )
            return prefix + tail, {
                "constraint_engine": "guidance_tool_only",
                "selected_tools": [name],
                "selected_tool_count": 1,
                "terminated_with_answer": False,
            }

        selected_calls: list[tuple[str, dict[str, Any]]] = []
        options = [*tool_map, *([NO_TOOL_SELECTION] if allow_answer else [])]
        for _ in range(self.config.max_calls_per_step):
            name = self._select(
                self._prompt(
                    formatted_prompt,
                    selected_calls,
                    allow_answer_hint=allow_answer,
                ),
                options,
                "tool_name",
            )
            if name == NO_TOOL_SELECTION:
                if not selected_calls:
                    return self._generate_terminal_answer(
                        formatted_prompt, max_new_tokens
                    )
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
            "terminated_with_answer": False,
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
                f"{tool_name}(",
                "call",
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
        prefix: str,
        container: str,
        depth: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if depth >= self.config.max_json_depth:
            return {}

        generated = {}
        required = set(schema.get("required", []))
        current_prefix = prefix
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
            field_prefix = self._append_mapping_prefix(
                current_prefix,
                field_name,
                container,
            )
            value = self._generate_value(
                formatted_prompt,
                tool_name,
                field_name,
                field_schema,
                selected_calls,
                field_prefix,
                depth,
                max_new_tokens,
            )
            generated[field_name] = value
            current_prefix = field_prefix + self._python_repr(value)
        return generated

    def _generate_value(
        self,
        formatted_prompt: str,
        tool_name: str,
        field_name: str,
        field_schema: dict,
        selected_calls: list[tuple[str, dict[str, Any]]],
        prefix: str,
        depth: int,
        max_new_tokens: int,
    ) -> Any:
        enum_values = field_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            options = [json.dumps(item, ensure_ascii=False) for item in enum_values]
            chosen = self._select(
                self._prompt(formatted_prompt, selected_calls, prefix),
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
                self._prompt(formatted_prompt, selected_calls, prefix),
                ["True", "False"],
                f"bool_{field_name}",
            ) == "True"

        if schema_type == "string":
            value = self._runtime_adapter.gen(
                self._prompt(formatted_prompt, selected_calls, prefix + "'"),
                key=f"string_{field_name}",
                max_tokens=min(64, max_new_tokens),
                stop="'",
            )
            return self._decode_generated_value(value, field_schema)

        if schema_type in {"integer", "float"}:
            type_hint = "number" if schema_type == "float" else schema_type
            regex = {"integer": INTEGER_PATTERN, "float": NUMBER_PATTERN}[schema_type]
            value = self._runtime_adapter.gen(
                self._prompt(formatted_prompt, selected_calls, prefix),
                key=f"{type_hint}_{field_name}",
                max_tokens=min({"integer": 16, "float": 20}[schema_type], max_new_tokens),
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
            items = []
            current_prefix = prefix + "["
            for index in range(count):
                item_prefix = self._append_array_prefix(current_prefix)
                item = self._generate_value(
                    formatted_prompt,
                    tool_name,
                    f"{field_name}_{index}",
                    item_schema,
                    selected_calls,
                    item_prefix,
                    depth + 1,
                    max_new_tokens,
                )
                items.append(item)
                current_prefix = item_prefix + self._python_repr(item)
            return items

        if schema_type in {"dict", "object"}:
            return {} if depth >= self.config.max_json_depth else self._generate_object(
                formatted_prompt,
                tool_name,
                self._object_schema(field_schema),
                selected_calls,
                prefix + "{",
                "object",
                depth + 1,
                max_new_tokens,
            )

        value = self._runtime_adapter.gen(
            self._prompt(formatted_prompt, selected_calls, prefix + "'"),
            key=f"str_{field_name}",
            max_tokens=min(64, max_new_tokens),
            stop="'",
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
        allow_answer_hint: bool = False,
    ) -> str:
        suffix = []
        if selected_calls:
            suffix.append(f"Previously selected calls: {self._render_calls(selected_calls)}")
        suffix.extend(lines)
        if allow_answer_hint:
            suffix.append(TERMINAL_ANSWER_SELECTOR_HINT)
        return formatted_prompt.rstrip() + "\n\n" + "\n".join(suffix) + "\n"

    def _generate_terminal_answer(
        self,
        formatted_prompt: str,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        answer = self._runtime_adapter.gen(
            formatted_prompt.rstrip(),
            key="terminal_answer",
            max_tokens=max_new_tokens,
        )
        return answer, {
            "constraint_engine": self.config.mode,
            "selected_tools": [],
            "selected_tool_count": 0,
            "terminated_with_answer": True,
        }

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
            stripped = value.strip()
            if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
                stripped = stripped[1:-1]
            return stripped
        return value

    def _append_mapping_prefix(self, prefix: str, field_name: str, container: str) -> str:
        separator = "" if prefix.endswith(("(", "{")) else ", "
        if container == "object":
            return f"{prefix}{separator}{self._python_repr(field_name)}: "
        return f"{prefix}{separator}{field_name}="

    def _append_array_prefix(self, prefix: str) -> str:
        return prefix if prefix.endswith("[") else f"{prefix}, "

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
