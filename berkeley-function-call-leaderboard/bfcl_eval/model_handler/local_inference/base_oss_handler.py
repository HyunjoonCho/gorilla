import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import requests
from bfcl_eval.constants.backends import SUPPORTED_LOCAL_BACKENDS
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction
from openai import OpenAI
from overrides import EnforceOverrides, final, override
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class _OpenAICompatCompletionResponse:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.choices = [SimpleNamespace(text=text)]
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OSSHandler(BaseHandler, EnforceOverrides):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name
        self.model_style = ModelStyle.OSSMODEL
        self.dtype = dtype

        # Will be overridden in batch_inference method
        # Used to indicate where the tokenizer and config should be loaded from
        self.model_path_or_id = None

        # Read from env vars with fallbacks
        self.local_server_endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
        self.local_server_port = os.getenv("LOCAL_SERVER_PORT", LOCAL_SERVER_PORT)

        # Support custom base_url and api_key for remote/local OpenAI-compatible deployments (e.g., vLLM)
        # Use REMOTE_OPENAI_* variables to avoid conflicts with main OPENAI_* variables
        self.base_url = os.getenv("REMOTE_OPENAI_BASE_URL", f"http://{self.local_server_endpoint}:{self.local_server_port}/v1")
        self.api_key = os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.backend = None
        self.model = None
        self._generation_lock = None

    def _resolve_torch_dtype(self, torch_module):
        dtype_mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "half": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
        }
        normalized_dtype = str(self.dtype).lower()
        if normalized_dtype not in dtype_mapping:
            raise ValueError(
                f"Unsupported dtype '{self.dtype}' for transformers backend. Supported values: {', '.join(dtype_mapping.keys())}."
            )
        return dtype_mapping[normalized_dtype]

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # TODO: Let oss model support FC methods as well, depends on their model type
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        return default_decode_execute_prompting(result, has_tool_call_tag)

    @final
    def spin_up_local_server(
        self,
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
        local_model_path: Optional[str],
        lora_modules: Optional[list[str]] = None,
        enable_lora: bool = False,
        max_lora_rank: Optional[int] = None,
    ):
        """
        Spin up a local server for the model.
        If the server is already running, skip the setup.
        """
        if backend not in SUPPORTED_LOCAL_BACKENDS:
            raise ValueError(
                f"Backend '{backend}' is not supported. Supported backends: {', '.join(SUPPORTED_LOCAL_BACKENDS)}."
            )
        self.backend = backend

        # Determine the model source
        if local_model_path is not None:
            # Validate the local_model_path
            if not os.path.isdir(local_model_path):
                raise ValueError(
                    f"local_model_path '{local_model_path}' does not exist or is not a directory."
                )

            required_files = ["config.json", "tokenizer_config.json"]
            for file_name in required_files:
                if not os.path.exists(os.path.join(local_model_path, file_name)):
                    raise ValueError(
                        f"Required file '{file_name}' not found in local_model_path '{local_model_path}'."
                    )

            self.model_path_or_id = local_model_path
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "local_files_only": True,
                "trust_remote_code": True,
            }
        else:
            self.model_path_or_id = self.model_name_huggingface
            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path_or_id,
                "trust_remote_code": True,
            }

        # For remote OpenAI-compatible endpoints, use specified tokenizer path if provided
        is_remote_endpoint = bool(os.getenv("REMOTE_OPENAI_BASE_URL"))
        tokenizer_path = os.getenv("REMOTE_OPENAI_TOKENIZER_PATH", self.model_path_or_id)

        if is_remote_endpoint and os.getenv("REMOTE_OPENAI_TOKENIZER_PATH"):
            # Use specified tokenizer for remote endpoints
            tokenizer_kwargs = {
                "pretrained_model_name_or_path": tokenizer_path,
                "trust_remote_code": True,
            }
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
                config = AutoConfig.from_pretrained(**tokenizer_kwargs)
                print(f"Loaded tokenizer from REMOTE_OPENAI_TOKENIZER_PATH: {tokenizer_path}")
            except Exception as e:
                print(f"Failed to load tokenizer from {tokenizer_path}, falling back to model path: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
                config = AutoConfig.from_pretrained(**load_kwargs)
        else:
            # Standard loading for local models or when no specific tokenizer path is provided
            self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
            config = AutoConfig.from_pretrained(**load_kwargs)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif self.tokenizer.model_max_length is not None:
            self.max_context_length = self.tokenizer.model_max_length
        else:
            if not hasattr(self, "max_context_length"):
                raise ValueError(
                    "Model does not have a max_position_embeddings attribute or tokenizer.model_max_length attribute. Please set the max_context_length attribute in the corresponding model handler."
                )
        print(f"Max context length: {self.max_context_length}")

        self._server_process = process = None
        self._stdout_thread = stdout_thread = None
        self._stderr_thread = stderr_thread = None
        # Event to signal threads to stop; no need to see logs after server is ready
        # declare early so it always exists
        self._stop_event = threading.Event()

        if backend == "transformers":
            if skip_server_setup:
                print(
                    "`--skip-server-setup` is ignored for backend=transformers. Running direct in-process inference."
                )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path_or_id),
                    trust_remote_code=True,
                    torch_dtype=self._resolve_torch_dtype(torch),
                    device_map="auto",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{self.model_path_or_id}' with AutoModelForCausalLM for backend=transformers. "
                    "This model may require a server backend (`vllm`/`sglang`) or a custom handler."
                ) from e

            self.model.eval()
            self._generation_lock = threading.Lock()
            return

        try:
            if not skip_server_setup:
                if backend == "vllm":
                    process = subprocess.Popen(
                        [
                            "vllm",
                            "serve",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tensor-parallel-size",
                            str(num_gpus),
                            "--gpu-memory-utilization",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ]
                        + (["--enable-lora"] if enable_lora else [])
                        + (
                            ["--max-lora-rank", str(max_lora_rank)]
                            if max_lora_rank is not None
                            else []
                        )
                        + (
                            sum(
                                [["--lora-modules", lora_module] for lora_module in lora_modules],
                                [],
                            )
                            if lora_modules
                            else []
                        ),
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                elif backend == "sglang":

                    process = subprocess.Popen(
                        [
                            "python",
                            "-m",
                            "sglang.launch_server",
                            "--model-path",
                            str(self.model_path_or_id),
                            "--port",
                            str(self.local_server_port),
                            "--dtype",
                            str(self.dtype),
                            "--tp",
                            str(num_gpus),
                            "--mem-fraction-static",
                            str(gpu_memory_utilization),
                            "--trust-remote-code",
                        ],
                        stdout=subprocess.PIPE,  # Capture stdout
                        stderr=subprocess.PIPE,  # Capture stderr
                        text=True,  # To get the output as text instead of bytes
                    )
                def log_subprocess_output(pipe, stop_event):
                    # Read lines until the pipe is closed (EOF)
                    for line in iter(pipe.readline, ""):
                        if not stop_event.is_set():
                            print(line, end="")
                    print("server log tracking thread stopped successfully.")

                # Start threads to read and print stdout and stderr
                stdout_thread = threading.Thread(
                    target=log_subprocess_output, args=(process.stdout, self._stop_event)
                )
                stderr_thread = threading.Thread(
                    target=log_subprocess_output, args=(process.stderr, self._stop_event)
                )
                stdout_thread.setDaemon(True)
                stderr_thread.setDaemon(True)
                stdout_thread.start()
                stderr_thread.start()

            self._server_process = process
            self._stdout_thread = stdout_thread
            self._stderr_thread = stderr_thread

            # Wait for the server to be ready
            server_ready = False
            while not server_ready:
                # Check if the process has terminated unexpectedly
                if not skip_server_setup and process.poll() is not None:
                    # Output the captured logs
                    stdout, stderr = process.communicate()
                    print(stdout)
                    print(stderr)
                    raise Exception(
                        f"Subprocess terminated unexpectedly with code {process.returncode}"
                    )
                try:
                    # Make a simple request to check if the server is up
                    response = requests.get(f"{self.base_url}/models")
                    if response.status_code == 200:
                        server_ready = True
                        print("server is ready!")
                except requests.exceptions.ConnectionError:
                    # If the connection is not ready, wait and try again
                    time.sleep(1)

            # Signal threads to stop reading output
            self._stop_event.set()

        except Exception as e:
            # Clean-up everything we already started, then re-raise
            if self._server_process and self._server_process.poll() is None:
                self._server_process.terminate()
            if self._stop_event:
                self._stop_event.set()
            if self._stdout_thread:
                self._stdout_thread.join(timeout=2)
            if self._stderr_thread:
                self._stderr_thread.join(timeout=2)
            raise e

    def shutdown_local_server(self):
        """Terminate the locally launched OSS model server if it is still running."""
        # Ensure the server process is terminated properly
        process = getattr(self, "_server_process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                # Wait for the process to terminate fully
                process.wait(timeout=15)
                print("Process terminated successfully.")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()  # Wait again to ensure it's fully terminated
                print("Process killed.")

        # Tell the log-reader threads to stop and wait for them
        if getattr(self, "_stop_event", None):
            self._stop_event.set()
        if getattr(self, "_stdout_thread", None):
            self._stdout_thread.join(timeout=2)
        if getattr(self, "_stderr_thread", None):
            self._stderr_thread.join(timeout=2)

    #### Prompting methods ####

    def _format_prompt(self, messages, function):
        """
        Manually apply the chat template to construct the formatted prompt.
        This way, we can have full control over the final formatted prompt and is generally recommended for advanced use cases.
        """
        raise NotImplementedError(
            "OSS Models should implement their own prompt formatting."
        )

    @override
    def _query_prompting(self, inference_data: dict):
        # We use the OpenAI Completions API for server backends and direct generation for transformers.
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # Tokenize the formatted prompt to get token count
        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))

        # Determine the number of tokens to request. Cap it at 4096 if the model has a larger limit.
        if self.max_context_length < input_token_count + 2:
            # If the prompt is already at the max length, just request 1000 token, we will get an error anyway
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        start_time = time.time()

        if self.backend == "transformers":
            api_response = self._query_prompting_transformers(
                formatted_prompt=formatted_prompt,
                max_new_tokens=leftover_tokens_count,
            )
        else:
            api_response = self._query_prompting_openai(
                formatted_prompt=formatted_prompt,
                max_new_tokens=leftover_tokens_count,
            )
        end_time = time.time()

        return api_response, end_time - start_time

    def _query_prompting_openai(
        self,
        formatted_prompt: str,
        max_new_tokens: int,
    ):
        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        if len(extra_body) > 0:
            return self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=formatted_prompt,
                max_tokens=max_new_tokens,
                extra_body=extra_body,
                timeout=3600,  # Avoid timeout errors
            )

        return self.client.completions.create(
            model=self.model_path_or_id,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            timeout=3600,  # Avoid timeout errors
        )

    def _query_prompting_transformers(
        self,
        formatted_prompt: str,
        max_new_tokens: int,
    ):
        if self.model is None:
            raise RuntimeError(
                "Transformers backend is not initialized. Please call `spin_up_local_server` first."
            )

        encoded_prompt = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        model_input_device = next(self.model.parameters()).device
        encoded_prompt = {
            key: value.to(model_input_device) for key, value in encoded_prompt.items()
        }
        input_ids = encoded_prompt["input_ids"]
        prompt_tokens = int(input_ids.shape[-1])

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        if hasattr(self, "stop_token_ids"):
            generate_kwargs["eos_token_id"] = self.stop_token_ids

        lock = self._generation_lock or threading.Lock()
        with lock, torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=encoded_prompt.get("attention_mask"),
                **generate_kwargs,
            )

        completion_token_ids = generated_ids[0, input_ids.shape[-1] :]
        completion_tokens = int(completion_token_ids.shape[-1])
        generated_text = self.tokenizer.decode(
            completion_token_ids,
            skip_special_tokens=getattr(self, "skip_special_tokens", True),
        )
        return _OpenAICompatCompletionResponse(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response.choices[0].text,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )

        return inference_data
