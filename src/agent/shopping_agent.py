"""Main shopping agent orchestrator.

This module contains the ShoppingAgent class, which orchestrates the entire
query processing flow including LLM interaction, tool selection, context
management, and product extraction.
"""

import asyncio
import hashlib
import json
import re
import time
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq

from src.agent.reasoning import reasoner
from src.analytics.latency_tracker import latency_tracker
from src.analytics.logger import logger
from src.analytics.tracker import tracker
from src.analytics.langfuse_client import langfuse_client
from src.analytics.cost_tracker import cost_tracker
from src.mcp.mcp_client import tool_registry
from src.mcp.tools import product_tools, search_tools, cart_tools  # Import to register tools
from src.memory.conversation_store import conversation_store
from src.memory.user_preferences import preference_tracker
from src.utils.cache import cache_service
from src.utils.semantic_cache import semantic_cache
from src.agent.model_router import model_router
from src.utils.circuit_breaker import CircuitBreakerConfig, async_circuit_breaker
from src.utils.config import settings
from src.utils.retry import llm_retry, db_retry


class ShoppingAgent:
    """Main shopping assistant agent that orchestrates query processing.

    This class handles:
    - LLM initialization (Groq/OpenAI/Anthropic)
    - Tool creation and integration
    - Query processing with context enhancement
    - Product extraction and filtering
    - Caching and performance optimization
    - Error handling and resilience
    """

    def __init__(self):
        """Initialize the shopping agent with LLM and tools."""
        self.provider = settings.llm_provider.lower()
        self._llm_cache = {}
        self.llm = self._initialize_llm()
        self.tools = self._create_langchain_tools()

        # Store default persona and tone
        self.persona = "friendly"
        self.tone = "warm"

        self.agent = self._create_agent()

        logger.info(
            f"ShoppingAgent initialized successfully with {self.provider}"
        )

    def _initialize_llm(self, model: Optional[str] = None):
        """Initialize the configured LLM provider."""

        if model is None:
            model = settings.llm_model

        # Return cached model instance when available
        if model in self._llm_cache:
            return self._llm_cache[model]

        if self.provider == "groq":
            if not settings.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is required when using Groq provider"
                )

            llm = ChatGroq(
                model=model,
                temperature=settings.llm_temperature,
                groq_api_key=settings.groq_api_key,
            )

            logger.info(f"Initialized Groq LLM: {model}")
            self._llm_cache[model] = llm
            return llm

        elif self.provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when using Anthropic provider"
                )

            # Lazy import so Anthropic is not required for Groq mode
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=model,
                temperature=settings.llm_temperature,
                anthropic_api_key=settings.anthropic_api_key,
            )

            logger.info(f"Initialized Anthropic LLM: {model}")
            self._llm_cache[model] = llm
            return llm

        elif self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI provider"
                )

            # Lazy import so OpenAI is not required for Groq mode
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                temperature=settings.llm_temperature,
                openai_api_key=settings.openai_api_key,
            )

            logger.info(f"Initialized OpenAI LLM: {model}")
            self._llm_cache[model] = llm
            return llm

        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                "Use 'groq', 'anthropic', or 'openai'."
            )

    def _create_langchain_tools(self) -> List[StructuredTool]:
        """Convert MCP tools to LangChain StructuredTool instances."""
        from pydantic import create_model

        tools = []

        tools_config = [
            (
                "search_products",
                "Search for products by query. Returns list of products with details.",
            ),
            (
                "check_product_availability",
                "Check if a product is available and in stock.",
            ),
            (
                "check_product_price",
                "Get the current price of a product.",
            ),
            (
                "web_search",
                "Search the web for product reviews and information.",
            ),
            (
                "add_to_cart",
                "Add a product to the shopping cart.",
            ),
            (
                "get_cart",
                "Get the user's shopping cart.",
            ),
        ]

        for tool_name, description in tools_config:
            # Get the actual MCP tool to access its schema
            mcp_tool = tool_registry.get_tool(tool_name)

            if not mcp_tool:
                logger.warning(
                    f"Tool {tool_name} not found in registry, skipping"
                )
                continue

            # Get parameter schema from MCP tool
            params_schema = mcp_tool.get_parameters()

            async_wrapper = self._create_async_tool_wrapper(tool_name)
            sync_wrapper = self._create_sync_tool_fallback(
                tool_name,
                async_wrapper,
            )

            # Create Pydantic model from JSON schema
            if params_schema and "properties" in params_schema:
                field_definitions = {}

                for prop_name, prop_info in params_schema[
                    "properties"
                ].items():
                    prop_type = prop_info.get("type", "string")
                    required = prop_name in params_schema.get(
                        "required", []
                    )

                    # Map JSON schema types to Python types
                    if prop_type == "string":
                        field_type = str
                    elif prop_type == "integer":
                        field_type = int
                    elif prop_type == "number":
                        field_type = float
                    elif prop_type == "boolean":
                        field_type = bool
                    else:
                        field_type = str

                    if required:
                        field_definitions[prop_name] = (
                            field_type,
                            ...,
                        )
                    else:
                        from typing import Optional

                        field_definitions[prop_name] = (
                            Optional[field_type],
                            None,
                        )

                ArgsModel = create_model(
                    f"{tool_name}_Args",
                    **field_definitions,
                )

                tools.append(
                    StructuredTool(
                        name=tool_name,
                        description=description,
                        func=sync_wrapper,
                        coroutine=async_wrapper,
                        args_schema=ArgsModel,
                    )
                )
            else:
                tools.append(
                    StructuredTool.from_function(
                        func=sync_wrapper,
                        coroutine=async_wrapper,
                        name=tool_name,
                        description=description,
                    )
                )

        return tools

    def _create_async_tool_wrapper(self, tool_name: str):
        """Create an async wrapper for MCP tools."""

        async def async_executor(*args, **kwargs):
            tool = tool_registry.get_tool(tool_name)

            if not tool:
                return {"error": "Tool not found"}

            # Handle Pydantic model instance
            if args and hasattr(args[0], "__dict__"):
                kwargs = {
                    **args[0].__dict__,
                    **kwargs,
                }
                args = ()

            # Handle dict as first argument
            elif args and isinstance(args[0], dict):
                kwargs = {
                    **args[0],
                    **kwargs,
                }
                args = ()

            return await tool.execute(*args, **kwargs)

        return async_executor

    def _create_sync_tool_fallback(
        self,
        tool_name: str,
        async_wrapper,
    ):
        """Sync fallback for environments that call tools synchronously."""

        def sync_wrapper(*args, **kwargs):
            try:
                asyncio.get_running_loop()

                raise RuntimeError(
                    f"Tool '{tool_name}' was called synchronously while "
                    "an event loop is running. Use the async tool invocation path."
                )

            except RuntimeError as e:
                if "no running event loop" in str(e).lower():
                    return asyncio.run(
                        async_wrapper(*args, **kwargs)
                    )
                raise

        return sync_wrapper
    def _create_agent(
        self,
        persona: Optional[str] = None,
        tone: Optional[str] = None,
    ):
        """Create LangChain agent with system prompt and tools."""

        from langchain_core.prompts import (
            ChatPromptTemplate,
            MessagesPlaceholder,
        )

        persona = persona or self.persona
        tone = tone or self.tone

        system_prompt = self._get_system_prompt(
            persona,
            tone,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
        )

    def _get_system_prompt(
        self,
        persona: Optional[str] = "friendly",
        tone: Optional[str] = "warm",
    ) -> str:
        """Get the system prompt for the shopping assistant."""

        from pathlib import Path

        current_dir = Path(__file__).parent
        prompt_file = (
            current_dir
            / "prompts"
            / "shopping_assistant.txt"
        )

        try:
            with open(
                prompt_file,
                "r",
                encoding="utf-8",
            ) as f:
                raw_prompt = f.read()

        except FileNotFoundError:
            logger.warning(
                f"Prompt file not found at {prompt_file}, "
                "using fallback prompt"
            )
            raw_prompt = self._get_fallback_prompt()

        except Exception as e:
            logger.error(
                f"Error loading prompt file: {e}, "
                "using fallback prompt"
            )
            raw_prompt = self._get_fallback_prompt()

        base_prompt, personas, tones = self._parse_prompt_blocks(
            raw_prompt
        )

        persona_instructions = personas.get(
            (persona or "friendly").lower(),
            personas.get("friendly", ""),
        )

        tone_instructions = tones.get(
            (tone or "warm").lower(),
            tones.get("warm", ""),
        )

        customized_prompt = f"""{base_prompt}

PERSONA & COMMUNICATION STYLE:
{persona_instructions}

TONE OF VOICE:
{tone_instructions}

Remember to maintain your persona and tone consistently throughout the conversation."""

        return customized_prompt

    def _parse_prompt_blocks(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        """Parse prompt file containing BASE, PERSONA and TONE blocks."""

        base_lines: List[str] = []
        personas: Dict[str, List[str]] = {}
        tones: Dict[str, List[str]] = {}

        current_section: Optional[str] = None
        current_key: Optional[str] = None

        for raw_line in (text or "").splitlines():
            line = raw_line.rstrip("\n")
            marker = line.strip()

            if marker == "@@BASE":
                current_section = "base"
                current_key = None
                continue

            if marker.lower().startswith("@@persona "):
                current_section = "persona"
                current_key = marker.split(
                    " ",
                    1,
                )[1].strip().lower()

                personas.setdefault(
                    current_key,
                    [],
                )
                continue

            if marker.lower().startswith("@@tone "):
                current_section = "tone"
                current_key = marker.split(
                    " ",
                    1,
                )[1].strip().lower()

                tones.setdefault(
                    current_key,
                    [],
                )
                continue

            if current_section == "base":
                base_lines.append(line)

            elif current_section == "persona" and current_key:
                personas[current_key].append(line)

            elif current_section == "tone" and current_key:
                tones[current_key].append(line)

            else:
                base_lines.append(line)

        base_prompt = "\n".join(base_lines).strip()

        persona_map = {
            k: "\n".join(v).strip()
            for k, v in personas.items()
            if "\n".join(v).strip()
        }

        tone_map = {
            k: "\n".join(v).strip()
            for k, v in tones.items()
            if "\n".join(v).strip()
        }

        return (
            base_prompt,
            persona_map,
            tone_map,
        )

    def _get_persona_instructions(
        self,
        persona: str,
    ) -> str:
        """Get persona-specific instructions."""

        base, personas, _ = self._parse_prompt_blocks(
            self._get_system_prompt_raw()
        )

        return personas.get(
            persona.lower(),
            personas.get("friendly", ""),
        )

    def _get_tone_instructions(
        self,
        tone: str,
    ) -> str:
        """Get tone-specific instructions."""

        base, _, tones = self._parse_prompt_blocks(
            self._get_system_prompt_raw()
        )

        return tones.get(
            tone.lower(),
            tones.get("warm", ""),
        )

    def _get_system_prompt_raw(self) -> str:
        """Load the prompt file contents."""

        from pathlib import Path

        current_dir = Path(__file__).parent

        prompt_file = (
            current_dir
            / "prompts"
            / "shopping_assistant.txt"
        )

        try:
            return prompt_file.read_text(
                encoding="utf-8"
            )
        except Exception:
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if file cannot be loaded."""

        return """You are an AI shopping assistant that helps users find and purchase products.

CORE RESPONSIBILITIES:
1. Use search_products tool when users want to find products. Extract price constraints and pass min_price/max_price when available.
2. Always include product purchase links when available.
3. Strictly respect user's price range.
4. Remember conversation context.
5. Use web_search for product details, specifications, or reviews when needed.
6. For cart operations, search for the product first.

RESPONSE FORMAT:
**Product Name** - Price
• Feature 1
• Feature 2
• Rating: X.X/5 (if available)
• Buy here: product URL

COMMUNICATION:
- Be friendly, helpful, and conversational
- Use natural language
- Be concise but thorough
- If products don't match criteria, explain why and suggest alternatives
- Never fabricate product information

SAFETY:
- Never provide information about illegal products
- Never share personal information
- Never make medical claims
- Only use information from tool results

Remember: Help users find the perfect products within their budget while maintaining conversation context."""

    @async_circuit_breaker(
        config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            name="llm_api",
        )
    )
    @llm_retry
    async def _invoke_agent_with_retry(
        self,
        messages,
        llm: Optional[Any] = None,
    ):
        """Invoke agent with retry and circuit breaker protection."""

        try:
            return await asyncio.wait_for(
                self.agent.ainvoke(
                    {"messages": messages}
                ),
                timeout=90.0,
            )

        except asyncio.TimeoutError:
            logger.error(
                "LLM invocation timed out after 90 seconds"
            )

            raise TimeoutError(
                "LLM request timed out. Please try again "
                "with a simpler query."
            )

    async def stream_response(
        self,
        messages: List,
    ) -> AsyncIterator[str]:
        """Stream agent response token by token."""

        try:
            async for chunk in self.agent.astream(
                {"messages": messages}
            ):
                if isinstance(chunk, dict):
                    for key, value in chunk.items():
                        if (
                            isinstance(value, dict)
                            and "messages" in value
                        ):
                            for msg in value["messages"]:
                                if (
                                    hasattr(msg, "content")
                                    and msg.content
                                ):
                                    yield msg.content

                                elif (
                                    isinstance(msg, dict)
                                    and "content" in msg
                                ):
                                    yield msg["content"]

                elif hasattr(chunk, "content"):
                    yield chunk.content

                elif isinstance(chunk, str):
                    yield chunk

        except Exception as e:
            logger.debug(
                f"Agent streaming not available, "
                f"falling back to non-streaming: {e}"
            )

            response = await self.agent.ainvoke(
                {"messages": messages}
            )

            if (
                isinstance(response, dict)
                and "messages" in response
            ):
                full_text = ""

                for msg in response["messages"]:
                    if hasattr(msg, "content"):
                        full_text += msg.content
                    elif isinstance(msg, dict):
                        full_text += msg.get(
                            "content",
                            "",
                        )

                chunk_size = 10

                for i in range(
                    0,
                    len(full_text),
                    chunk_size,
                ):
                    yield full_text[
                        i : i + chunk_size
                    ]

                    await asyncio.sleep(0.01)

    def _track_ttft(
        self,
        request_id: str,
        llm_start_time: float,
    ):
        """Track Time To First Token."""

        try:

            async def get_first_token():
                async for chunk in self.llm.astream([]):
                    return time.time()

            first_token_time = asyncio.run(
                get_first_token()
            )

            ttft = (
                first_token_time
                - llm_start_time
            )

            latency_tracker.track_ttft(
                request_id,
                ttft,
            )

            logger.debug(
                f"First token received after {ttft:.3f}s"
            )

            return ttft

        except Exception as e:
            logger.debug(
                f"TTFT tracking failed: {e}"
            )
            return None

    def _extract_products_from_response(
        self,
        response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract products from agent response messages."""

        products = []

        if (
            not isinstance(response, dict)
            or "messages" not in response
        ):
            return products

        messages = response.get(
            "messages",
            [],
        )

        for msg in messages:
            if isinstance(msg, ToolMessage):
                products = self._extract_from_tool_message(
                    msg
                )

                if products:
                    break

            elif isinstance(msg, dict):
                products = self._extract_from_dict_message(
                    msg
                )

                if products:
                    break

        return products

    def _extract_from_tool_message(
        self,
        msg: ToolMessage,
    ) -> List[Dict[str, Any]]:
        """Extract products from a ToolMessage."""

        content = getattr(
            msg,
            "content",
            None,
        )

        if not content:
            return []

        if isinstance(content, str):
            try:
                content_dict = json.loads(content)

                if (
                    isinstance(content_dict, dict)
                    and "products" in content_dict
                ):
                    products = content_dict.get(
                        "products",
                        [],
                    )

                    if products:
                        logger.info(
                            f"Extracted {len(products)} products "
                            "from ToolMessage (JSON string)"
                        )
                        return products

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                pass

        elif (
            isinstance(content, dict)
            and "products" in content
        ):
            products = content.get(
                "products",
                [],
            )

            if products:
                logger.info(
                    f"Extracted {len(products)} products "
                    "from ToolMessage (dict)"
                )

                return products

        return []

    def _extract_from_dict_message(
        self,
        msg: dict,
    ) -> List[Dict[str, Any]]:
        """Extract products from a dict-based message."""

        if (
            "name" in msg
            and msg.get("name") == "search_products"
        ):
            content = msg.get(
                "content",
                {},
            )

            if (
                isinstance(content, dict)
                and "products" in content
            ):
                products = content.get(
                    "products",
                    [],
                )

                if products:
                    logger.info(
                        f"Extracted {len(products)} products "
                        "from message dict"
                    )

                    return products

        content = msg.get(
            "content",
            {},
        )

        if (
            isinstance(content, dict)
            and "products" in content
        ):
            products = content.get(
                "products",
                [],
            )

            if products:
                logger.info(
                    f"Extracted {len(products)} products "
                    "from message content"
                )

                return products

        if isinstance(content, str):
            try:
                content_dict = json.loads(content)

                if (
                    isinstance(content_dict, dict)
                    and "products" in content_dict
                ):
                    products = content_dict.get(
                        "products",
                        [],
                    )

                    if products:
                        logger.info(
                            f"Extracted {len(products)} products "
                            "from message JSON string"
                        )

                        return products

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                pass

        return []

    def _extract_tools_used(
        self,
        response: Dict[str, Any],
    ) -> List[str]:
        """Extract list of tools used from agent response."""

        tools_used = []

        if (
            not isinstance(response, dict)
            or "messages" not in response
        ):
            return tools_used

        for msg in response.get(
            "messages",
            [],
        ):
            tool_calls = None

            if (
                hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                tool_calls = msg.tool_calls

            elif (
                isinstance(msg, dict)
                and "tool_calls" in msg
            ):
                tool_calls = msg.get(
                    "tool_calls",
                    [],
                )

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = (
                        tool_call.get("name", "")
                        if isinstance(
                            tool_call,
                            dict,
                        )
                        else getattr(
                            tool_call,
                            "name",
                            "",
                        )
                    )

                    if (
                        tool_name
                        and tool_name not in tools_used
                    ):
                        tools_used.append(
                            tool_name
                        )

        return tools_used

    async def _refetch_products_with_price_filter(
        self,
        query: str,
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Refetch products with price filtering as fallback."""

        from src.mcp.tools.product_tools import (
            product_search_tool,
        )

        try:
            tool_result = await product_search_tool.execute(
                query,
                max_results=15,
                min_price=min_price,
                max_price=max_price,
            )

            if (
                isinstance(tool_result, dict)
                and "products" in tool_result
            ):
                return tool_result.get(
                    "products",
                    [],
                )

        except Exception as e:
            logger.error(
                f"Failed to refetch products with price filter: {e}",
                exc_info=True,
            )

        return []

    def _should_refetch_products(
        self,
        products: List[Dict[str, Any]],
        price_range: tuple[
            Optional[float],
            Optional[float],
        ],
    ) -> bool:
        """Determine if products should be refetched."""

        if len(products) < 3:
            return True

        if (
            not price_range
            or (
                not price_range[0]
                and not price_range[1]
            )
        ):
            return False

        min_price, max_price = price_range

        matching_count = 0

        for product in products:
            price = self._parse_product_price(
                product.get("price", 0)
            )

            price_matches = True

            if (
                min_price is not None
                and price < min_price
            ):
                price_matches = False

            if (
                max_price is not None
                and price > max_price
            ):
                price_matches = False

            if price_matches:
                matching_count += 1

        if matching_count < 3:
            logger.info(
                f"Only {matching_count} products match price range, "
                "refetching with filters"
            )

            return True

        return False

    def _parse_product_price(
        self,
        price: Any,
    ) -> float:
        """Parse product price from various formats."""

        if isinstance(
            price,
            (int, float),
        ):
            return float(price)

        if isinstance(price, str):
            price_str = (
                price
                .replace("$", "")
                .replace(",", "")
                .replace("USD", "")
                .strip()
            )

            numeric_chars = "".join(
                c
                for c in price_str
                if c.isdigit() or c == "."
            )

            try:
                return (
                    float(numeric_chars)
                    if numeric_chars
                    else 0.0
                )

            except (
                ValueError,
                AttributeError,
            ):
                return 0.0

        return 0.0

    async def process_query(
        self,
        query: str,
        session_id: str,
        user_id: Optional[int] = None,
        persona: Optional[str] = "friendly",
        tone: Optional[str] = "warm",
    ) -> Dict[str, Any]:
        """Process a user query with full orchestration."""

        request_id = (
            latency_tracker.generate_request_id()
        )

        start_time = time.time()

        langfuse_trace = None
        trace_id = None

        if langfuse_client.enabled:
            try:
                langfuse_trace = langfuse_client.trace(
                    name="shopping_assistant_query",
                    input={
                        "query": query,
                        "persona": persona,
                        "tone": tone,
                    },
                    user_id=(
                        str(user_id)
                        if user_id
                        else None
                    ),
                    session_id=session_id,
                    tags=[
                        "shopping",
                        "agent",
                    ],
                )

                if langfuse_trace:
                    try:
                        if hasattr(
                            langfuse_trace,
                            "id",
                        ):
                            trace_id = langfuse_trace.id

                        elif isinstance(
                            langfuse_trace,
                            dict,
                        ):
                            trace_id = (
                                langfuse_trace.get(
                                    "id"
                                )
                            )

                        elif hasattr(
                            langfuse_trace,
                            "trace_id",
                        ):
                            trace_id = (
                                langfuse_trace.trace_id
                            )

                    except (
                        AttributeError,
                        TypeError,
                    ) as attr_err:
                        logger.debug(
                            "Failed to extract trace_id "
                            f"from langfuse_trace: {attr_err}"
                        )

                        trace_id = None

            except Exception as trace_err:
                logger.warning(
                    f"Failed to create Langfuse trace: {trace_err}"
                )

                langfuse_trace = None
                trace_id = None

        try:
            if trace_id:
                try:
                    langfuse_client.span(
                        trace_id=trace_id,
                        name="get_conversation_history",
                        input={
                            "session_id": session_id
                        },
                    )

                    langfuse_client.span(
                        trace_id=trace_id,
                        name="get_user_preferences",
                        input={
                            "user_id": user_id
                        },
                    )

                except Exception as span_err:
                    logger.debug(
                        f"Failed to create Langfuse spans: {span_err}"
                    )

            history_task = (
                self._get_conversation_history(
                    session_id,
                    request_id,
                )
            )

            preferences_task = (
                self._get_user_preferences(
                    user_id,
                    request_id,
                )
            )

            history, preferences = await asyncio.gather(
                history_task,
                preferences_task,
            )

            if trace_id:
                try:
                    langfuse_client.span(
                        trace_id=trace_id,
                        name="detect_intent",
                        input={
                            "query": query
                        },
                    )

                except Exception as span_err:
                    logger.debug(
                        f"Failed to create Langfuse span "
                        f"for detect_intent: {span_err}"
                    )

            intent = self._detect_intent(
                query,
                request_id,
            )

            if (
                persona != self.persona
                or tone != self.tone
            ):
                self.persona = persona
                self.tone = tone

                self.agent = self._create_agent(
                    persona,
                    tone,
                )

                logger.info(
                    f"Agent recreated with persona: {persona}, tone: {tone}"
                )

            messages = self._build_messages(
                history,
                query,
            )

            messages.append(
                HumanMessage(
                    content=query
                )
            )

            logger.debug(
                f"Messages being sent to LLM: "
                f"{len(messages)} total messages"
            )

            logger.debug(
                f"History count: {len(history)}"
            )

            logger.debug(
                f"Current query: {query}"
            )

            context_hash = (
                self._create_context_hash(
                    messages
                )
            )

            cached_response = None

            if getattr(
                settings,
                "semantic_only_retrieval",
                False,
            ):
                try:
                    semantic_response = (
                        await semantic_cache.get_semantic(
                            query,
                            context_hash=context_hash,
                        )
                    )

                    if semantic_response:
                        cached_response = (
                            semantic_response
                        )

                        logger.info(
                            f"Semantic cache hit for query: "
                            f"{query[:50]}..."
                        )

                except Exception as e:
                    logger.debug(
                        f"Semantic cache lookup failed: {e}"
                    )

            else:
                cached_response = (
                    await cache_service.get_llm_response(
                        query,
                        context_hash=context_hash,
                    )
                )

                if not cached_response:
                    try:
                        semantic_response = (
                            await semantic_cache.get_semantic(
                                query,
                                context_hash=context_hash,
                            )
                        )

                        if semantic_response:
                            cached_response = (
                                semantic_response
                            )

                            logger.info(
                                f"Semantic cache hit for query: "
                                f"{query[:50]}..."
                            )

                    except Exception as e:
                        logger.debug(
                            f"Semantic cache lookup failed: {e}"
                        )

            if cached_response:
                (
                    agent_response,
                    products,
                    tools_used,
                ) = self._extract_from_cache(
                    cached_response
                )

            else:
                (
                    agent_response,
                    products,
                    tools_used,
                    response,
                ) = await self._process_with_llm(
                    messages,
                    query,
                    intent,
                    request_id,
                    langfuse_trace,
                    trace_id,
                )

                if getattr(
                    settings,
                    "semantic_only_retrieval",
                    False,
                ):
                    try:
                        await semantic_cache.set_semantic(
                            query=query,
                            response_data={
                                "response": agent_response,
                                "products": products,
                                "tools_used": tools_used,
                            },
                            context_hash=context_hash,
                            ttl=settings.cache_llm_response_ttl,
                        )

                    except Exception as e:
                        logger.debug(
                            f"Failed to store in semantic cache: {e}"
                        )

                else:
                    await self._cache_llm_response(
                        query,
                        context_hash,
                        agent_response,
                        products,
                        tools_used,
                    )

            products = (
                await self._ensure_products_with_price_filter(
                    query,
                    products,
                    intent,
                    tools_used,
                )
            )

            if not getattr(
                settings,
                "semantic_only_retrieval",
                False,
            ):
                products = (
                    self._filter_products_by_brand(
                        query,
                        products,
                        history,
                    )
                )

            if (
                products
                and "search_products"
                not in (tools_used or [])
            ):
                tools_used = (
                    tools_used or []
                ) + ["search_products"]

            agent_response = (
                self._ground_response_to_products(
                    query=query,
                    agent_response=agent_response,
                    products=products,
                    tools_used=tools_used,
                    intent=intent,
                )
            )

            await self._store_conversation(
                session_id,
                query,
                agent_response,
                tools_used,
                request_id,
            )

            response_time = (
                time.time() - start_time
            )

            self._track_analytics(
                query,
                tools_used,
                agent_response,
                response_time,
                request_id,
            )

            if trace_id:
                try:
                    metadata = {
                        "session_id": session_id,
                        "request_id": request_id,
                        "latency_breakdown": (
                            latency_tracker.get_request_breakdown(
                                request_id
                            )
                        ),
                        "persona": persona,
                        "tone": tone,
                        "cached": (
                            cached_response is not None
                        ),
                    }

                    if getattr(
                        settings,
                        "bedrock_enabled",
                        False,
                    ):
                        metadata.update(
                            {
                                "llm_provider": "bedrock",
                                "aws_region": getattr(
                                    settings,
                                    "aws_region",
                                    "us-east-1",
                                ),
                                "model_identifier": (
                                    settings.llm_model
                                ),
                                "cloudwatch_namespace": getattr(
                                    settings,
                                    "cloudwatch_namespace",
                                    "ShoppingAssistant/Application",
                                ),
                            }
                        )

                    if getattr(
                        settings,
                        "cloudwatch_enabled",
                        False,
                    ):
                        region = getattr(
                            settings,
                            "aws_region",
                            "us-east-1",
                        )

                        namespace = getattr(
                            settings,
                            "cloudwatch_namespace",
                            "ShoppingAssistant/Application",
                        )

                        metadata[
                            "cloudwatch_link"
                        ] = (
                            f"https://{region}.console.aws.amazon.com/"
                            f"cloudwatch/home?region={region}"
                            f"#metricsV2:graph=~();namespace={namespace}"
                        )

                    langfuse_client.update_trace(
                        trace_id=trace_id,
                        output={
                            "response": (
                                agent_response[:500]
                                if agent_response
                                else ""
                            ),
                            "products_count": len(
                                products
                            ),
                            "tools_used": tools_used,
                            "intent": intent.get(
                                "type",
                                "unknown",
                            ),
                            "response_time": response_time,
                            "cached": (
                                cached_response
                                is not None
                            ),
                        },
                        metadata=metadata,
                    )

                    langfuse_client.flush()

                except Exception as e:
                    logger.debug(
                        f"Failed to update/flush "
                        f"Langfuse trace: {e}"
                    )

            return {
                "response": agent_response,
                "products": products,
                "tools_used": tools_used,
                "intent": intent["type"],
                "session_id": session_id,
                "request_id": request_id,
                "latency_breakdown": (
                    latency_tracker.get_request_breakdown(
                        request_id
                    )
                ),
                "cached": (
                    cached_response is not None
                ),
                "langfuse_trace_id": trace_id,
            }

        except Exception as e:
            if langfuse_trace:
                try:
                    langfuse_client.flush()
                except Exception as langfuse_err:
                    logger.debug(
                        f"Failed to flush Langfuse trace: "
                        f"{langfuse_err}"
                    )

            logger.error(
                f"Error processing query: {e}",
                exc_info=True,
            )

            return {
                "response": (
                    "I'm sorry, I encountered an error "
                    "processing your request. Please try again."
                ),
                "error": str(e),
                "session_id": session_id,
            }

    @db_retry
    async def _get_conversation_history(
        self,
        session_id: str,
        request_id: str,
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""

        with latency_tracker.track_component(
            "conversation_history",
            request_id,
        ):
            try:
                return await conversation_store.get_conversation_history(
                    session_id,
                    limit=10,
                )

            except Exception as e:
                logger.warning(
                    f"Failed to get conversation history "
                    f"after retries: {e}"
                )

                return []

    @db_retry
    async def _get_user_preferences(
        self,
        user_id: Optional[int],
        request_id: str,
    ) -> Dict[str, Any]:
        """Get user preferences."""

        if not user_id:
            return {}

        with latency_tracker.track_component(
            "user_preferences",
            request_id,
        ):
            try:
                return preference_tracker.get_preferences(
                    user_id
                )

            except Exception as e:
                logger.warning(
                    f"Failed to get user preferences "
                    f"after retries: {e}"
                )

                return {}

    def _detect_intent(
        self,
        query: str,
        request_id: str,
    ) -> Dict[str, Any]:
        """Detect user intent."""

        with latency_tracker.track_component(
            "intent_detection",
            request_id,
        ):
            intent = reasoner.determine_intent(
                query
            )

        logger.info(
            f"Intent detected: {intent['type']} "
            f"(confidence: {intent['confidence']})"
        )

        return intent

    def _build_messages(
        self,
        history: List[Dict[str, Any]],
        query: str,
    ) -> List:
        """Build message list from conversation history."""

        messages = []

        limited_history = (
            history[-10:]
            if len(history) > 10
            else history
        )

        if not limited_history:
            return messages

        for idx, msg in enumerate(
            limited_history
        ):
            position = (
                len(limited_history) - idx
            )

            if msg.get("user_message"):
                messages.append(
                    HumanMessage(
                        content=msg[
                            "user_message"
                        ]
                    )
                )

            if msg.get("agent_response"):
                agent_response = msg[
                    "agent_response"
                ]

                max_length = (
                    500
                    if position <= 5
                    else 300
                )

                if len(agent_response) > max_length:
                    agent_response = (
                        agent_response[:max_length]
                        + "..."
                    )

                messages.append(
                    AIMessage(
                        content=agent_response
                    )
                )

        logger.debug(
            f"Building messages with "
            f"{len(limited_history)} history entries. "
            f"Total messages: {len(messages)}"
        )

        return messages

    def _create_context_hash(
        self,
        messages: List,
    ) -> str:
        """Create hash from recent messages."""

        message_contents = [
            (
                msg.content
                if hasattr(msg, "content")
                else str(msg)
            )
            for msg in messages[-3:]
        ]

        return hashlib.sha256(
            json.dumps(
                message_contents,
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

    def _extract_from_cache(
        self,
        cached_response: Dict[str, Any],
    ) -> tuple[
        str,
        List[Dict],
        List[str],
    ]:
        """Extract data from cached response."""

        logger.info(
            "Cache hit for LLM response"
        )

        return (
            cached_response.get(
                "response",
                "",
            ),
            cached_response.get(
                "products",
                [],
            ),
            cached_response.get(
                "tools_used",
                [],
            ),
        )

    async def _process_with_llm(
        self,
        messages: List,
        query: str,
        intent: Dict[str, Any],
        request_id: str,
        langfuse_trace: Optional[Any] = None,
        trace_id: Optional[str] = None,
    ) -> tuple[
        str,
        List[Dict],
        List[str],
        Optional[Dict],
    ]:
        """Process query with LLM and extract response."""

        llm_start_time = time.time()
        first_token_time = None
        response = None

        selected_model = settings.llm_model
        complexity = "default"

        try:
            selected_model, complexity = (
                model_router.select_model(
                    query,
                    intent,
                )
            )

        except Exception as e:
            logger.debug(
                f"Model routing classification failed: {e}"
            )

        if not trace_id and langfuse_trace:
            try:
                if hasattr(
                    langfuse_trace,
                    "id",
                ):
                    trace_id = (
                        langfuse_trace.id
                    )

                elif isinstance(
                    langfuse_trace,
                    dict,
                ):
                    trace_id = (
                        langfuse_trace.get(
                            "id"
                        )
                    )

                elif hasattr(
                    langfuse_trace,
                    "trace_id",
                ):
                    trace_id = (
                        langfuse_trace.trace_id
                    )

            except Exception as e:
                logger.debug(
                    f"Failed to extract trace_id: {e}"
                )

                trace_id = None

        # TTFT probe
        try:
            async for chunk in self.llm.astream(
                messages
            ):
                if first_token_time is None:
                    first_token_time = (
                        time.time()
                    )

                    ttft = (
                        first_token_time
                        - llm_start_time
                    )

                    latency_tracker.track_ttft(
                        request_id,
                        ttft,
                    )

                    logger.debug(
                        f"First token received after "
                        f"{ttft:.3f}s"
                    )

                    break

        except Exception as e:
            logger.debug(
                f"LLM streaming not available: {e}"
            )

        llm_input = {
            "messages": [
                {
                    "role": (
                        msg.__class__.__name__
                        .replace(
                            "Message",
                            "",
                        )
                        .lower()
                    ),
                    "content": (
                        msg.content
                        if hasattr(
                            msg,
                            "content",
                        )
                        else str(msg)
                    ),
                }
                for msg in messages[-10:]
            ]
        }

        with latency_tracker.track_component(
            "llm_processing",
            request_id,
        ):
            try:
                response = (
                    await self._invoke_agent_with_retry(
                        messages
                    )
                )

            except TimeoutError as e:
                logger.error(
                    f"LLM processing timeout after "
                    f"90 seconds: {e}"
                )

                raise TimeoutError(
                    "LLM request timed out. "
                    "The query may be too complex or "
                    "the service is slow. Please try again."
                )

            except Exception as e:
                logger.error(
                    f"LLM processing error: {e}"
                )
                raise

        if first_token_time is None:
            llm_time = (
                time.time()
                - llm_start_time
            )

            estimated_ttft = (
                llm_time * 0.20
            )

            latency_tracker.track_ttft(
                request_id,
                estimated_ttft,
            )

        agent_response = (
            self._extract_agent_response(
                response
            )
        )

        tools_used = (
            self._extract_tools_used(
                response
            )
            if response
            else intent.get(
                "tools_needed",
                [],
            ).copy()
        )

        products = (
            self._extract_products_from_response(
                response
            )
            if response
            else []
        )

        if products:
            products = (
                self._validate_products_against_tools(
                    products,
                    tools_used,
                )
            )

        # Final deterministic deal response.
        # Product/tool data is authoritative; do not trust LLM-generated prices.
        if products:
            lines = [
                "Here are the best verified options I found:",
                "Why these match: ranked using requirement match, price, rating, availability, and deal quality.",
                "",
            ]

            for i, product in enumerate(products[:5], 1):
                name = product.get("name", "Unknown product")
                price = product.get("price")
                currency = str(product.get("currency", "INR")).upper()
                rating = product.get("rating")
                url = product.get("product_url") or product.get("url")

                if price is not None:
                    try:
                        price_text = f"₹{float(price):.2f}" if currency == "INR" else f"{currency} {float(price):.2f}"
                    except (TypeError, ValueError):
                        price_text = str(price)
                else:
                    price_text = "Price unavailable"

                lines.append(f"{i}. **{name}** - {price_text}")

                if rating is not None:
                    lines.append(f"   - Rating: {rating}/5")

                deal = product.get("deal_info") or {}
                if deal.get("is_deal"):
                    badge = deal.get("deal_badge")
                    savings = deal.get("savings_percent")
                    if badge:
                        if savings:
                            lines.append(f"   - Deal: {badge} ({savings:.1f}% savings)")
                        else:
                            lines.append(f"   - Deal: {badge}")

                if url:
                    lines.append(f"   - [View Product]({url})")

                lines.append("")

            agent_response = "\n".join(lines).rstrip()

        usage = None

        if trace_id:
            try:
                if (
                    hasattr(
                        response,
                        "response_metadata",
                    )
                    and response.response_metadata
                ):
                    usage = (
                        response.response_metadata.get(
                            "token_usage"
                        )
                    )

                elif (
                    isinstance(
                        response,
                        dict,
                    )
                    and "response_metadata"
                    in response
                ):
                    usage = (
                        response[
                            "response_metadata"
                        ].get(
                            "token_usage"
                        )
                    )

                generation_metadata = {
                    "complexity": complexity,
                    "actual_model": settings.llm_model,
                    "routed_model": selected_model,
                    "provider": settings.llm_provider,
                }

                langfuse_client.generation(
                    trace_id=trace_id,
                    name="llm_inference",
                    model=selected_model,
                    input=llm_input,
                    output={
                        "response": agent_response[:500],
                        "products_count": len(
                            products
                        ),
                        "tools_used": tools_used,
                    },
                    usage=usage,
                    metadata=generation_metadata,
                )

            except Exception as e:
                logger.debug(
                    f"Failed to create Langfuse generation: {e}"
                )

        input_tokens = 0
        output_tokens = 0

        if usage:
            input_tokens = (
                usage.get(
                    "input_tokens",
                    0,
                )
                or usage.get(
                    "prompt_tokens",
                    0,
                )
                or 0
            )

            output_tokens = (
                usage.get(
                    "output_tokens",
                    0,
                )
                or usage.get(
                    "completion_tokens",
                    0,
                )
                or 0
            )

        if (
            input_tokens == 0
            and output_tokens == 0
            and response
        ):
            try:
                if (
                    hasattr(
                        response,
                        "response_metadata",
                    )
                    and response.response_metadata
                ):
                    token_usage = (
                        response.response_metadata.get(
                            "token_usage",
                            {},
                        )
                    )

                    if token_usage:
                        input_tokens = (
                            token_usage.get(
                                "input_tokens",
                                0,
                            )
                            or token_usage.get(
                                "prompt_tokens",
                                0,
                            )
                            or 0
                        )

                        output_tokens = (
                            token_usage.get(
                                "output_tokens",
                                0,
                            )
                            or token_usage.get(
                                "completion_tokens",
                                0,
                            )
                            or 0
                        )

                elif isinstance(
                    response,
                    dict,
                ):
                    token_usage = (
                        response.get(
                            "response_metadata",
                            {},
                        ).get(
                            "token_usage",
                            {},
                        )
                    )

                    if token_usage:
                        input_tokens = (
                            token_usage.get(
                                "input_tokens",
                                0,
                            )
                            or token_usage.get(
                                "prompt_tokens",
                                0,
                            )
                            or 0
                        )

                        output_tokens = (
                            token_usage.get(
                                "output_tokens",
                                0,
                            )
                            or token_usage.get(
                                "completion_tokens",
                                0,
                            )
                            or 0
                        )

            except Exception as e:
                logger.debug(
                    f"Failed to extract token usage: {e}"
                )

        if (
            input_tokens > 0
            or output_tokens > 0
        ):
            try:
                cost_tracker.record_usage(
                    model=selected_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_id=request_id,
                    query=(
                        query[:200]
                        if query
                        else None
                    ),
                )

            except Exception as e:
                logger.debug(
                    f"Failed to track costs: {e}"
                )

        return (
            agent_response,
            products,
            tools_used,
            response,
        )

    def _extract_agent_response(
        self,
        response: Optional[Dict[str, Any]],
    ) -> str:
        """Extract agent response text."""

        if not response:
            return (
                "I'm sorry, I couldn't process that request."
            )

        if (
            isinstance(response, dict)
            and "messages" in response
        ):
            last_message = (
                response["messages"][-1]
                if response["messages"]
                else None
            )

            if last_message:
                if hasattr(
                    last_message,
                    "content",
                ):
                    return (
                        self._validate_and_clean_response(
                            last_message.content
                        )
                    )

                elif (
                    isinstance(
                        last_message,
                        dict,
                    )
                    and "content" in last_message
                ):
                    return (
                        self._validate_and_clean_response(
                            last_message[
                                "content"
                            ]
                        )
                    )

        if (
            isinstance(response, dict)
            and "output" in response
        ):
            return (
                self._validate_and_clean_response(
                    response["output"]
                )
            )

        content = str(response)

        return (
            self._validate_and_clean_response(
                content
            )
        )

    def _validate_and_clean_response(
        self,
        response_text: str,
    ) -> str:
        """Validate response and clean placeholder text."""

        if not isinstance(
            response_text,
            str,
        ):
            return response_text

        placeholder_patterns = [
            r"\[(?:CVS|Walmart|Walgreens|Target|Amazon|Store|Website|Product|Online)\s+(?:Website|Store|Link|Purchase|Buy)\s*Link?\]",
            r"\[(?:Website|Store|Product|Online|Purchase|Buy)\s*Link?\]",
            r"\[Link\s*to\s*[^\]]+\]",
            r"\[[^\]]*Link[^\]]*\]",
        ]

        for pattern in placeholder_patterns:
            matches = re.findall(
                pattern,
                response_text,
                re.IGNORECASE,
            )

            if matches:
                logger.warning(
                    f"Detected placeholder text in response: {matches}"
                )

                for match in matches:
                    response_text = (
                        response_text.replace(
                            match,
                            "Product link not available",
                        )
                    )

        url_pattern = r"https?://[^\s\)]+"

        urls = re.findall(
            url_pattern,
            response_text,
        )

        if urls:
            suspicious_patterns = [
                r"example\.com",
                r"placeholder",
                r"product_url",
                r"website_link",
            ]

            valid_urls = []

            for url in urls:
                is_suspicious = any(
                    re.search(
                        pattern,
                        url,
                        re.IGNORECASE,
                    )
                    for pattern in suspicious_patterns
                )

                if not is_suspicious:
                    try:
                        from urllib.parse import urlparse

                        parsed = urlparse(url)

                        if (
                            parsed.netloc
                            and "."
                            in parsed.netloc
                        ):
                            valid_urls.append(
                                url
                            )

                    except Exception:
                        pass

            if valid_urls:
                logger.debug(
                    f"Found {len(valid_urls)} valid URLs"
                )

        if (
            "[product_url]"
            in response_text
            or "[Product URL]"
            in response_text
        ):
            response_text = response_text.replace(
                "[product_url]",
                "Product link not available",
            )

            response_text = response_text.replace(
                "[Product URL]",
                "Product link not available",
            )

        store_address_pattern = (
            r"\b(at|located at|found at)\s+"
            r"\d+\s+[A-Z][a-z]+\s+"
            r"(Street|St|Road|Rd|Avenue|Ave|Drive|Dr|"
            r"Lane|Ln|Boulevard|Blvd|Way|Circle|Ct)"
        )

        potential_addresses = re.findall(
            store_address_pattern,
            response_text,
            re.IGNORECASE,
        )

        if potential_addresses:
            logger.warning(
                f"Potential hallucinated store addresses detected: "
                f"{potential_addresses}"
            )

        price_mentions = re.findall(
            r"\$\d+\.\d{2}",
            response_text,
        )

        if len(price_mentions) > 10:
            logger.warning(
                f"Many price mentions detected "
                f"({len(price_mentions)})"
            )

        return response_text

    def _ground_response_to_products(
        self,
        query: str,
        agent_response: str,
        products: List[Dict[str, Any]],
        tools_used: List[str],
        intent: Dict[str, Any],
        max_items: int = 5,
    ) -> str:
        """Ground response to actual returned products."""

        try:
            if (
                not products
                or "search_products"
                not in (tools_used or [])
            ):
                return agent_response

            from src.mcp.tools.product_tools import (
                extract_price_range,
            )

            try:
                price_range = (
                    extract_price_range(query)
                )
            except Exception:
                price_range = None

            max_price = (
                price_range[1]
                if price_range
                else None
            )

            header = (
                "Here are a few solid options I found"
            )

            if max_price is not None:
                header += (
                    f" under ₹{max_price:.0f}"
                )

            header += ":"

            lines = [header]

            lines.append(
                "Why these match: ranked by requirement match, price, "
                "rating, availability, and deal quality."
            )

            if max_price is not None:
                lines.append(
                    f"Budget filter applied: ≤ "
                    f"**₹{max_price:.0f}** "
                    "(when price was available)."
                )

            lines.append("")

            shown = 0

            def _clean_display_text(
                s: str,
            ) -> str:
                if not isinstance(
                    s,
                    str,
                ):
                    s = str(s)

                s = re.sub(
                    r"\s+",
                    " ",
                    s,
                ).strip()

                return "".join(
                    ch
                    for ch in s
                    if (
                        ord(ch) >= 32
                        and ord(ch) != 127
                    )
                )

            def _truncate(
                s: str,
                n: int,
            ) -> str:
                s = _clean_display_text(
                    s or ""
                )

                if len(s) <= n:
                    return s

                return (
                    s[: max(0, n - 3)]
                    .rstrip()
                    + "..."
                )

            for p in products:
                if shown >= max_items:
                    break

                name = _clean_display_text(
                    p.get("name")
                    or p.get("title")
                    or ""
                )

                if not name:
                    continue

                price = p.get("price")
                currency = str(
                    p.get("currency") or "INR"
                ).upper()

                price_str = ""

                if isinstance(
                    price,
                    (int, float),
                ):
                    if currency == "INR":
                        price_str = (
                            f"₹{float(price):.2f}"
                        )
                    else:
                        price_str = (
                            f"{currency} {float(price):.2f}"
                        )

                elif (
                    isinstance(price, str)
                    and price.strip()
                ):
                    price_str = (
                        _clean_display_text(
                            price.strip()
                        )
                    )

                url = _clean_display_text(
                    p.get("product_url")
                    or p.get("link")
                    or p.get("url")
                    or ""
                )

                link = "View Product"

                if isinstance(
                    url,
                    str,
                ) and url.startswith(
                    (
                        "http://",
                        "https://",
                    )
                ):
                    link = (
                        f"[View Product]({url})"
                    )

                bullet = (
                    f"{shown + 1}. **{name}**"
                )

                if price_str:
                    bullet += (
                        f" - {price_str}"
                    )

                lines.append(bullet)

                desc = (
                    p.get("description")
                    or ""
                )

                category = (
                    p.get("category")
                    or ""
                )

                what_parts = []

                if (
                    isinstance(
                        desc,
                        str,
                    )
                    and desc.strip()
                ):
                    what_parts.append(
                        _truncate(
                            desc,
                            140,
                        )
                    )

                elif (
                    isinstance(
                        category,
                        str,
                    )
                    and category.strip()
                    and category.strip().lower()
                    != "general"
                ):
                    what_parts.append(
                        "Category: "
                        + _truncate(
                            category,
                            60,
                        )
                    )

                rating = p.get("rating")
                reviews = p.get("reviews")

                try:
                    rating_val = (
                        float(rating)
                        if rating is not None
                        else 0.0
                    )
                except Exception:
                    rating_val = 0.0

                try:
                    reviews_val = (
                        int(reviews)
                        if reviews is not None
                        else 0
                    )
                except Exception:
                    reviews_val = 0

                if rating_val > 0:
                    if reviews_val > 0:
                        what_parts.append(
                            f"Rating: {rating_val:.1f}/5 "
                            f"({reviews_val} reviews)"
                        )
                    else:
                        what_parts.append(
                            f"Rating: "
                            f"{rating_val:.1f}/5"
                        )

                if what_parts:
                    lines.append(
                        "   - What it is: "
                        + " | ".join(
                            what_parts
                        )
                    )

                reasons = []

                deal_info = p.get("deal_info") or {}

                if deal_info.get("is_deal"):
                    savings = deal_info.get("savings_percent")
                    is_lowest = deal_info.get("is_lowest_price", False)

                    if is_lowest:
                        reasons.append("lowest price found")

                    if savings and float(savings) > 0:
                        reasons.append(
                            f"{float(savings):.1f}% below average"
                        )

                rating_value = p.get("rating")
                try:
                    if rating_value is not None and float(rating_value) >= 4.5:
                        reasons.append("high rating")
                except Exception:
                    pass

                if p.get("availability") or p.get("in_stock"):
                    reasons.append("available")

                if max_price is not None:
                    try:
                        price_val = (
                            self._parse_product_price(
                                p.get("price")
                            )
                        )
                    except Exception:
                        price_val = 0.0

                    if (
                        price_val
                        and price_val
                        <= float(max_price)
                    ):
                        reasons.append(
                            f"within budget "
                            f"(≤ ₹{float(max_price):.0f})"
                        )

                if reasons:
                    lines.append(
                        "   - Why it matches: "
                        + ", ".join(
                            reasons[:3]
                        )
                        + "."
                    )

                lines.append(
                    f"   - {link}"
                )

                lines.append("")

                shown += 1

            if shown == 0:
                return agent_response

            lines.append(
                "Want me to narrow it down "
                "(budget, hair/skin type, fragrance-free, etc.)?"
            )

            return "\n".join(
                lines
            ).strip()

        except Exception as e:
            logger.warning(
                f"Failed to ground response to products: {e}"
            )

            return agent_response

    async def _cache_llm_response(
        self,
        query: str,
        context_hash: str,
        agent_response: str,
        products: List[Dict],
        tools_used: List[str],
    ):
        """Cache LLM response."""

        response_data = {
            "response": agent_response,
            "products": products,
            "tools_used": tools_used,
        }

        await cache_service.set_llm_response(
            query,
            response_data,
            context_hash=context_hash,
        )

        try:
            await semantic_cache.set_semantic(
                query=query,
                response_data=response_data,
                context_hash=context_hash,
                ttl=settings.cache_llm_response_ttl,
            )

        except Exception as e:
            logger.debug(
                f"Failed to store in semantic cache: {e}"
            )

    def _validate_products_against_tools(
        self,
        products: List[Dict[str, Any]],
        tools_used: List[str],
    ) -> List[Dict[str, Any]]:
        """Validate products against tool results."""

        if not products:
            return []

        if "search_products" not in tools_used:
            logger.warning(
                "Products extracted but search_products "
                "tool wasn't used."
            )

            return products

        validated_products = []

        for product in products:
            if not product.get(
                "name"
            ) and not product.get(
                "title"
            ):
                logger.warning(
                    "Skipping product without name/title"
                )

                continue

            if (
                not product.get("source")
                and not product.get("id")
            ):
                logger.debug(
                    f"Product without source field: "
                    f"{product.get('name', 'Unknown')}"
                )

            validated_products.append(
                product
            )

        if len(validated_products) < len(
            products
        ):
            logger.warning(
                f"Filtered out "
                f"{len(products) - len(validated_products)} "
                "potentially hallucinated products"
            )

        return validated_products

    async def _ensure_products_with_price_filter(
        self,
        query: str,
        products: List[Dict[str, Any]],
        intent: Dict[str, Any],
        tools_used: List[str],
    ) -> List[Dict[str, Any]]:
        """Ensure products are available and match price filters."""

        # Product searches are authoritative for shopping queries.
        # Do not let the LLM skip the real search tool and return 0 products.
        is_shopping_query = bool(query and query.strip())

        if not is_shopping_query:
            return products

        from src.mcp.tools.product_tools import (
            extract_price_range,
        )

        price_range = extract_price_range(
            query
        )

        min_price = (
            price_range[0]
            if price_range
            else None
        )

        max_price = (
            price_range[1]
            if price_range
            else None
        )

        if not self._should_refetch_products(
            products,
            price_range,
        ):
            return products

        min_str = (
            f"₹{min_price:.2f}"
            if min_price is not None
            else "$0"
        )

        max_str = (
            f"₹{max_price:.2f}"
            if max_price is not None
            else "inf"
        )

        logger.info(
            f"Fetching products with price filter: "
            f"{min_str}-{max_str}"
        )

        fallback_products = (
            await self._refetch_products_with_price_filter(
                query,
                min_price,
                max_price,
            )
        )

        if not fallback_products:
            return products

        if (
            len(products) == 0
            or (
                price_range
                and len(fallback_products)
                > len(products)
            )
        ):
            products = fallback_products

        else:
            existing_ids = {
                p.get("id", "")
                for p in products
            }

            for product in fallback_products:
                product_id = product.get(
                    "id",
                    "",
                )

                if (
                    product_id
                    and product_id
                    not in existing_ids
                ):
                    products.append(
                        product
                    )

                    existing_ids.add(
                        product_id
                    )

        if (
            "search_products"
            not in tools_used
        ):
            tools_used.append(
                "search_products"
            )

        return products

    def _filter_products_by_brand(
        self,
        query: str,
        products: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter products by brand."""

        if not products:
            return products

        query_lower = query.lower()

        combined_context = query_lower

        for msg in history[-2:]:
            if msg.get("user_message"):
                combined_context += (
                    " "
                    + msg.get(
                        "user_message",
                        "",
                    ).lower()
                )

        excluded_brands = []

        exclusion_patterns = [
            r"without\s+(\w+)",
            r"not\s+(\w+)",
            r"exclude\s+(\w+)",
            r"don'?t\s+want\s+(\w+)",
            r"no\s+(\w+)",
            r"except\s+(\w+)",
        ]

        known_brands = [
            "nike",
            "adidas",
            "puma",
            "reebok",
            "bose",
            "sony",
            "apple",
            "samsung",
            "jbl",
        ]

        for pattern in exclusion_patterns:
            matches = re.findall(
                pattern,
                combined_context,
                re.IGNORECASE,
            )

            for match in matches:
                brand = match.strip()

                if (
                    len(brand) > 2
                    and (
                        brand[0].isupper()
                        or brand.lower()
                        in known_brands
                    )
                ):
                    excluded_brands.append(
                        brand.lower()
                    )

        included_brands = []

        inclusion_patterns = [
            r"only\s+(\w+)",
            r"show\s+me\s+(\w+)",
            r"(\w+)\s+brand",
            r"what\s+about\s+(\w+)",
            r"(\w+)\s+only",
        ]

        for pattern in inclusion_patterns:
            matches = re.findall(
                pattern,
                combined_context,
                re.IGNORECASE,
            )

            for match in matches:
                brand = match.strip()

                if (
                    len(brand) > 2
                    and (
                        brand[0].isupper()
                        or brand.lower()
                        in known_brands
                    )
                ):
                    included_brands.append(
                        brand.lower()
                    )

        if (
            not excluded_brands
            and not included_brands
        ):
            return products

        filtered_products = []

        for product in products:
            product_name = (
                product.get("name", "")
                + " "
                + product.get("title", "")
            ).lower()

            product_brand = (
                product.get(
                    "brand",
                    "",
                )
                or ""
            ).lower()

            should_exclude = False
            should_include = True

            for excluded in excluded_brands:
                if (
                    excluded in product_name
                    or excluded in product_brand
                ):
                    should_exclude = True
                    break

            if included_brands:
                should_include = False

                for included in included_brands:
                    if (
                        included in product_name
                        or included in product_brand
                    ):
                        should_include = True
                        break

            if (
                not should_exclude
                and should_include
            ):
                filtered_products.append(
                    product
                )

        return (
            filtered_products
            if filtered_products
            else products
        )

    async def _store_conversation(
        self,
        session_id: str,
        query: str,
        agent_response: str,
        tools_used: List[str],
        request_id: str,
    ):
        """Store conversation in history."""

        with latency_tracker.track_component(
            "conversation_storage",
            request_id,
        ):
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                lambda: conversation_store.add_conversation(
                    session_id=session_id,
                    user_message=query,
                    agent_response=agent_response,
                    tools_used=tools_used,
                ),
            )

            try:
                for limit in [
                    5,
                    10,
                    20,
                ]:
                    cache_key = (
                        f"conversation:"
                        f"{session_id}:"
                        f"{limit}"
                    )

                    await cache_service.delete(
                        cache_key
                    )

            except Exception as cache_err:
                logger.warning(
                    f"Failed to invalidate "
                    f"conversation cache: {cache_err}"
                )

    def _track_analytics(
        self,
        query: str,
        tools_used: List[str],
        agent_response: str,
        response_time: float,
        request_id: str,
    ):
        """Track analytics and decision metrics."""

        latency_tracker.request_times[
            request_id
        ]["total"] = response_time

        tracker.track_decision(
            query=query,
            tool_used=(
                ", ".join(tools_used)
                if tools_used
                else "no_tools"
            ),
            result=agent_response[:100],
            response_time=response_time,
        )


# Global agent instance
_shopping_agent_instance = None


def get_shopping_agent() -> ShoppingAgent:
    """Get or create shopping agent instance."""

    global _shopping_agent_instance

    if _shopping_agent_instance is None:
        try:
            _shopping_agent_instance = (
                ShoppingAgent()
            )

            logger.info(
                "Shopping agent initialized successfully"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize shopping agent: {e}"
            )

            raise

    return _shopping_agent_instance
