"""Langfuse client for monitoring and tracing."""

from typing import Optional, Dict, Any, List

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

from src.utils.config import settings
from src.analytics.logger import logger


class LangfuseClient:
    """Langfuse client wrapper for tracing and monitoring."""

    def __init__(self):
        """Initialize Langfuse client if enabled."""
        self.enabled = settings.langfuse_enabled
        self.client = None

        # Langfuse is optional. BUYorBYE currently has it disabled.
        if not self.enabled:
            logger.info("Langfuse is disabled in configuration")
            return

        if Langfuse is None:
            logger.warning(
                "Langfuse is enabled but the langfuse package is not installed. "
                "Disabling Langfuse."
            )
            self.enabled = False
            return

        try:
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                self.client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )

                # SDK compatibility:
                # older versions expose .trace/.span/.generation;
                # newer versions use start_span/start_generation.
                self._old_api = hasattr(self.client, "trace")

                logger.info(
                    f"Langfuse client initialized for project: "
                    f"{settings.langfuse_project_name}"
                )
            else:
                logger.warning(
                    "Langfuse keys not configured, disabling Langfuse"
                )
                self.enabled = False

        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self.enabled = False

    def trace(
        self,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a trace in Langfuse."""

        if not self.enabled or not self.client:
            return None

        try:
            # Old SDK API
            if getattr(self, "_old_api", False):
                return self.client.trace(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata or {},
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags or [],
                )

            # New SDK API
            trace_id = self.client.create_trace_id(
                seed=session_id or user_id
            )

            root = self.client.start_span(
                trace_context={"trace_id": trace_id},
                name=name,
                input=input,
                output=output,
                metadata=metadata or {},
            )

            try:
                root.end()
            except Exception:
                pass

            return {"id": trace_id}

        except Exception as e:
            logger.error(f"Failed to create Langfuse trace: {e}")
            return None

    def span(
        self,
        trace_id: str,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a span within a trace."""

        if not self.enabled or not self.client:
            return _NullSpanContext()

        return _SpanContext(
            client=self.client,
            old_api=getattr(self, "_old_api", False),
            trace_id=trace_id,
            name=name,
            input=input,
            output=output,
            metadata=metadata or {},
        )

    def generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a generation (LLM call) within a trace."""

        if not self.enabled or not self.client:
            return None

        try:
            if getattr(self, "_old_api", False):
                return self.client.generation(
                    trace_id=trace_id,
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage,
                    metadata=metadata or {},
                )

            gen = self.client.start_generation(
                trace_context={"trace_id": trace_id},
                name=name,
                model=model,
                input=input,
                output=output,
                metadata=metadata or {},
                usage_details=usage,
            )

            try:
                gen.end()
            except Exception:
                pass

            return {
                "id": getattr(gen, "id", None),
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(
                f"Failed to create Langfuse generation: {e}"
            )
            return None

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
    ):
        """Add a score to a trace."""

        if not self.enabled or not self.client:
            return None

        try:
            if hasattr(self.client, "create_score"):
                self.client.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment,
                    data_type=data_type,
                )

                return {
                    "success": True,
                    "trace_id": trace_id,
                    "name": name,
                    "value": value,
                }

            elif hasattr(self.client, "score_current_trace"):
                logger.warning(
                    f"Using score_current_trace fallback for {name}"
                )

                return self.client.score_current_trace(
                    name=name,
                    value=value,
                    comment=comment,
                )

            else:
                logger.warning(
                    "Langfuse client has no score method"
                )
                return None

        except Exception as e:
            logger.error(
                f"Failed to create Langfuse score "
                f"'{name}' for trace {trace_id}: {e}"
            )
            return None

    def update_trace(
        self,
        trace_id: str,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update an existing trace."""

        if not self.enabled or not self.client or not trace_id:
            return None

        try:
            if output or metadata:
                with self.span(
                    trace_id=trace_id,
                    name="final_output",
                    output=output,
                    metadata=metadata,
                ):
                    pass

            return True

        except Exception as e:
            logger.error(
                f"Failed to update Langfuse trace: {e}"
            )
            return None

    def flush(self):
        """Flush pending events to Langfuse."""

        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(
                    f"Failed to flush Langfuse events: {e}"
                )


class _SpanContext:
    """Context manager for Langfuse spans."""

    def __init__(
        self,
        client,
        old_api: bool,
        trace_id: str,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.old_api = old_api
        self.trace_id = trace_id
        self.name = name
        self.input = input
        self.output = output
        self.metadata = metadata
        self.span_obj = None

    def __enter__(self):
        """Start the span."""

        try:
            if self.old_api:
                self.span_obj = self.client.span(
                    trace_id=self.trace_id,
                    name=self.name,
                    input=self.input,
                    output=None,
                    metadata=self.metadata or {},
                )
            else:
                self.span_obj = self.client.start_span(
                    trace_context={"trace_id": self.trace_id},
                    name=self.name,
                    input=self.input,
                    output=None,
                    metadata=self.metadata or {},
                )

        except Exception as e:
            logger.error(
                f"Failed to start Langfuse span "
                f"'{self.name}': {e}"
            )
            self.span_obj = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span."""

        if self.span_obj:
            try:
                if self.output:
                    if hasattr(self.span_obj, "update"):
                        self.span_obj.update(
                            output=self.output
                        )

                if hasattr(self.span_obj, "end"):
                    self.span_obj.end(
                        output=self.output
                    )
                elif hasattr(self.span_obj, "update"):
                    self.span_obj.update(
                        output=self.output
                    )

            except Exception as e:
                logger.debug(
                    f"Failed to end Langfuse span "
                    f"'{self.name}': {e}"
                )

        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        return self.__exit__(
            exc_type,
            exc_val,
            exc_tb,
        )

    def update(
        self,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update span output or metadata."""

        if self.span_obj:
            try:
                if output:
                    self.output = output

                if metadata:
                    self.metadata = {
                        **(self.metadata or {}),
                        **metadata,
                    }

                if hasattr(self.span_obj, "update"):
                    self.span_obj.update(
                        output=output,
                        metadata=metadata,
                    )

            except Exception as e:
                logger.debug(
                    f"Failed to update Langfuse span "
                    f"'{self.name}': {e}"
                )


class _NullSpanContext:
    """Null context manager when Langfuse is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        return False

    def update(
        self,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        pass


# Global Langfuse client instance
langfuse_client = LangfuseClient()
