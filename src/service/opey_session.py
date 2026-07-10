from typing import Annotated

from auth.session import session_verifier, SessionData, session_cookie
from auth.auth import OBPConsentAuth, OBPBearerAuth
from auth.usage_tracker import usage_tracker
from fastapi import Depends, Request
from uuid import UUID

from agent.graph_builder import OpeyAgentGraphBuilder, create_basic_opey_graph
from agent.components.tools import create_approval_store
from service.mcp_tools_cache import get_mcp_tools, get_mcp_tools_with_auth, get_auth_required_servers

from service.checkpointer import get_global_checkpointer
from service.redis_client import get_redis_client
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables.graph import MermaidDrawMethod


import os
import hashlib
import logging

from langchain_core.tools import BaseTool, StructuredTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('opey.session')

# HTTP methods considered read-only. In SAFE mode any tool call carrying a
# `method` argument outside this set is refused before it reaches the OBP-API.
_READ_ONLY_HTTP_METHODS = {"GET", "HEAD", "OPTIONS"}


def compute_thread_namespace(session_id) -> str:
    """Per-session, non-reversible namespace derived from the session id.

    Pure function so control endpoints (e.g. stream stop) can confine a
    thread_id using only the session cookie, without building the agent graph.
    """
    return hashlib.sha256(str(session_id).encode()).hexdigest()[:16]


def compute_effective_thread_id(session_id, client_thread_id: str | None) -> str:
    """Map a client-supplied thread_id into a session's private namespace.

    Idempotent: an already-namespaced id is returned unchanged. A thread_id
    belonging to another session cannot be reached because the caller cannot
    reproduce that session's namespace prefix.
    """
    prefix = f"{compute_thread_namespace(session_id)}::"
    if not client_thread_id:
        return f"{prefix}default"
    if client_thread_id.startswith(prefix):
        return client_thread_id
    return f"{prefix}{client_thread_id}"


def _make_readonly_tool(tool: BaseTool) -> BaseTool:
    """Wrap a tool so that, in SAFE mode, any call with a non-read HTTP method
    is refused at execution time instead of hitting the OBP-API.

    This is a hard, server-side gate for the generic `obp_requests(method, path, …)`
    style tool where the model supplies an arbitrary HTTP method. Tools that do
    not accept a `method` argument are returned unchanged (there is nothing to
    gate on generically); per-endpoint write tools, if any are exposed by the
    MCP server, must still be gated server-side by that server.
    """
    try:
        arg_names = set((getattr(tool, "args", None) or {}).keys())
    except Exception:
        arg_names = set()

    if "method" not in arg_names:
        return tool

    original = tool

    async def _guarded(**kwargs):
        method = kwargs.get("method")
        if method and str(method).strip().upper() not in _READ_ONLY_HTTP_METHODS:
            logger.warning(
                f"SAFE mode blocked non-read tool call: {original.name} "
                f"method={method} path={kwargs.get('path')}"
            )
            return (
                f"BLOCKED: '{method}' requests are not permitted in SAFE (read-only) mode. "
                f"Only GET/read operations are allowed. Do not retry this write operation; "
                f"tell the user it is not available in read-only mode."
            )
        return await original.ainvoke(kwargs)

    return StructuredTool(
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        coroutine=_guarded,
        metadata=getattr(original, "metadata", None),
    )

class OpeySession:
    """
    Class to manage Opey sessions.
    Depends first on the authentication layer, i.e. session_verifier
    """
    def __init__(self, request: Request, session_data: Annotated[SessionData, Depends(session_verifier)], session_id: Annotated[UUID, Depends(session_cookie)], checkpointer: Annotated[BaseCheckpointSaver, Depends(get_global_checkpointer)]):
        # Store session data and check usage limits for anonymous sessions
        self.session_data = session_data
        self.session_id = session_id
        # Note: Usage limits will be checked when methods are called

        # Store session data in request state for middleware to update
        request.state.session_data = session_data
        request.state.session_id = session_id

        # Get consent_id from the session data (None for anonymous sessions)
        self.consent_id = session_data.consent_id
        self.is_anonymous = session_data.is_anonymous
        
        # Set up the model used
        self._setup_model()
        
        # Initialize simplified approval store
        redis_client = get_redis_client() if os.getenv("REDIS_URL") else None
        user_id = str(session_id)  # Use session_id as user identifier
        self.approval_store = create_approval_store(
            session_id=str(session_id),
            user_id=user_id,
            redis_client=redis_client,
        )
        
        # Initialize auth object only if not anonymous
        if not self.is_anonymous:
            if session_data.bearer_token:
                self.auth = OBPBearerAuth(bearer_token=session_data.bearer_token)
            elif self.consent_id:
                self.auth = OBPConsentAuth(consent_id=self.consent_id)

        # Store bearer token for MCP authentication
        self._bearer_token = session_data.bearer_token
        
        # Store dependencies for async initialization
        self._checkpointer = checkpointer

        obp_api_mode = os.getenv("OBP_API_MODE")

        # For anonymous sessions, limit to SAFE or NONE modes only
        if self.is_anonymous and obp_api_mode in ["DANGEROUS", "TEST"]:
            logger.warning(f"Anonymous session attempted to use {obp_api_mode} mode. Defaulting to SAFE mode.")
            obp_api_mode = "SAFE"
        
        self._obp_api_mode = obp_api_mode
        self.graph = None  # Will be initialized in async_init()
        self._tools = []  # Will be populated in async_init()

    async def async_init(self, bearer_token: str | None = None) -> "OpeySession":
        """
        Async initialization for components that require async setup.
        
        This handles loading MCP tools with bearer token authentication when needed.
        Must be called after __init__ before using the session.
        
        Args:
            bearer_token: OAuth bearer token for MCP server authentication.
                         If provided, overrides the token stored in session_data.
        
        Returns:
            self for chaining
        """
        # Use provided token, fall back to stored session token
        token = bearer_token if bearer_token is not None else self._bearer_token
        
        logger.info(f"DEBUG: token value = {token[:20] if token else 'None'}... (type: {type(token)})")
        
        # Get tools - use authenticated if bearer token present and servers require it
        auth_servers = get_auth_required_servers()
        if token and auth_servers:
            logger.info(f"Loading MCP tools with bearer token for servers: {auth_servers}")
            tools = await get_mcp_tools_with_auth(token)
        else:
            # Use cached tools from startup
            tools = get_mcp_tools()
        
        if not tools:
            logger.warning("No MCP tools available - agent will have limited capabilities")

        # SAFE mode is read-only: hard-gate every method-bearing tool so a
        # non-GET call is refused before it reaches the OBP-API, regardless of
        # what the prompt or MCP server does. Applies to explicit SAFE and to
        # anonymous sessions (which are force-downgraded to SAFE above).
        if self._obp_api_mode == "SAFE":
            tools = [_make_readonly_tool(t) for t in tools]
            logger.info("SAFE mode: applied read-only gate to method-bearing tools")

        # Store tools for consent retry (tools_by_name in config)
        self._tools = tools

        # Prepare prompt addition for no-tool scenario
        no_tools_prompt = None
        if not tools:
            no_tools_prompt = (
                "\n\n=== IMPORTANT: NO TOOLS AVAILABLE ===\n"
                "Your API tools are currently unavailable. Do NOT fabricate tool calls, simulate API responses, "
                "or generate fake data. When users ask for API operations or live data:\n"
                "1. Inform them your tools are currently unavailable\n"
                "2. Suggest they check their authentication or try again later\n"
                "3. You may provide general information about OBP API concepts from your training, "
                "but clearly state you cannot access live data or perform actions right now.\n"
                "=================================="
            )

        # Initialize the graph with the appropriate tools based on the OBP API mode
        match self._obp_api_mode:
            case "NONE":
                logger.info("OBP API mode set to NONE: Calls to the OBP-API will not be available")
                builder = (OpeyAgentGraphBuilder()
                          .with_tools(tools)
                          .with_model(self._model_name, temperature=0.5)
                          .with_checkpointer(self._checkpointer)
                          .enable_human_review(False))
                if no_tools_prompt:
                    builder.add_to_system_prompt(no_tools_prompt)
                self.graph = builder.build()

            case "SAFE":
                if self.is_anonymous:
                    logger.info("Anonymous session using SAFE mode: Only GET requests to OBP-API will be available")
                    prompt_addition = "Note: This is an anonymous session with limited capabilities. User can only make GET requests to the OBP-API. Ensure all responses adhere to this restriction."
                    builder = (OpeyAgentGraphBuilder()
                              .with_tools(tools)
                              .with_model(self._model_name, temperature=0.5)
                              .add_to_system_prompt(prompt_addition)
                              .with_checkpointer(self._checkpointer)
                              .enable_human_review(False))
                    if no_tools_prompt:
                        builder.add_to_system_prompt(no_tools_prompt)
                    self.graph = builder.build()
                else:
                    logger.info("OBP API mode set to SAFE: GET requests to the OBP-API will be available")
                    builder = (OpeyAgentGraphBuilder()
                              .with_tools(tools)
                              .with_model(self._model_name, temperature=0.5)
                              .with_checkpointer(self._checkpointer)
                              .enable_human_review(False))
                    if no_tools_prompt:
                        builder.add_to_system_prompt(no_tools_prompt)
                    self.graph = builder.build()

            case "DANGEROUS":
                logger.info("OBP API mode set to DANGEROUS: All requests to the OBP-API will be available (consent handled by MCP server).")
                builder = (OpeyAgentGraphBuilder()
                          .with_tools(tools)
                          .with_model(self._model_name, temperature=0.5)
                          .with_checkpointer(self._checkpointer)
                          .enable_human_review(False))  # Consent flow handles authorization
                if no_tools_prompt:
                    builder.add_to_system_prompt(no_tools_prompt)
                self.graph = builder.build()

            case "TEST":
                logger.info("OBP API mode set to TEST: All requests to the OBP-API will be available AND WILL BE APPROVED BY DEFAULT.")
                test_prompt = "You are in TEST mode. Operations will be auto-approved. DO NOT USE IN PRODUCTION."
                builder = (OpeyAgentGraphBuilder()
                          .with_tools(tools)
                          .add_to_system_prompt(test_prompt)
                          .with_model(self._model_name, temperature=0.5)
                          .with_checkpointer(self._checkpointer)
                          .enable_human_review(False))
                if no_tools_prompt:
                    builder.add_to_system_prompt(no_tools_prompt)
                self.graph = builder.build()

            case _:
                logger.error(f"OBP API mode set to {self._obp_api_mode}: Unknown OBP API mode. Defaulting to NONE.")
                self.graph = create_basic_opey_graph(tools)
                self.graph.checkpointer = self._checkpointer

        self.graph.checkpointer = self._checkpointer
        return self
        
    def _setup_model(self):
        """
        Set up the model for the session.
        """
        from agent.utils.model_factory import LLMProviders, get_available_models
        model_provider = os.getenv("MODEL_PROVIDER")
        if not model_provider:
            raise ValueError("MODEL_PROVIDER environment variable must be set")
        
        if not (model_provider := model_provider.lower()) in [provider.value for provider in LLMProviders]:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {model_provider}. Supported providers: {[provider.value for provider in LLMProviders]}")
        
        try:
            
            available_models = get_available_models(LLMProviders(model_provider))
        except RuntimeError as e:
            logger.error(f"Error checking available models: {e}")
            raise
        
        logger.info(f"Using model provider: {model_provider}")
        logger.info(f"Available models for provider {model_provider}: {available_models}")
        
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME environment variable must be set")
        
        if model_name not in available_models:
            raise ValueError(f"MODEL_NAME {model_name} is not available for provider {model_provider}. Available models: {available_models}")
        
        logger.info(f"Using model: {model_name}")
        self._model_name = model_name

    def effective_thread_id(self, client_thread_id: str | None) -> str:
        """Map a client-supplied thread_id into this session's private namespace.

        Idempotent: an already-namespaced id is returned unchanged, so the value
        handed back to the client (via X-Thread-ID) can be sent straight back on
        the next request without being double-prefixed. A thread_id belonging to
        another session cannot be reached because the caller cannot reproduce
        that session's namespace prefix.
        """
        return compute_effective_thread_id(self.session_id, client_thread_id)

    def enforce_limits(self) -> None:
        """Enforce anonymous-session usage limits, raising HTTP 429 if exceeded.

        Must be called before doing work on behalf of an anonymous session so a
        client cannot bypass token/request caps by never triggering a check.
        """
        usage_tracker.check_limits(self.session_data)

    def build_config(self, base_config: dict | None = None) -> dict:
        """
        Build a complete RunnableConfig by merging base session config with runtime config.
        This ensures model context is available to all nodes without clashing with
        service-level config like thread_id.
        
        Args:
            base_config: Optional config dict from service endpoints (e.g., with thread_id)
        
        Returns:
            Merged config dict with all necessary context
        
        Example:
            # In service.py:
            config = opey_session.build_config({'configurable': {'thread_id': thread_id}})
        """
        base_config = base_config or {}
        
        # Session-level configuration (model context, approval store)
        session_configurable = {
            "model_name": self._model_name,
            "model_kwargs": {},  # Add model_kwargs if needed in future
            "approval_store": self.approval_store,
            "tools_by_name": {t.name: t for t in self._tools},
        }
        
        # Merge: base config takes precedence for runtime values like thread_id
        merged_configurable = {
            **session_configurable,
            **base_config.get("configurable", {})
        }

        # Enforce session-private thread namespacing at the single chokepoint every
        # route funnels through. Even if a caller passes a raw or foreign thread_id,
        # the checkpointer key is confined to this session's namespace.
        if "thread_id" in merged_configurable:
            merged_configurable["thread_id"] = self.effective_thread_id(
                merged_configurable["thread_id"]
            )

        # LangGraph caps graph execution at `recursion_limit` super-steps (default 25).
        # Opey's tool-call cycle is ~5 super-steps per round, so 25 gives ~5 tool
        # calls per turn. Override via OPEY_RECURSION_LIMIT env var.
        return {
            "recursion_limit": int(os.getenv("OPEY_RECURSION_LIMIT", "100")),
            **base_config,
            "configurable": merged_configurable
        }

    def update_token_usage(self, token_count: int) -> None:
        """
        Update token usage for the session.

        Args:
            token_count: Number of tokens used
        """
        if self.is_anonymous:
            usage_tracker.update_token_usage(self.session_data, token_count)

    def update_request_count(self) -> None:
        """
        Update request count for the session.
        """
        if self.is_anonymous:
            usage_tracker.update_request_count(self.session_data)

    def get_usage_info(self) -> dict:
        """
        Get usage information for the session.

        Returns:
            Dictionary containing usage information
        """
        return usage_tracker.get_usage_info(self.session_data)

    def get_threads_for_user(self):
        """
        Get the threads for the user
        Returns:
            List of threads for the user
        """
        raise NotImplementedError("This method is not implemented yet")


    def generate_mermaid_diagram(self, path: str):
        """
        Generate a mermaid diagram from the agent graph
        path (str): The path to save the diagram
        """
        try:
            if os.path.exists(path):
                os.remove(path)
            graph_png = self.graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
                output_file_path=path,
            )
            return graph_png
        except Exception as e:
            print("Error generating mermaid diagram:", e)
            return None
