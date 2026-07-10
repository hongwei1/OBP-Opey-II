from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Global checkpointer dictionary
checkpointers = {}

def get_global_checkpointer() -> AsyncSqliteSaver:
    """
    Get the checkpointer for the app.

    Raises a clear error if the app lifespan has not yet initialized the shared
    AsyncSqliteSaver, instead of surfacing an opaque KeyError to the request.
    """
    saver = checkpointers.get('aiosql')
    if saver is None:
        raise RuntimeError(
            "Global checkpointer not initialized. The application lifespan must run "
            "(setting checkpointers['aiosql']) before any request is served."
        )
    return saver