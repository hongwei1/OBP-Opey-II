import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException

logger = logging.getLogger('opey.service.middleware')


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses for debugging authentication issues"""

    async def dispatch(self, request: Request, call_next):
        # Log incoming request details. Never log header VALUES — Authorization,
        # Consent-Id and Cookie carry credentials. Names only.
        logger.debug(f"REQUEST_DEBUG: {request.method} {request.url}")
        logger.debug(f"REQUEST_DEBUG: Header names: {list(request.headers.keys())}")

        # Check for session cookie specifically
        session_cookie_value = request.cookies.get("session")
        logger.debug(f"REQUEST_DEBUG: Session cookie present: {bool(session_cookie_value)}")
        if session_cookie_value:
            logger.debug(f"REQUEST_DEBUG: Session cookie length: {len(session_cookie_value)}")

        try:
            response = await call_next(request)

            # Log response details. Response headers may include Set-Cookie
            # (the session id), so log names only.
            logger.debug(f"RESPONSE_DEBUG: Status {response.status_code}")
            logger.debug(f"RESPONSE_DEBUG: Header names: {list(response.headers.keys())}")

            # For error responses, try to log the body
            if response.status_code >= 400:
                logger.error(f"ERROR_RESPONSE_DEBUG: Status {response.status_code} for {request.url}")

            return response

        except HTTPException as exc:
            logger.error(f"HTTP_EXCEPTION_DEBUG: Status {exc.status_code}, Detail: {exc.detail}")
            logger.error(f"HTTP_EXCEPTION_DEBUG: Exception type: {type(exc)}")
            raise
        except Exception as exc:
            logger.error(f"UNEXPECTED_EXCEPTION_DEBUG: {type(exc)}: {str(exc)}")
            raise