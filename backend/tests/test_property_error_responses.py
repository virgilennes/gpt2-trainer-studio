"""Property test: Structured error responses (Property 13).

**Validates: Requirements 10.3, 1.6, 2.6, 5.5, 6.7**

For any API error response from the backend, the response body should contain
an `error_code` field (non-empty string) and a `message` field (non-empty string).
"""

import pytest
from hypothesis import given, settings, strategies as st
from fastapi.testclient import TestClient

from backend.app.models.schemas import ErrorResponse
from backend.app.api.error_handler import AppError
from backend.app.main import app

# Known error codes from the design document
ERROR_CODES = [
    "MODEL_LOAD_FAILED",
    "MODEL_NOT_LOADED",
    "DATASET_DOWNLOAD_FAILED",
    "DATASET_NOT_PREPARED",
    "TRAINING_ERROR",
    "TRAINING_ALREADY_RUNNING",
    "INVALID_CONFIG",
    "EVALUATION_ERROR",
    "GENERATION_ERROR",
    "WS_CONNECTION_ERROR",
    "DEMO_STEP_FAILED",
    "INTERNAL_ERROR",
]


class TestErrorResponseModel:
    """Property tests for the ErrorResponse model itself."""

    @given(
        error_code=st.sampled_from(ERROR_CODES),
        message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_error_response_has_non_empty_error_code_and_message(
        self, error_code: str, message: str
    ) -> None:
        """**Validates: Requirements 10.3, 1.6, 2.6, 5.5, 6.7**

        For any ErrorResponse, error_code and message are non-empty strings.
        """
        resp = ErrorResponse(error_code=error_code, message=message)
        assert isinstance(resp.error_code, str)
        assert len(resp.error_code) > 0
        assert isinstance(resp.message, str)
        assert len(resp.message) > 0


class TestErrorHandlerMiddleware:
    """Property tests verifying the error handler returns structured responses."""

    @given(
        error_code=st.sampled_from(ERROR_CODES),
        message=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
        status_code=st.sampled_from([400, 404, 409, 422, 500]),
    )
    @settings(max_examples=100)
    def test_app_error_returns_structured_response(
        self, error_code: str, message: str, status_code: int
    ) -> None:
        """**Validates: Requirements 10.3**

        For any AppError raised, the response contains non-empty error_code and message.
        """
        # Create a temporary route that raises the error
        route_path = f"/test-error-{error_code}-{status_code}"

        # Remove any previously added test route to avoid duplicates
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != route_path]

        @app.get(route_path)
        async def raise_error():
            raise AppError(
                error_code=error_code,
                message=message,
                status_code=status_code,
            )

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(route_path)

        data = response.json()
        assert "error_code" in data
        assert "message" in data
        assert isinstance(data["error_code"], str)
        assert len(data["error_code"]) > 0
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

        # Cleanup
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != route_path]

    def test_generic_exception_returns_structured_response(self) -> None:
        """Unhandled exceptions also produce structured error responses."""
        route_path = "/test-generic-error"
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != route_path]

        @app.get(route_path)
        async def raise_generic():
            raise RuntimeError("something went wrong")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get(route_path)

        data = response.json()
        assert "error_code" in data
        assert "message" in data
        assert isinstance(data["error_code"], str)
        assert len(data["error_code"]) > 0
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0

        # Cleanup
        app.routes[:] = [r for r in app.routes if getattr(r, "path", None) != route_path]
