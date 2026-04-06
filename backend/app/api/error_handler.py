"""Structured error handler middleware for the FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.models.schemas import ErrorResponse


class AppError(Exception):
    """Application-level error with structured error code."""

    def __init__(self, error_code: str, message: str, details: str | None = None, status_code: int = 400):
        self.error_code = error_code
        self.message = message
        self.details = details
        self.status_code = status_code
        super().__init__(message)


def register_error_handlers(app: FastAPI) -> None:
    """Register exception handlers on the FastAPI app."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        error = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump())

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        error = ErrorResponse(
            error_code="INVALID_CONFIG",
            message="Validation error",
            details=str(exc),
        )
        return JSONResponse(status_code=422, content=error.model_dump())

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        error = ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details=str(exc) if str(exc) else None,
        )
        return JSONResponse(status_code=500, content=error.model_dump())
