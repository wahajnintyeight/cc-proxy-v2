import logging


class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


def configure_logging(logger_name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.addFilter(MessageFilter())
    return logging.getLogger(logger_name)


def _provider_from_model(model_name: str) -> str:
    if not model_name:
        return "unknown"
    if "/" in model_name:
        return model_name.split("/", 1)[0]
    return "anthropic"


def _model_without_prefix(model_name: str) -> str:
    if not model_name:
        return "unknown"
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def log_request_beautifully(
    method: str,
    path: str,
    requested_model: str,
    routed_model: str,
    num_messages: int,
    num_tools: int,
    status_code: int,
    response_time_ms: float,
) -> None:
    """Log requests in plain, structured, easy-to-scan format."""
    endpoint = path.split("?")[0] if "?" in path else path
    provider = _provider_from_model(routed_model)
    requested = _model_without_prefix(requested_model)
    routed = _model_without_prefix(routed_model)
    print(
        (
            f"{method} {endpoint} status={status_code} provider={provider} "
            f"requested_model={requested} routed_model={routed} "
            f"messages={num_messages} tools={num_tools} response_ms={response_time_ms:.2f}"
        )
    )
