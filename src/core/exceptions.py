class LlmApiException(Exception):
    pass


def handle_api_exception(e: Exception, service_name: str = "AI service", is_overloaded: bool = False) -> LlmApiException:
    if isinstance(e, LlmApiException):
        return e

    if is_overloaded:
        return LlmApiException(
            f"All primary and fallback {service_name}s are currently overloaded. Please try again later."
        )

    error_str = str(e).lower()
    if "token" in error_str or "maximum context length" in error_str:
        return LlmApiException(f"The input/response exceeded the maximum token limit for the {service_name}.")
    elif any(auth_kw in error_str for auth_kw in ["401", "403", "invalid api key", "unauthorized", "forbidden"]):
        return LlmApiException(f"Authentication issue with the {service_name}. Please check API credentials.")
    else:
        return LlmApiException(f"An unexpected {service_name} error occurred: {str(e)}")
