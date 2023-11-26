class OpenAIRequestError(Exception):
    """Exception raised for errors in the request to OpenAI API."""
    pass

class OpenAIResponseParsingError(Exception):
    """Exception raised for errors in parsing the response from OpenAI API."""
    pass