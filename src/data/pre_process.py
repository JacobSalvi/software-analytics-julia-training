import swifter  # DO NOT REMOVE


def clean_docstring(docstring):
    """
    Clean docstring from any unwanted characters
    """
    docstring.strip().lower()
    #TODO: Add more cleaning steps
    return docstring


def clean_function_header(function_header):
    """
    Clean function header from any unwanted characters
    """
    function_header.strip()
    return function_header


def clean_function_body(function_body):
    """
    Clean function body from any unwanted characters
    """
    function_body.strip()
    return function_body


def process_data(data):
    data["doc_string"] = data["doc_string"].swifter.apply(clean_docstring)
    data["function_header"] = data["function_header"].swifter.apply(clean_function_header)
    data["function_body"] = data["function_body"].swifter.apply(clean_function_body)
    return data
