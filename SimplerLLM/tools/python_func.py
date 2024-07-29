
import traceback
import sys
import io


def execute_python_code(input_code):
    """
    Executes a given Python code snippet and captures its standard output.

    Parameters:
    input_code (str): A string containing the Python code to be executed.

    Returns:
    tuple: A tuple containing two elements:
        - output (str): Captured standard output of the executed code if successful, None otherwise.
        - error_trace (str): Traceback of the exception if an error occurs, None otherwise.
    """
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(input_code, globals())
        # Reset standard output
        sys.stdout = old_stdout
        output = new_stdout.getvalue()
        return output, None
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, error_trace