import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)



def verbose_print(message, level='info', end='\n\n'):
    """
    Prints a message with color and style based on the level of verbosity.

    Parameters:
        message (str): The message to print.
        level (str): The verbosity level ('debug', 'info', 'warning', 'error', 'critical'). Default is 'info'.
        end (str): The end character to print after the message. Default is '\n'.
    """
    styles = {
        'debug': (Fore.CYAN, Style.DIM),
        'info': (Fore.GREEN, Style.NORMAL),
        'warning': (Fore.YELLOW, Style.BRIGHT),
        'error': (Fore.RED, Style.NORMAL),
        'critical': (Fore.WHITE, Back.RED, Style.BRIGHT)
    }

    color, *style = styles.get(level, (Fore.WHITE, Style.NORMAL))
    style = ''.join(style)
    print(f"{color}{style}{message}{Style.RESET_ALL}", end=end)