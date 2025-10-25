"""
Pattern extraction utilities for SimplerLLM.

This module provides regex-based pattern extraction, validation, and normalization
for common data types like emails, phone numbers, URLs, dates, and more.
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
from urllib.parse import urlparse


# ==============================================================================
# PREDEFINED REGEX PATTERNS
# ==============================================================================

PREDEFINED_PATTERNS = {
    # Email - RFC 5322 compliant (simplified)
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

    # Phone - Multiple formats (US and International)
    "phone": r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})",

    # URL - HTTP/HTTPS URLs
    "url": r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)",

    # Date - Multiple formats (ISO, US, EU)
    "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",

    # Time - 12 and 24 hour formats
    "time": r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\s?[APap][Mm])?\b",

    # SSN - US Social Security Number
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",

    # Credit Card - Major providers (Visa, MasterCard, Amex, Discover)
    "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",

    # Zip Code - US 5 or 9 digit
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",

    # IPv4 Address
    "ipv4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",

    # IPv6 Address (simplified)
    "ipv6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",

    # Currency - Dollar amounts
    "currency": r"\$?\s?\d{1,3}(?:,?\d{3})*(?:\.\d{2})?\b",

    # Hex Color - #RGB or #RRGGBB
    "hex_color": r"#(?:[0-9a-fA-F]{3}){1,2}\b",

    # Username - Alphanumeric with underscores/hyphens
    "username": r"@?[a-zA-Z0-9_-]{3,20}\b",

    # Hashtag
    "hashtag": r"#[a-zA-Z0-9_]+\b",

    # File path - Windows and Unix
    "filepath": r"(?:[a-zA-Z]:\\|/)(?:[^\\/:*?\"<>|\r\n]+[\\\/])*[^\\/:*?\"<>|\r\n]*",
}


# ==============================================================================
# PATTERN EXTRACTION FUNCTIONS
# ==============================================================================

def get_predefined_pattern(pattern_name: str) -> Optional[str]:
    """
    Get a predefined regex pattern by name.

    Args:
        pattern_name: Name of the pattern (e.g., 'email', 'phone', 'url')

    Returns:
        The regex pattern string, or None if not found

    Example:
        >>> get_predefined_pattern('email')
        '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
    """
    return PREDEFINED_PATTERNS.get(pattern_name.lower())


def extract_pattern_from_text(
    text: str,
    pattern: str,
    extract_all: bool = False,
    flags: int = 0
) -> List[Dict[str, Union[str, int]]]:
    """
    Extract pattern matches from text with position information.

    Args:
        text: The text to search in
        pattern: The regex pattern to search for
        extract_all: If True, extract all matches; if False, only first match
        flags: Optional regex flags (e.g., re.IGNORECASE)

    Returns:
        List of dictionaries with 'value' and 'position' keys

    Example:
        >>> extract_pattern_from_text("Email: john@example.com", r"\\S+@\\S+", extract_all=False)
        [{'value': 'john@example.com', 'position': 7}]
    """
    matches = []

    try:
        if extract_all:
            for match in re.finditer(pattern, text, flags):
                matches.append({
                    'value': match.group(),
                    'position': match.start()
                })
        else:
            match = re.search(pattern, text, flags)
            if match:
                matches.append({
                    'value': match.group(),
                    'position': match.start()
                })
    except re.error as e:
        # Invalid regex pattern
        return []

    return matches


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address beyond basic regex matching.

    Args:
        email: The email address to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_email("user@example.com")
        (True, "Valid email format")
    """
    # Check for @ symbol
    if email.count('@') != 1:
        return False, "Email must contain exactly one @ symbol"

    local, domain = email.split('@')

    # Validate local part
    if not local or len(local) > 64:
        return False, "Local part must be 1-64 characters"

    # Validate domain part
    if not domain or len(domain) > 255:
        return False, "Domain must be 1-255 characters"

    # Check for valid TLD (at least 2 characters)
    if '.' not in domain:
        return False, "Domain must contain a dot"

    tld = domain.split('.')[-1]
    if len(tld) < 2:
        return False, "Top-level domain must be at least 2 characters"

    return True, "Valid email format"


def validate_phone(phone: str) -> Tuple[bool, str]:
    """
    Validate a phone number format.

    Args:
        phone: The phone number to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_phone("(555) 123-4567")
        (True, "Valid phone format")
    """
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)

    # US phone numbers should have 10 or 11 digits (with country code)
    if len(digits) == 10:
        return True, "Valid 10-digit phone number"
    elif len(digits) == 11 and digits[0] == '1':
        return True, "Valid phone number with country code"
    else:
        return False, f"Phone number has {len(digits)} digits, expected 10 or 11"


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate a URL beyond basic regex matching.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_url("https://example.com")
        (True, "Valid URL format")
    """
    try:
        result = urlparse(url)

        # Check for scheme (http/https)
        if not result.scheme or result.scheme not in ['http', 'https']:
            return False, "URL must have http or https scheme"

        # Check for netloc (domain)
        if not result.netloc:
            return False, "URL must have a valid domain"

        # Check for valid domain structure
        if '.' not in result.netloc:
            return False, "Domain must contain a dot"

        return True, "Valid URL format"
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"


def validate_date(date_str: str) -> Tuple[bool, str]:
    """
    Validate and parse a date string.

    Args:
        date_str: The date string to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_date("2025-01-15")
        (True, "Valid date: 2025-01-15")
    """
    # Try common date formats
    date_formats = [
        '%Y-%m-%d',      # ISO format
        '%m/%d/%Y',      # US format
        '%d-%m-%Y',      # EU format
        '%B %d, %Y',     # January 15, 2025
        '%b %d, %Y',     # Jan 15, 2025
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt)
            return True, f"Valid date: {parsed_date.strftime('%Y-%m-%d')}"
        except ValueError:
            continue

    return False, "Date format not recognized"


def validate_ssn(ssn: str) -> Tuple[bool, str]:
    """
    Validate a US Social Security Number.

    Args:
        ssn: The SSN to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_ssn("123-45-6789")
        (True, "Valid SSN format")
    """
    # Remove hyphens
    digits = ssn.replace('-', '')

    if len(digits) != 9:
        return False, "SSN must have 9 digits"

    # Check for invalid SSN patterns
    if digits.startswith('000') or digits.startswith('666') or digits.startswith('9'):
        return False, "Invalid SSN: first 3 digits cannot be 000, 666, or 9xx"

    if digits[3:5] == '00':
        return False, "Invalid SSN: middle 2 digits cannot be 00"

    if digits[5:9] == '0000':
        return False, "Invalid SSN: last 4 digits cannot be 0000"

    return True, "Valid SSN format"


def validate_credit_card(card_number: str) -> Tuple[bool, str]:
    """
    Validate a credit card number using Luhn algorithm.

    Args:
        card_number: The credit card number to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_credit_card("4532015112830366")
        (True, "Valid credit card (Visa)")
    """
    # Remove spaces and hyphens
    digits = re.sub(r'[\s-]', '', card_number)

    if not digits.isdigit():
        return False, "Credit card must contain only digits"

    # Luhn algorithm
    def luhn_check(card_num):
        total = 0
        reverse_digits = card_num[::-1]
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    if not luhn_check(digits):
        return False, "Credit card failed Luhn check"

    # Identify card type
    card_type = "Unknown"
    if digits.startswith('4'):
        card_type = "Visa"
    elif digits.startswith(('51', '52', '53', '54', '55')):
        card_type = "MasterCard"
    elif digits.startswith(('34', '37')):
        card_type = "American Express"
    elif digits.startswith('6011') or digits.startswith('65'):
        card_type = "Discover"

    return True, f"Valid credit card ({card_type})"


def validate_zip_code(zip_code: str) -> Tuple[bool, str]:
    """
    Validate a US zip code.

    Args:
        zip_code: The zip code to validate

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_zip_code("12345")
        (True, "Valid 5-digit zip code")
    """
    digits = re.sub(r'\D', '', zip_code)

    if len(digits) == 5:
        return True, "Valid 5-digit zip code"
    elif len(digits) == 9:
        return True, "Valid 9-digit zip code (ZIP+4)"
    else:
        return False, f"Zip code has {len(digits)} digits, expected 5 or 9"


def validate_ip_address(ip: str, version: int = 4) -> Tuple[bool, str]:
    """
    Validate an IP address.

    Args:
        ip: The IP address to validate
        version: IP version (4 or 6)

    Returns:
        Tuple of (is_valid: bool, message: str)

    Example:
        >>> validate_ip_address("192.168.1.1")
        (True, "Valid IPv4 address")
    """
    if version == 4:
        parts = ip.split('.')
        if len(parts) != 4:
            return False, "IPv4 address must have 4 octets"

        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False, f"Invalid octet value: {num}"
            return True, "Valid IPv4 address"
        except ValueError:
            return False, "IPv4 octets must be numeric"

    elif version == 6:
        # Basic IPv6 validation (simplified)
        parts = ip.split(':')
        if len(parts) < 3 or len(parts) > 8:
            return False, "Invalid IPv6 format"

        for part in parts:
            if part and not all(c in '0123456789abcdefABCDEF' for c in part):
                return False, "IPv6 parts must be hexadecimal"

        return True, "Valid IPv6 address"

    return False, "IP version must be 4 or 6"


# ==============================================================================
# NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_email(email: str) -> str:
    """
    Normalize an email address.

    Args:
        email: The email to normalize

    Returns:
        Normalized email (lowercase, trimmed)

    Example:
        >>> normalize_email("  John.Doe@EXAMPLE.COM  ")
        'john.doe@example.com'
    """
    return email.strip().lower()


def normalize_phone(phone: str, format: str = 'E164') -> str:
    """
    Normalize a phone number to a standard format.

    Args:
        phone: The phone number to normalize
        format: Output format ('E164', 'US', 'DOTS', 'DASHES')

    Returns:
        Normalized phone number

    Example:
        >>> normalize_phone("(555) 123-4567", format='E164')
        '+15551234567'
    """
    # Extract only digits
    digits = re.sub(r'\D', '', phone)

    # Add country code if missing (assume US)
    if len(digits) == 10:
        digits = '1' + digits

    if format == 'E164':
        return f'+{digits}'
    elif format == 'US':
        return f'({digits[1:4]}) {digits[4:7]}-{digits[7:11]}'
    elif format == 'DOTS':
        return f'{digits[1:4]}.{digits[4:7]}.{digits[7:11]}'
    elif format == 'DASHES':
        return f'{digits[1:4]}-{digits[4:7]}-{digits[7:11]}'
    else:
        return digits


def normalize_url(url: str) -> str:
    """
    Normalize a URL.

    Args:
        url: The URL to normalize

    Returns:
        Normalized URL (lowercase domain, with protocol)

    Example:
        >>> normalize_url("EXAMPLE.COM/Path")
        'https://example.com/Path'
    """
    url = url.strip()

    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Parse and normalize
    parsed = urlparse(url)

    # Lowercase the domain
    normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"

    # Add path, query, etc.
    if parsed.path:
        normalized += parsed.path
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"

    return normalized


def normalize_date(date_str: str, format: str = 'ISO8601') -> Optional[str]:
    """
    Normalize a date to a standard format.

    Args:
        date_str: The date string to normalize
        format: Output format ('ISO8601', 'US', 'EU')

    Returns:
        Normalized date string, or None if parsing failed

    Example:
        >>> normalize_date("01/15/2025", format='ISO8601')
        '2025-01-15'
    """
    # Try to validate and parse the date
    is_valid, message = validate_date(date_str)

    if not is_valid:
        return None

    # Extract the ISO date from validation message
    if "Valid date:" in message:
        iso_date = message.split("Valid date: ")[1]
        date_obj = datetime.strptime(iso_date, '%Y-%m-%d')
    else:
        return None

    # Format according to requested format
    if format == 'ISO8601':
        return date_obj.strftime('%Y-%m-%d')
    elif format == 'US':
        return date_obj.strftime('%m/%d/%Y')
    elif format == 'EU':
        return date_obj.strftime('%d-%m-%Y')
    else:
        return date_obj.strftime('%Y-%m-%d')


def normalize_ssn(ssn: str) -> str:
    """
    Normalize a Social Security Number to XXX-XX-XXXX format.

    Args:
        ssn: The SSN to normalize

    Returns:
        Normalized SSN

    Example:
        >>> normalize_ssn("123456789")
        '123-45-6789'
    """
    digits = re.sub(r'\D', '', ssn)
    return f'{digits[0:3]}-{digits[3:5]}-{digits[5:9]}'


def normalize_zip_code(zip_code: str, format: str = 'short') -> str:
    """
    Normalize a zip code.

    Args:
        zip_code: The zip code to normalize
        format: 'short' (5-digit) or 'long' (9-digit with hyphen)

    Returns:
        Normalized zip code

    Example:
        >>> normalize_zip_code("12345-6789", format='short')
        '12345'
    """
    digits = re.sub(r'\D', '', zip_code)

    if format == 'short':
        return digits[:5]
    elif format == 'long' and len(digits) >= 9:
        return f'{digits[0:5]}-{digits[5:9]}'
    else:
        return digits


# ==============================================================================
# PROMPT ENGINEERING FUNCTIONS
# ==============================================================================

def create_pattern_extraction_prompt(
    base_prompt: str,
    pattern_name: str,
    extract_all: bool = False
) -> str:
    """
    Create an optimized prompt for pattern extraction.

    Args:
        base_prompt: The user's original prompt
        pattern_name: Name of the pattern to extract
        extract_all: Whether to extract all occurrences

    Returns:
        Enhanced prompt with pattern extraction instructions

    Example:
        >>> create_pattern_extraction_prompt("Find the contact info", "email", False)
        "Find the contact info\\n\\nPlease provide a valid email address..."
    """
    pattern_instructions = {
        'email': 'Please provide a valid email address in the format: username@domain.com',
        'phone': 'Please provide a phone number in one of these formats: (XXX) XXX-XXXX, XXX-XXX-XXXX, or XXX.XXX.XXXX',
        'url': 'Please provide a complete URL with protocol (e.g., https://example.com)',
        'date': 'Please provide a date in one of these formats: YYYY-MM-DD, MM/DD/YYYY, or Month DD, YYYY',
        'time': 'Please provide a time in 12-hour (HH:MM AM/PM) or 24-hour (HH:MM) format',
        'ssn': 'Please provide a Social Security Number in the format: XXX-XX-XXXX',
        'credit_card': 'Please provide a credit card number (digits only or with spaces/hyphens)',
        'zip_code': 'Please provide a US zip code in 5-digit (XXXXX) or 9-digit (XXXXX-XXXX) format',
        'ipv4': 'Please provide an IPv4 address in the format: XXX.XXX.XXX.XXX',
        'ipv6': 'Please provide an IPv6 address',
        'currency': 'Please provide a currency amount in the format: $X,XXX.XX or XXXX.XX',
        'hex_color': 'Please provide a hex color code in the format: #RRGGBB or #RGB',
        'username': 'Please provide a username (alphanumeric with underscores/hyphens)',
        'hashtag': 'Please provide a hashtag starting with #',
        'filepath': 'Please provide a valid file path',
    }

    instruction = pattern_instructions.get(
        pattern_name.lower(),
        f'Please provide a valid {pattern_name}'
    )

    quantity = "all instances of" if extract_all else "a single"

    enhanced_prompt = f"""{base_prompt}

{instruction}. Provide {quantity} the {pattern_name} in your response.

IMPORTANT: Include the {pattern_name} clearly in your response text."""

    return enhanced_prompt


def get_validation_function(pattern_type: str):
    """
    Get the appropriate validation function for a pattern type.

    Args:
        pattern_type: The type of pattern (e.g., 'email', 'phone')

    Returns:
        The validation function, or None if not available
    """
    validators = {
        'email': validate_email,
        'phone': validate_phone,
        'url': validate_url,
        'date': validate_date,
        'ssn': validate_ssn,
        'credit_card': validate_credit_card,
        'zip_code': validate_zip_code,
        'ipv4': lambda ip: validate_ip_address(ip, version=4),
        'ipv6': lambda ip: validate_ip_address(ip, version=6),
    }

    return validators.get(pattern_type.lower())


def get_normalization_function(pattern_type: str):
    """
    Get the appropriate normalization function for a pattern type.

    Args:
        pattern_type: The type of pattern (e.g., 'email', 'phone')

    Returns:
        The normalization function, or None if not available
    """
    normalizers = {
        'email': normalize_email,
        'phone': normalize_phone,
        'url': normalize_url,
        'date': normalize_date,
        'ssn': normalize_ssn,
        'zip_code': normalize_zip_code,
    }

    return normalizers.get(pattern_type.lower())
