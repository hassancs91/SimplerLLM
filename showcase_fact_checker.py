"""
AI Fact-Checker Showcase - Multi-Provider Validation Demo

A visually appealing terminal demo showcasing the LLMValidator feature
for fact-checking claims using multiple AI providers.
"""

import sys
import time
from typing import List, Dict

from SimplerLLM.language import LLM, LLMProvider
from SimplerLLM.language.llm_validator import LLMValidator


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Color Codes for Terminal Styling
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    # Basic colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Reset
    RESET = "\033[0m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal UI Components
# ═══════════════════════════════════════════════════════════════════════════════

def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_header():
    """Print the application header."""
    header = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║   {Colors.WHITE}█▀▀ ▄▀█ █▀▀ ▀█▀ ▄▄ █▀▀ █░█ █▀▀ █▀▀ █▄▀ █▀▀ █▀█{Colors.CYAN}                    ║
    ║   {Colors.WHITE}█▀░ █▀█ █▄▄ ░█░ ░░ █▄▄ █▀█ ██▄ █▄▄ █░█ ██▄ █▀▄{Colors.CYAN}                    ║
    ║                                                                       ║
    ║   {Colors.YELLOW}Multi-Provider AI Validation System{Colors.CYAN}                             ║
    ║   {Colors.DIM}Powered by SimplerLLM - LLMValidator{Colors.CYAN}                             ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(header)


def print_section_header(title: str, icon: str = "►"):
    """Print a section header."""
    width = 70
    print(f"\n{Colors.BLUE}{Colors.BOLD}  {icon} {title}")
    print(f"  {'─' * width}{Colors.RESET}")


def print_claim_box(claim: str, claim_num: int, total: int):
    """Print a styled box around the claim being checked."""
    print(f"""
{Colors.WHITE}{Colors.BOLD}  ┌─────────────────────────────────────────────────────────────────────┐
  │  {Colors.CYAN}CLAIM {claim_num}/{total}{Colors.WHITE}                                                          │
  ├─────────────────────────────────────────────────────────────────────┤{Colors.RESET}""")

    # Word wrap the claim
    words = claim.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 65:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        padding = 65 - len(line)
        print(f"{Colors.WHITE}  │  {Colors.YELLOW}\"{line}\"{' ' * padding}{Colors.WHITE}│{Colors.RESET}")

    print(f"{Colors.WHITE}  └─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")


def print_loading_animation(message: str, duration: float = 0.5):
    """Print a loading animation."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r{Colors.CYAN}  {frames[i % len(frames)]} {message}...{Colors.RESET}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r{Colors.GREEN}  ✓ {message} complete!{Colors.RESET}    ")


def print_validator_results(result):
    """Print individual validator results in a nice format."""
    print(f"\n{Colors.MAGENTA}  ┌─ Validator Responses ─────────────────────────────────────────────┐{Colors.RESET}")

    for v in result.validators:
        # Determine status icon and color
        if v.error:
            icon = "✗"
            color = Colors.RED
            status = "ERROR"
        elif v.score >= 0.7:
            icon = "✓"
            color = Colors.GREEN
            status = "VALID"
        elif v.score >= 0.4:
            icon = "◐"
            color = Colors.YELLOW
            status = "UNCERTAIN"
        else:
            icon = "✗"
            color = Colors.RED
            status = "INVALID"

        # Score bar visualization
        bar_width = 20
        filled = int(v.score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"{Colors.MAGENTA}  │{Colors.RESET}")
        print(f"{Colors.MAGENTA}  │  {Colors.WHITE}{Colors.BOLD}{v.provider_name}{Colors.RESET} {Colors.DIM}({v.model_name}){Colors.RESET}")
        print(f"{Colors.MAGENTA}  │  {Colors.RESET}  {color}{icon} {status}{Colors.RESET}  Score: {color}{bar}{Colors.RESET} {Colors.BOLD}{v.score:.0%}{Colors.RESET}  Confidence: {v.confidence:.0%}")

        # Truncate explanation for display
        explanation = v.explanation[:80] + "..." if len(v.explanation) > 80 else v.explanation
        print(f"{Colors.MAGENTA}  │  {Colors.RESET}  {Colors.DIM}{explanation}{Colors.RESET}")

    print(f"{Colors.MAGENTA}  │{Colors.RESET}")
    print(f"{Colors.MAGENTA}  └─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")


def print_verdict(result):
    """Print the final verdict with visual styling."""
    # Determine verdict
    if result.overall_score >= 0.7:
        verdict = "VERIFIED"
        icon = "✓"
        color = Colors.GREEN
        bg = Colors.BG_GREEN
    elif result.overall_score >= 0.4:
        verdict = "UNCERTAIN"
        icon = "◐"
        color = Colors.YELLOW
        bg = Colors.BG_YELLOW
    else:
        verdict = "FALSE"
        icon = "✗"
        color = Colors.RED
        bg = Colors.BG_RED

    # Consensus indicator
    consensus_icon = "✓" if result.consensus else "✗"
    consensus_color = Colors.GREEN if result.consensus else Colors.YELLOW

    print(f"""
{Colors.WHITE}{Colors.BOLD}  ╔═══════════════════════════════════════════════════════════════════════╗
  ║                           {Colors.RESET}{bg}{Colors.WHITE}{Colors.BOLD}  {icon} {verdict}  {Colors.RESET}{Colors.WHITE}{Colors.BOLD}                              ║
  ╠═══════════════════════════════════════════════════════════════════════╣{Colors.RESET}
  {Colors.WHITE}║{Colors.RESET}  Overall Score:    {color}{Colors.BOLD}{result.overall_score:.0%}{Colors.RESET}                                          {Colors.WHITE}║{Colors.RESET}
  {Colors.WHITE}║{Colors.RESET}  Confidence:       {result.overall_confidence:.0%}                                          {Colors.WHITE}║{Colors.RESET}
  {Colors.WHITE}║{Colors.RESET}  Consensus:        {consensus_color}{consensus_icon} {result.consensus_details[:45]}{Colors.RESET}  {Colors.WHITE}║{Colors.RESET}
  {Colors.WHITE}║{Colors.RESET}  Time:             {result.total_execution_time:.2f}s                                         {Colors.WHITE}║{Colors.RESET}
{Colors.WHITE}{Colors.BOLD}  ╚═══════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")


def print_summary(results: List[Dict]):
    """Print a summary of all checked claims."""
    print_section_header("SUMMARY", "📊")

    verified = sum(1 for r in results if r["result"].overall_score >= 0.7)
    uncertain = sum(1 for r in results if 0.4 <= r["result"].overall_score < 0.7)
    false = sum(1 for r in results if r["result"].overall_score < 0.4)

    print(f"""
{Colors.WHITE}{Colors.BOLD}  ┌─────────────────────────────────────────────────────────────────────┐
  │                        FACT-CHECK RESULTS                           │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │    {Colors.GREEN}✓ VERIFIED:  {verified}{Colors.WHITE}                                                    │
  │    {Colors.YELLOW}◐ UNCERTAIN: {uncertain}{Colors.WHITE}                                                    │
  │    {Colors.RED}✗ FALSE:     {false}{Colors.WHITE}                                                    │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘{Colors.RESET}
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Claims - Mix of true, false, and uncertain claims
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CLAIMS = [
    {
        "claim": "The Great Wall of China is visible from space with the naked eye.",
        "context": "Common misconception about the Great Wall",
    },
    {
        "claim": "Water boils at 100 degrees Celsius at sea level under normal atmospheric pressure.",
        "context": "Basic physics fact",
    },
    {
        "claim": "Humans only use 10% of their brain capacity.",
        "context": "Popular myth about human brain usage",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

def create_validators():
    """Create the AI validators."""
    return [
        LLM.create(LLMProvider.OPENAI, model_name="gpt-4o-mini"),
        LLM.create(LLMProvider.ANTHROPIC, model_name="claude-3-5-haiku-20241022"),
        LLM.create(LLMProvider.GEMINI, model_name="gemini-2.0-flash"),
    ]


def run_fact_checker():
    """Run the fact-checker demo."""
    clear_screen()
    print_header()

    # Initialize
    print_section_header("INITIALIZING", "⚡")
    print_loading_animation("Loading AI validators", 1.0)

    try:
        validators = create_validators()
        validator = LLMValidator(
            validators=validators,
            parallel=True,
            default_threshold=0.7,
            verbose=False,
        )
        print(f"{Colors.GREEN}  ✓ Loaded {len(validators)} AI validators{Colors.RESET}")
        print(f"{Colors.DIM}    • OpenAI GPT-4o-mini{Colors.RESET}")
        print(f"{Colors.DIM}    • Anthropic Claude 3.5 Haiku{Colors.RESET}")
        print(f"{Colors.DIM}    • Google Gemini 2.0 Flash{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}  ✗ Failed to initialize validators: {e}{Colors.RESET}")
        return

    # Validation prompt
    validation_prompt = """
    You are a fact-checker. Evaluate whether this claim is factually accurate.

    Consider:
    - Scientific consensus and established facts
    - Common misconceptions vs verified information
    - Nuance and context that might affect accuracy

    Score 1.0 for completely accurate claims.
    Score 0.5 for partially accurate or context-dependent claims.
    Score 0.0 for false or misleading claims.
    """

    # Process each claim
    print_section_header("FACT-CHECKING CLAIMS", "🔍")

    results = []
    total_claims = len(TEST_CLAIMS)

    for i, item in enumerate(TEST_CLAIMS, 1):
        print_claim_box(item["claim"], i, total_claims)
        print_loading_animation("Querying AI validators", 0.3)

        try:
            result = validator.validate(
                content=item["claim"],
                validation_prompt=validation_prompt,
                context=item.get("context", ""),
                aggregation="average",
            )

            results.append({"claim": item["claim"], "result": result})

            print_validator_results(result)
            print_verdict(result)

        except Exception as e:
            print(f"{Colors.RED}  ✗ Error: {e}{Colors.RESET}")

        # Pause between claims for video effect
        if i < total_claims:
            time.sleep(0.5)

    # Print summary
    if results:
        print_summary(results)

    # Footer
    print(f"\n{Colors.DIM}  ─────────────────────────────────────────────────────────────────────{Colors.RESET}")
    print(f"{Colors.CYAN}  Powered by SimplerLLM - LLMValidator{Colors.RESET}")
    print(f"{Colors.DIM}  https://github.com/hassancs91/SimplerLLM{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        run_fact_checker()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}  Interrupted by user{Colors.RESET}\n")
        sys.exit(0)
