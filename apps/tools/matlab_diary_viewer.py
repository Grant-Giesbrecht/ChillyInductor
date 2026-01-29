import re
from typing import Optional

def display_matlab_diary(
    diary_text: str,
    *,
    strong_color: Optional[str] = None,
    link_color: Optional[str] = None,
    show_link_target: bool = False,
) -> None:
    """
    Print MATLAB diary output to a Python console, converting common MATLAB diary
    HTML-ish markup into ANSI colored text (via colorama if available).

    Handles:
      - <strong>...</strong>  -> colored/bright text
      - <a href="...">...</a> -> colored link text (optionally appends target)
      - stray backspace chars from diary formatting (\\x08) removed
      - basic HTML entities: &lt; &gt; &amp; &quot; &#39;

    Parameters
    ----------
    diary_text:
        Raw diary text (string) read from MATLAB diary file.
    strong_color, link_color:
        Override colors (must be colorama Fore.* strings). If None, defaults used.
    show_link_target:
        If True, prints link text as: "TEXT (matlab:...)" or "TEXT (D:\\path...)".
    """
    # Optional colorama support (falls back to plain text if missing)
    try:
        from colorama import Fore, Style, init as _colorama_init
        _colorama_init(autoreset=True)
        RESET = Style.RESET_ALL
        STRONG = (strong_color if strong_color is not None else (Fore.GREEN + Style.BRIGHT))
        LINK = (link_color if link_color is not None else (Fore.CYAN + Style.BRIGHT))
    except Exception:
        Fore = Style = None  # noqa: N816
        RESET = ""
        STRONG = ""
        LINK = ""

    s = diary_text

    # MATLAB diary often includes literal backspace characters (shown as ] in your paste)
    # Remove them. (This won’t “replay” terminal backspacing; it just cleans the text.)
    s = s.replace("\x08", "")

    # Decode a few common HTML entities (MATLAB diary sometimes includes these)
    s = (s.replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&amp;", "&")
           .replace("&quot;", '"')
           .replace("&#39;", "'"))

    # Replace <strong>...</strong>
    # Use non-greedy match; DOTALL so it works across wrapped lines.
    def _strong_sub(m: re.Match) -> str:
        inner = m.group(1)
        return f"{STRONG}{inner}{RESET}"

    s = re.sub(r"<strong>(.*?)</strong>", _strong_sub, s, flags=re.DOTALL | re.IGNORECASE)

    # Replace <a ...>...</a>
    # Capture href=... and the displayed text. Ignore other attributes.
    # Example: <a href="matlab:..." style="font-weight:bold">RBDataTaker</a>
    def _a_sub(m: re.Match) -> str:
        href = m.group("href")
        text = m.group("text")
        if show_link_target:
            return f"{LINK}{text}{RESET} ({href})"
        return f"{LINK}{text}{RESET}"

    s = re.sub(
        r'<a\s+[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<text>.*?)</a>',
        _a_sub,
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Drop any remaining tags (MATLAB diary occasionally has other simple tags)
    # If you prefer to keep unknown tags, delete this line.
    s = re.sub(r"</?[^>]+>", "", s)

    print(s)


# --- Convenience helper for files ---
def display_matlab_diary_file(path: str, *, encoding: str = "utf-8", errors: str = "replace", **kwargs) -> None:
    with open(path, "r", encoding=encoding, errors=errors) as f:
        display_matlab_diary(f.read(), **kwargs)

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Display a MATLAB diary file with console colorization."
    )
    parser.add_argument(
        "filename",
        help="Path to MATLAB diary text file"
    )
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Show link targets (e.g. matlab:... or file paths) after link text"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
    )

    args = parser.parse_args()

    try:
        display_matlab_diary_file(
            args.filename,
            encoding=args.encoding,
            show_link_target=args.show_links,
        )
    except FileNotFoundError:
        print(f"Error: file not found: {args.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while processing diary file: {e}", file=sys.stderr)
        sys.exit(2)
