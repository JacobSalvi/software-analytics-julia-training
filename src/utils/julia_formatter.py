import re

def julia_formatter(code: str) -> str:
    lines = code.splitlines()
    formatted_lines = []
    indent_level = 0
    indent_size = 4

    increase_indent_keywords = ["function", "if", "for", "while", "begin", "try", "let", "struct", "mutable struct"]
    decrease_indent_keywords = ["end", "else", "elseif", "catch", "finally"]

    block_stack = []

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            formatted_lines.append("")
            continue

        inline_match = re.match(r'(.*?\b(if|for|while|function|try|let|struct)\b.*?\bend\b.*)', stripped_line)
        if inline_match:
            formatted_lines.append(" " * (indent_level * indent_size) + stripped_line)
            continue

        if any(stripped_line.startswith(keyword) for keyword in decrease_indent_keywords):
            if block_stack:
                block_stack.pop()
            indent_level = max(indent_level - 1, 0)


        formatted_lines.append(" " * (indent_level * indent_size) + stripped_line)


        if any(stripped_line.startswith(keyword) for keyword in increase_indent_keywords):
            block_stack.append(stripped_line.split()[0])  # Push the keyword onto the stack
            indent_level += 1

    while block_stack:
        formatted_lines.append(" " * ((indent_level - 1) * indent_size) + "end")
        block_stack.pop()
        indent_level = max(indent_level - 1, 0)

    formatted_code = "\n".join(formatted_lines)

    formatted_code = re.sub(r'\s*([=+\-*/<>|&!])\s*', r' \1 ', formatted_code)  # Operators
    formatted_code = re.sub(r'\s*,\s*', r', ', formatted_code)  # Commas
    formatted_code = re.sub(r'\s*\(\s*', r'(', formatted_code)  # Open parenthesis
    formatted_code = re.sub(r'\s*\)\s*', r')', formatted_code)  # Close parenthesis

    return formatted_code
