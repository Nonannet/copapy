import re
import argparse
import os


def strip_leading_html_tag(text: str) -> str:
    """Remove a leading HTML element if the first non-space character is '<'."""
    trimmed = text.lstrip()
    #if not trimmed.startswith('<'):
    #    return text

    match = re.match(
        r'^\s*<\s*(?P<tag>[A-Za-z][A-Za-z0-9:-]*)(?:\s+[^<>]*)?>'
        r'(?P<body>.*?)(?:</\s*(?P=tag)\s*>)?',
        trimmed,
        re.DOTALL,
    )
    if not match:
        return text

    return trimmed[match.end():]


def extract_sections(md_text: str) -> dict[str, tuple[str, str, str]]:
    """
    Extracts sections based on headings (#...).
    Returns {heading_text: (prefix, full_heading, section_content)}
    Works for simple Markdown, not fully strict.

    If strip_first_heading is True, omit the first heading/section from the output.
    """

    # regex captures: heading marks (###...), heading text, and the following content
    pattern = re.compile(
        r'^(#{1,6})\s+(.*?)\s*$'          # heading level + heading text
        r'(.*?(?:```.*?```.*?)*?)'        # section content (lazy)
        r'(?=^#{1,6}\s+|\Z)',             # stop at next heading or end of file
        re.MULTILINE | re.DOTALL
    )

    sections: dict[str, tuple[str, str, str]] = {}
    matched = list(pattern.findall(md_text))

    def adjust_prefix(p: str) -> str:
        level = len(p)
        if level >= 3:
            return '#' * (level - 1)
        return p

    for prefix, title, content in matched:
        assert isinstance(content, str)
        content = strip_leading_html_tag(content)
        adjusted = adjust_prefix(prefix)
        sections[title] = (adjusted, adjusted + ' ' + title, content.strip().replace('](docs/source/media/', '](media/'))

    return sections


def section_text(titles: list[str], sections: dict, content_only_titles: set[str] = None) -> str:
    """Return formatted text for multiple sections.

    Args:
        titles: List of section titles to include
        sections: The sections dictionary from extract_sections
        content_only_titles: Section titles that should only output content (no heading)
    """
    content_only_titles = content_only_titles or set()
    parts = []
    for title in titles:
        prefix, full_heading, content = sections[title]
        if title in content_only_titles:
            parts.append(content)
        else:
            parts.append(f"\n{full_heading}\n{content}")
    return '\n'.join(parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sections from README.md and generate documentation files')
    parser.add_argument('--readme', type=str, default='README.md', help='README.md path')
    parser.add_argument('--build-dir', type=str, default='docs/build', help='Build directory for output files')
    args = parser.parse_args()

    readme_path = args.readme
    build_dir = args.build_dir

    with open(readme_path, 'rt') as f:
        readme = extract_sections(f.read())

    with open(os.path.join(build_dir, 'start.md'), 'wt') as f:
        f.write('# Introduction\n' + section_text(
            ['Copapy', 'Current state', 'Install', 'License'], readme, {'Copapy'}))

    with open(os.path.join(build_dir, 'examples.md'), 'wt') as f:
        f.write(section_text(['Basic example', 'IMU Sensor fusion', 'Inverse kinematics'], readme))

    with open(os.path.join(build_dir, 'compiler.md'), 'wt') as f:
        f.write(section_text(['How it works'], readme))
