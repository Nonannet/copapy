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


def extract_sections(md_text: str) -> dict[str, tuple[str, str]]:
    """
    Extracts sections based on headings (#...).
    Returns {heading_text: section_content}
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

    sections: dict[str, tuple[str, str]] = {}
    matched = list(pattern.findall(md_text))

    for prefix, title, content in matched:
        assert isinstance(content, str)
        content = strip_leading_html_tag(content)
        sections[title] = (prefix + ' ' + title, content.strip().replace('](docs/source/media/', '](media/'))

    return sections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sections from README.md and generate documentation files')
    parser.add_argument('--readme', type=str, default='README.md', help='README.md path')
    parser.add_argument('--build-dir', type=str, default='docs/source', help='Build directory for output files (default: docs/source)')
    args = parser.parse_args()

    readme_path = args.readme
    build_dir = args.build_dir

    with open(readme_path, 'rt') as f:
        readme = extract_sections(f.read())

    with open(os.path.join(build_dir, 'start.md'), 'wt') as f:
        f.write('# Introduction\n' + '\n'.join('\n'.join(readme[s]) if s != 'Copapy' else readme[s][1] for s in [
            'Copapy', 'Current state', 'Install', 'License']))

    with open(os.path.join(build_dir, 'examples.md'), 'wt') as f:
        f.write('\n'.join('\n'.join(readme[s]) for s in ['Examples',
            'Basic example', 'Inverse kinematics']))

    with open(os.path.join(build_dir, 'compiler.md'), 'wt') as f:
        f.write('\n'.join(readme[s][1] for s in ['How it works']))
