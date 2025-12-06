import re

def extract_sections(md_text: str) -> dict[str, str]:
    """
    Extracts sections based on headings (#...).
    Returns {heading_text: section_content}
    Works for simple Markdown, not fully strict.
    """

    # regex captures: heading marks (###...), heading text, and the following content
    pattern = re.compile(
        r'^(#{1,6})\s+(.*?)\s*$'          # heading level + heading text
        r'(.*?)'                          # section content (lazy)
        r'(?=^#{1,6}\s+|\Z)',             # stop at next heading or end of file
        re.MULTILINE | re.DOTALL
    )

    sections: dict[str, str] = {}
    for _, title, content in pattern.findall(md_text):
        sections[title] = content.strip()

    return sections

if __name__ == '__main__':
    with open('README.md', 'rt') as f:
        readme = extract_sections(f.read())

    with open('docs/source/start.md', 'wt') as f:
        f.write('\n'.join(readme[s] for s in ['Copapy', 'Current state']))