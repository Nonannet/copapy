import re
import argparse
import os

def extract_sections(md_text: str) -> dict[str, str]:
    """
    Extracts sections based on headings (#...).
    Returns {heading_text: section_content}
    Works for simple Markdown, not fully strict.
    """

    # regex captures: heading marks (###...), heading text, and the following content
    pattern = re.compile(
        r'^(#{1,6})\s+(.*?)\s*$'          # heading level + heading text
        r'(.*?(?:```.*?```.*?)*?)'        # section content (lazy)
        r'(?=^#{1,6}\s+|\Z)',             # stop at next heading or end of file
        re.MULTILINE | re.DOTALL
    )

    sections: dict[str, str] = {}
    for _, title, content in pattern.findall(md_text):
        assert isinstance(content, str)
        sections[title] = content.strip().replace('](docs/source/media/', '](media/')

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
        f.write('\n'.join(f"{s}\n" + readme[s.strip(' #')] for s in [
            '# Copapy', '## Current state', '## Install', '## Examples',
            '### Basic example', '### Inverse kinematics', '## License']))

    with open(os.path.join(build_dir, 'compiler.md'), 'wt') as f:
        f.write('\n'.join(readme[s] for s in ['How it works']))
