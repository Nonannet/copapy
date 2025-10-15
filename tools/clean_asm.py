import argparse
import re

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Input file path")
    args = parser.parse_args()

    regex = r"(\:\n)(.*?)(^[^\n]+\n[^\n]+result_.*?\n\n)"
    subst = "\\g<1><b>\\g<2></b><i>\\g<3></i>"

    with open(args.path, 'rt') as f:
        text = f.read()

    outp_flag = "Disassembly of section .text" not in text

    text = text.replace('\t', ' ' * 4)
    text = text.replace('>', '&gt;')
    text = text.replace('<', '&lt;')

    text = re.sub(regex, subst, text, 0, re.MULTILINE | re.DOTALL)

    text = re.sub(r"(?<= ) ", "&nbsp;", text, 0, re.MULTILINE | re.DOTALL)

    print('<code>')
    for line in text.splitlines():
        if outp_flag:
            print(line + '<br>')

        if "Disassembly of section .text:" in line:
            outp_flag = True
            
    print('</code>')

if __name__ == "__main__":
    main()
