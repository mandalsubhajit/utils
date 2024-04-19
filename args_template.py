# Usage: python args_template.py -i a.txt

import argparse

parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument("-i", "--input", dest='inputvalsss', nargs='+', type=str, required=True,  help="Specify the input file.")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.inputvalsss)
