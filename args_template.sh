#!/bin/bash

USAGE="Usage: $(basename "$0") [-h|--help] [-i|--infile] filepath [-o|--outfile] filepath [-m|--model] filepath

where:
	-h, --help	Help
	-i, --infile	Input file
	-o, --outfile	Output file
	-m, --model	Model file

Examples: sh subhajit.sh -i a.txt -o b.txt -m c.pkl
"

opt_short="hiom:"
opt_long="help,infile,outfile,model,foo:"

OPTS=$(getopt -o "$opt_short" -l "$opt_long" -- "$@")

if [ $? -ne 0 ] ; then
        echo "Wrong input parameter!"; 1>&2
        exit 1;
fi

while [ ! $# -eq 0 ]
do
    case "$1" in
        -h|--help) 
            echo "$USAGE"
            exit 0 ;;
        -i|--infile) 
            INFILE=$2
            shift 2 ;;
        -o|--outfile)
            OUTFILE=$2
            shift 2 ;;
        -m|--model)
            MODEL=$2
            shift 2 ;;
        --foo)
            FOO=$2
            shift 2 ;;
        --) # End of input reading
            shift; break ;;
        * )
            shift;
    esac
done

if [ -z "$MODEL" ] ; then
    echo "Model argument is required! [-m | --model]" 1>&2
    exit 1
fi

if ! [ -z "$INFILE" ] && ! [ -f "$INFILE" ] ; then
    echo "$INFILE does not exist!" 1>&2
    exit 1
fi

echo "infile: $INFILE outfile: $OUTFILE model: $MODEL"
echo 'cannot evaluate "$FOO" if double quote used within single quote.'
echo 'can evaluate '"$FOO"' when single quote ended and double quote started.'

# Print with color: If it does not work, try adding "-e" after echo, e.g. echo -e "${GREEN}`date`${NC}"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo "I ${RED}love${NC} Stack Overflow"
echo "${GREEN}`date`${NC}"
