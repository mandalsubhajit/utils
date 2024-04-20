#!/bin/bash

# How to define and parse input arguments for shell script
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

# If MODEL variable is not set, raise an exception
if [ -z "$MODEL" ] ; then
    echo "Model argument is required! [-m | --model]" 1>&2
    exit 1
fi

# If INFILE variable is set, but the file does not exist raise an exception
if ! [ -z "$INFILE" ] && ! [ -f "$INFILE" ] ; then
    echo "$INFILE does not exist!" 1>&2
    exit 1
fi

echo "infile: $INFILE outfile: $OUTFILE model: $MODEL"

# How to handle quotes within quotes
echo 'cannot evaluate "$FOO" if double quotes used within single quotes.'
echo 'can evaluate '"$FOO"' when single quotes ended and double quotes started.'

# Print with color
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e "I ${RED}love${NC} Stack Overflow"
echo -e "${GREEN}`date`${NC}"

: "
Color Reference:

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

# Bold
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White

# Underline
UBlack='\033[4;30m'       # Black
URed='\033[4;31m'         # Red
UGreen='\033[4;32m'       # Green
UYellow='\033[4;33m'      # Yellow
UBlue='\033[4;34m'        # Blue
UPurple='\033[4;35m'      # Purple
UCyan='\033[4;36m'        # Cyan
UWhite='\033[4;37m'       # White

# Background
On_Black='\033[40m'       # Black
On_Red='\033[41m'         # Red
On_Green='\033[42m'       # Green
On_Yellow='\033[43m'      # Yellow
On_Blue='\033[44m'        # Blue
On_Purple='\033[45m'      # Purple
On_Cyan='\033[46m'        # Cyan
On_White='\033[47m'       # White

# High Intensity
IBlack='\033[0;90m'       # Black
IRed='\033[0;91m'         # Red
IGreen='\033[0;92m'       # Green
IYellow='\033[0;93m'      # Yellow
IBlue='\033[0;94m'        # Blue
IPurple='\033[0;95m'      # Purple
ICyan='\033[0;96m'        # Cyan
IWhite='\033[0;97m'       # White

# Bold High Intensity
BIBlack='\033[1;90m'      # Black
BIRed='\033[1;91m'        # Red
BIGreen='\033[1;92m'      # Green
BIYellow='\033[1;93m'     # Yellow
BIBlue='\033[1;94m'       # Blue
BIPurple='\033[1;95m'     # Purple
BICyan='\033[1;96m'       # Cyan
BIWhite='\033[1;97m'      # White

# High Intensity backgrounds
On_IBlack='\033[0;100m'   # Black
On_IRed='\033[0;101m'     # Red
On_IGreen='\033[0;102m'   # Green
On_IYellow='\033[0;103m'  # Yellow
On_IBlue='\033[0;104m'    # Blue
On_IPurple='\033[0;105m'  # Purple
On_ICyan='\033[0;106m'    # Cyan
On_IWhite='\033[0;107m'   # White
"
