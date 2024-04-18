# Usage: sh thisfile.sh [-i|--infile] a.txt [-o|--outfile] b.txt

opt_short="iom:"
opt_long="infile,outfile,model:"

OPTS=$(getopt -o "$opt_short" -l "$opt_long" -- "$@")

if [ $? -ne 0 ] ; then
        echo "Wrong input parameter!"; 1>&2
        exit 1;
fi

while [ ! $# -eq 0 ]
do
    case "$1" in
        -i|--infile) 
            INFILE=$2
            shift 2 ;;
        -o|--outfile)
            OUTFILE=$2
            shift 2 ;;
        -m|--model)
            MODEL=$2
            shift 2 ;;
        --) # End of input reading
            shift; break ;;
        * )
            shift;
    esac
done

if [ -z "$MODEL" ] ; then
    echo "Model is required! [-m | --model]" 1>&2
    exit 1
fi

echo "infile: $INFILE outfile: $OUTFILE model: $MODEL"
