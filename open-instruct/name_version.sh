# Generate a unique name for a file in a given directory

# Check for at least 2 arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 base_name directory"
    exit 1
fi

echo "1: $1"
echo "2: $2"

# determine the output model name
BASE=$1
COUNTER=1
CHOSEN_NAME="${BASE}$(printf "%02d" $COUNTER)"

while [ -d "$2/$CHOSEN_NAME" ]; do
    COUNTER=$((COUNTER + 1))
    CHOSEN_NAME="${BASE}$(printf "%02d" $COUNTER)"
done

echo "$CHOSEN_NAME"

