# Generate a unique name for a file in a given directory

# Check for at least 2 arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 base_name directory"
    exit 1
fi

BASE=$1
COUNTER=1
CHOSEN_NAME="${BASE}$(printf "%02d" $COUNTER)"

while [ -d "$2/$CHOSEN_NAME" ]; do
    COUNTER=$((COUNTER + 1))
    CHOSEN_NAME="${BASE}$(printf "%02d" $COUNTER)"
done

echo "$CHOSEN_NAME"

