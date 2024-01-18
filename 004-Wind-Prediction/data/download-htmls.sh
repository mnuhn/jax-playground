# Script to download wind data.
WHAT=$1
YEAR=$2
MONTH=$3

if [[ "${WHAT}" != "mythenquai" && "${WHAT}" != "tiefenbrunnen" ]]; then
  echo "1st arg must either 'mythenquai' or 'tiefenbrunnen'."
  exit 1
fi

if ! [[ "${YEAR}" =~ ^20[0-9]{2}$ ]]; then
    echo "2nd arg must be a year starting with 20"
    exit 1
fi

if ! [[ "${MONTH}" =~ ^0[1-9]$|^1[0-2]$ ]]; then
    echo "3rd arg must be a month between 1 and 12"
    exit 1
fi

START="01.${MONTH}.${YEAR}"
END="31.${MONTH}.${YEAR}"

mkdir -p data/${WHAT}/

# curl \
#  -X POST \
#  URL_HERE_FIGURE_IT_OUT_YOURSELF \
#  > data/${WHAT}/${YEAR}-${MONTH}.html
