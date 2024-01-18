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

curl \
  -X POST \
  https://www.tecson-data.ch/zurich/${WHAT}/uebersicht/messwerte.php \
  -d "messw_beg=${START}&messw_end=${END}&felder%5B%5D=Temp2m&felder%5B%5D=Feuchte&felder%5B%5D=WGmax&felder%5B%5D=WGavr&felder%5B%5D=Umr_Beaufort&felder%5B%5D=WRvek&felder%5B%5D=Windchill&felder%5B%5D=TempWasser&felder%5B%5D=LuftdruckQFE&felder%5B%5D=Taupunkt&auswahl=2&combilog=tiefenbrunnen&suchen=Werte+anzeigen" \
  > data/${WHAT}/${YEAR}-${MONTH}.html
