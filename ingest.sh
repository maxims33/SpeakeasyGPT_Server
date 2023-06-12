#!/bin/sh

# Common exports across scripts
source ./common_env.sh

export ENABLE_DEBUG=True
#export FACTORY_TYPE=GOOGLE

ingest_path="./filebox/";
do_clean=false;

echo "Usage: ingest.sh -c TRUE_IF_DO_CLEAN_FIRST -f FILEPATH_TO_INGEST_FILES_FROM"

while getopts c:p: flag
do
  case "${flag}" in
    c) do_clean=${OPTARG};;
    p) ingest_path="${OPTARG}";;
  esac
done
echo "Path: $ingest_path";
echo "Clean: $do_clean";

if ${do_clean} == true; then
  echo "Cleaning DBs"
  ./cleandbs.sh
fi

python ingest.py --ingest-path=${ingest_path}

