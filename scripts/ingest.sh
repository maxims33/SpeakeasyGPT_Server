#!/bin/bash

# Common exports across scripts
source ./scripts/common_env.sh

export ENABLE_DEBUG=True

ingest_path="./filebox/";
do_clean=false;

echo "Usage: scripts/ingest.sh -c TRUE_IF_DO_CLEAN_FIRST -p FILEPATH_TO_INGEST_FILES_FROM"

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
  ./scripts/cleandbs.sh
fi

pipenv run python ingest.py --ingest-path=${ingest_path}
