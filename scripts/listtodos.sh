#!/bin/sh
grep -r "#TODO" ./ --exclude="listtodos.sh" --exclude-dir=".pythonlibs" --exclude-dir="db_*" --exclude-dir="*.db" --exclude-dir="__pycache__" --exclude-dir=".git"
