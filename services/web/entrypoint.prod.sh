#!/bin/sh

echo "Checking for classifier model files..."
#MODEL_DIR=./project/models
sh ./project/model_download.sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
      sleep 0.5
    done

    echo "PostgreSQL started"
fi

exec "$@"
