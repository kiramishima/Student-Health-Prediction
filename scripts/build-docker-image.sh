#!/usr/bin/env bash

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="student-api:${LOCAL_TAG}"
echo "Building a new mage image with tag ${LOCAL_IMAGE_NAME}"
ls -l
docker build -t ${LOCAL_IMAGE_NAME} -f ./docker-images/health.Dockerfile .