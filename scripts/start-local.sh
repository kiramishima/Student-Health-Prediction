. ./scripts/build-docker-image.sh

docker run -it \
    --rm \
    --name health-student-predictor \
    -p 9696:9696 \
    student-api:2025-01-13-01-02