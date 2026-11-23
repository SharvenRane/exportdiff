#!/bin/bash
# usage: run.sh NAME MODEL LIMIT TARGETS
set -e
cd "${EXPORTDIFF_WORKDIR:?set EXPORTDIFF_WORKDIR to the folder holding repo/, data/, work/ and out/}"
LIM=""; [ "$3" != "all" ] && LIM="--limit $3"
docker run --rm --name exportdiff-$1 --cpus 8 --memory 10g -e OMP_NUM_THREADS=8 -e PYTHONPATH=/repo/src \
  -e PYTHONDONTWRITEBYTECODE=1 -e YOLO_CONFIG_DIR=/work/ycfg \
  -v $PWD/repo:/repo:ro -v $PWD/data:/data:ro -v $PWD/work:/work -v $PWD/out:/out -w /work exportdiff-dev \
  nice -n 10 python -m exportdiff run --model /work/weights/$2 --data /data/val2017 \
  --ann /data/annotations/instances_val2017.json --calib /data/coco128/images/train2017 \
  --targets $4 $LIM --out /out/$1 --work /work/cache-${2%.pt}
