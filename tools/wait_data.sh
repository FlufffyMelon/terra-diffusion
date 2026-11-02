#!/bin/bash
# Ждет конца скачивания iNaturalist и детекции, потом останавливает детекторы.
# Скачивание идет на логин-ноде, с вычислительной его процесс не виден, поэтому смотрим на файлы:
# готово, когда 15 минут нет новых фото и обе модели обработали все, что есть.
# Аргументы: крайний срок (например "2026-11-03 03:00"), затем номера задач детекторов.
# После крайнего срока идем дальше с тем, что есть: обучение важнее, чем последние кадры.

DEADLINE=$(date -d "$1" +%s); shift
R=$HOME/DiffusionLearn_data
prev=-1; stable=0
while true; do
    n=$(find "$R/inat/images" -type f ! -name "*.tmp" | wc -l)
    y=$(wc -l < "$R/detections/yolo26x_inat.jsonl")
    d=$(wc -l < "$R/detections/rtdetr-x_inat.jsonl")
    if [ "$n" -eq "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
    prev=$n
    echo "$(date +%H:%M) фото $n, yolo $y, rtdetr $d, без новых $stable мин"
    if [ "$stable" -ge 15 ] && [ "$y" -ge "$n" ] && [ "$d" -ge "$n" ]; then echo "данные готовы"; break; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "крайний срок, идем дальше с тем, что есть"; break; fi
    sleep 60
done
scancel "$@"
