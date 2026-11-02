#!/usr/bin/env python
"""
Скачивает фото кошек для предобучения. Запуск можно прервать и повторить: готовые файлы пропускаются.

  oi   : Open Images V7, кадры с боксами класса Cat, размеченными людьми, без рисунков и групп.
         В <root>/oi: images/ и boxes.json (координаты в долях кадра).
  inat : iNaturalist, Felis catus с пометкой captive и лицензией CC без ND.
         В <root>/inat: images/ и photos.jsonl (лицензия и автор каждого фото, их нужно сохранить).

Запуск на логин-ноде, в tmux:
  .venv/bin/python download_cats.py oi --workers 32
  .venv/bin/python download_cats.py inat --workers 64
"""
import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "DiffusionLearn/0.1 (personal study project)"}
CAT = "/m/01yrx"
OI_CSV = {"train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
          "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
          "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"}
INAT_API = "https://api.inaturalist.org/v2/observations"   # v2 отдает только нужные поля, ответ в 400 раз легче v1
INAT_FIELDS = "id,photos.id,photos.url,photos.license_code,photos.attribution,photos.original_dimensions"
INAT_LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"   # без ND: кропы являются производной работой


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["oi", "inat"])
    p.add_argument("--root", type=str, default=str(Path.home() / "DiffusionLearn_data"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="для проверки: oi = кадров, inat = страниц API")
    # iNaturalist предупреждает о блокировке при медиа выше 5 ГБ в час или 24 ГБ в день, держимся ниже. 0 = без ограничения
    p.add_argument("--max-gb-hour", type=float, default=4.0, help="только inat")
    p.add_argument("--max-gb-day", type=float, default=20.0, help="только inat, сутки по UTC, счетчик переживает перезапуск")
    return p.parse_args()


def get(url, tries=5):
    """Содержимое url, с повторами при сбоях сети и лимитах сервера."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
        except OSError:
            if k == tries - 1:
                raise
        time.sleep(2 ** k)


def fetch(job):
    """Скачивает один файл через .tmp, чтобы оборванная загрузка не выглядела готовой. Возвращает текст ошибки или None."""
    url, path = job
    if path.exists() or (path.suffix != ".jpg" and path.with_suffix(".jpg").exists()):
        return None
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(get(url))
        tmp.rename(path)
    except Exception as e:
        return f"{type(e).__name__} {str(e)[:60]}"


def fetch_all(jobs, workers):
    """Возвращает число скачанных и словарь ошибок."""
    done, errs = 0, {}
    with ThreadPoolExecutor(workers) as ex:
        for err in ex.map(fetch, jobs):
            if err:
                errs[err] = errs.get(err, 0) + 1
            else:
                done += 1
    return done, errs


def oi_boxes(path):
    """Боксы кошек из трех csv, потоком, чтобы не держать файлы на диске. Кадры с рисунками и группами отбрасываем."""
    if path.exists():
        return json.loads(path.read_text())
    res = {}
    for split, url in OI_CSV.items():
        t0 = time.time()
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            lines = io.TextIOWrapper(r, encoding="utf-8")
            head = next(lines).strip().split(",")
            for line in lines:
                if f",{CAT}," not in line:
                    continue
                x = dict(zip(head, line.strip().split(",")))
                res.setdefault(x["ImageID"], dict(split=split, boxes=[]))["boxes"].append(x)
        print(f"разметка {split}: кадров всего {len(res)}, {time.time() - t0:.0f} с", flush=True)
    keep = {}
    for k, r in res.items():
        if any(b["IsDepiction"] != "0" or b["IsGroupOf"] != "0" for b in r["boxes"]):
            continue
        keep[k] = dict(split=r["split"], boxes=[dict(x1=float(b["XMin"]), y1=float(b["YMin"]), x2=float(b["XMax"]), y2=float(b["YMax"]),
                                                     truncated=b["IsTruncated"] == "1", occluded=b["IsOccluded"] == "1")
                                                for b in r["boxes"]])
    path.write_text(json.dumps(keep))
    print(f"оставлено {len(keep)} кадров из {len(res)}", flush=True)
    return keep


def download_oi(a, root):
    d = root / "oi"
    (d / "images").mkdir(parents=True, exist_ok=True)
    boxes = oi_boxes(d / "boxes.json")
    names = sorted(boxes)[:a.limit or None]
    print(f"кадров к загрузке {len(names)}", flush=True)
    t0, total, errs = time.time(), 0, {}
    for i in range(0, len(names), 500):
        jobs = [(f"https://open-images-dataset.s3.amazonaws.com/{boxes[k]['split']}/{k}.jpg", d / "images" / f"{k}.jpg") for k in names[i:i + 500]]
        n, e = fetch_all(jobs, a.workers)
        total += n
        for k, v in e.items():
            errs[k] = errs.get(k, 0) + v
        print(f"готово {total} из {len(names)}, {total / (time.time() - t0):.1f} кадров/с, ошибки {errs}", flush=True)


class Quota:
    """Держит объем скачанного ниже лимитов: пауза после страницы, суточный счетчик хранится на диске."""

    def __init__(self, path, gb_hour, gb_day):
        self.path, self.gb_hour, self.gb_day = path, gb_hour, gb_day
        self.t0, self.session = time.time(), 0
        self.day, self.today = self.utc_day(), 0
        if path.exists():
            s = json.loads(path.read_text())
            if s["day"] == self.day:
                self.today = s["bytes"]

    @staticmethod
    def utc_day():
        return time.strftime("%Y-%m-%d", time.gmtime())

    def add(self, n):
        self.session += n
        if self.utc_day() != self.day:
            self.day, self.today = self.utc_day(), 0
        self.today += n
        self.path.write_text(json.dumps(dict(day=self.day, bytes=self.today)))
        if self.gb_hour:
            wait = self.session / 2 ** 30 / self.gb_hour * 3600 - (time.time() - self.t0)
            if wait > 0:
                time.sleep(wait)
        if self.gb_day and self.today >= self.gb_day * 2 ** 30:
            wait = 86400 - time.time() % 86400 + 60
            print(f"суточный лимит {self.gb_day} ГБ, пауза {wait / 3600:.1f} ч до новых суток UTC", flush=True)
            time.sleep(wait)
            self.t0, self.session = time.time(), 0


def download_inat(a, root):
    d = root / "inat"
    (d / "images").mkdir(parents=True, exist_ok=True)
    state = d / "photos.jsonl"
    last = 0
    if state.exists():   # продолжаем с последнего полностью записанного наблюдения
        for line in state.open():
            last = max(last, json.loads(line)["obs"])
    print(f"продолжаем после наблюдения {last}", flush=True)
    quota = Quota(d / "usage.json", a.max_gb_hour, a.max_gb_day)
    t0, pages, photos, errs = time.time(), 0, 0, {}
    while not a.limit or pages < a.limit:
        q = dict(taxon_id=118552, photos="true", captive="true", photo_license=INAT_LICENSES, order_by="id", order="asc",
                 per_page=200, id_above=last, fields=INAT_FIELDS)
        j = json.loads(get(f"{INAT_API}?" + urllib.parse.urlencode(q)))
        obs = j["results"]
        if not obs:
            break
        recs = [dict(id=p["id"], obs=o["id"], license=p["license_code"], attribution=p.get("attribution"),
                     url=p["url"].replace("square", "large"), dims=p.get("original_dimensions"))
                for o in obs for p in o["photos"] if p.get("license_code") in INAT_LICENSES.split(",")]
        paths = [d / "images" / f"{r['id']}{Path(urllib.parse.urlparse(r['url']).path).suffix.lower()}" for r in recs]
        before = sum(p.stat().st_size for p in paths if p.exists())
        n, e = fetch_all([(r["url"], p) for r, p in zip(recs, paths)], a.workers)
        size = sum(p.stat().st_size for p in paths if p.exists()) - before
        with state.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        last = obs[-1]["id"]
        pages += 1
        photos += n
        for k, v in e.items():
            errs[k] = errs.get(k, 0) + v
        print(f"страница {pages}, наблюдение {last}, осталось наблюдений {j['total_results'] - len(obs)}, фото {photos}, "
              f"сегодня {quota.today / 2 ** 30:.1f} ГБ, {photos / (time.time() - t0):.1f} фото/с, ошибки {errs}", flush=True)
        quota.add(size)
        time.sleep(1.1)   # к API не чаще запроса в секунду


def main():
    a = parse_args()
    root = Path(a.root)
    (download_oi if a.source == "oi" else download_inat)(a, root)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
