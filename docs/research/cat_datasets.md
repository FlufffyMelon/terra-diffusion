# Открытые датасеты с реальными фото домашних кошек для диффузии 512x512

Дата проверки: 2026-11-02. Цель: совместное обучение на чужих кошках и Терре, латент 64x64x4, кроп обязан содержать весь бокс кошки.

## 1. Вывод

- Основа: **iNaturalist, Felis catus** (короткая сторона >=512 у 99,4% выборки; 59 374 наблюдения с пометкой captive и CC-лицензией, 139 507 со всеми CC) плюс **Open Images V7, класс Cat** (13 873 кадра с боксами, короткая сторона >=512 у 99% выборки).
- Вместе это примерно 72 тыс. кадров >=512 до фильтра детектором, а с уличными кошками iNaturalist примерно 150 тыс.
- Готового набора "десятки тысяч фото кошек целиком, >=512, с боксами" не нашлось: LSUN Cat большой (1,66 млн), но измеренная короткая сторона не больше 400; AFHQ это 512x512, но только морды.
- Самое близкое к Терре по породе: метки Birman (от 537 кадров), Ragdoll (от 591), Himalayan (от 5 804) и Domestic long-haired cat (от 258) в Open Images и около 200 кадров Birman в Oxford-IIIT Pet (у Oxford >=512 только около 1%).

Обозначения: **[И]** измерено мной, **[З]** заявлено источником, **[О]** моя оценка.

## 2. Ранжированная таблица

| # | Датасет | Кошек | Разрешение: короткая сторона | Содержание кадра | Разметка | Лицензия | Размер, скачивание | Вердикт |
|---|---------|-------|------------------------------|------------------|----------|----------|--------------------|---------|
| 1 | iNaturalist, Felis catus | 197 075 наблюдений с фото; CC 139 507; captive+CC 59 374 [И] | p50 1536, >=512 99,4%, >=1024 90,3%, n=360 [И] | Глазами не проверено. Есть уличные; captive косвенно указывает на домашних | Боксов нет, вид подтверждён сообществом только частично | На каждое фото своя; чаще всего CC BY-NC | Порядка 20–60 ГБ [О], бакет S3 по photo_id | **Брать** |
| 2 | Open Images V7, Cat | 13 873 с боксами; 56 747 с меткой в train [И] | p50 686, >=512 99,0%, n=200, все классы [И]; длинная сторона до 1024 [З] | Повседневные фото Flickr; медианный бокс 50% кадра [И] | Боксы, метки пород на уровне изображения | Аннотации CC BY 4.0, картинки CC BY 2.0 [З] | Около 4,4 ГБ на 13,9 тыс. [О], downloader.py по ID | **Брать** |
| 3 | Cat Individuals (Kaggle) | 13 536, всего 518 особей [З] | От 195x261 до 4608x3453 [З]; доля >=512 не измерена, медиана файла 889 КБ [И] | Приюты, смартфоны [З]; предназначен для распознавания морд | ID особи | CC BY 4.0 (метка загрузившего) | 11,2 ГБ, нужен аккаунт Kaggle | Брать после замера |
| 4 | LSUN Cat | 1 657 266 [З] | p50 399, максимум 400, >=512 0%, n=608 [И] | Фото из веб-поиска; глазами не проверено | Только класс | Не указана [З] | 42 ГБ, cat.zip | Резерв для предобучения с апскейлом x1,3 |
| 5 | Pexels (photo-concept-bucket) | Около 3,5–5 тыс. [О] | >=1024 у 100%, n=160 [И] | Качественные фото, в т.ч. дом и диван | Теги, подписи | Условия сайта запрещают автоскачивание для ML | Только URL; копия на 130 тыс. фото 205 ГБ | Лучшее качество, но см. запрет |
| 6 | LCW (Kaggle) | Около 140 тыс. особей [З]; около 361 тыс. файлов [О] | Не измерено; медиана файла 55 КБ [И] | Объявления PetFinder | ID особи по папкам | Метка Apache 2.0 у загрузившего, права на фото неясны | 22 ГБ, нужен аккаунт Kaggle | Нужен замер |
| 7 | AFHQ / AFHQv2, Cat | Около 5 000 [З] | 512x512 [З, И] | Только морды | Нет | CC BY-NC 4.0 [З] | Весь набор 729 МБ на HF | Не для цели |
| 8 | Oxford-IIIT Pet, кошки | Около 2 400, 12 пород [З] | p50 345, >=512 1,0%, n=300, все классы [И] | Питомцы целиком и крупно | Породы, бокс головы, trimap | CC BY-SA 4.0 [З] | Около 800 МБ | Только файнтюн по породе |
| 9 | CAT dataset (Kaggle crawford) | 9 993 на HF [И] | p50 375, >=512 29%, n=300 [И] | Разное; разметка на головы | 9 точек головы | CC0 у загрузившего | 2,2 ГБ | Около 2,9 тыс. кадров [О], слабо |
| 10 | COCO 2017, cat | 4 298, 4 968 экземпляров [И] | Весь набор кошек: p50 449, >=512 6,8% train, 4,9% val; длинная сторона до 640 [И] | Сцены; медианный бокс 21% кадра [И] | Боксы, маски | На фото свои CC, много NC [И] | Весь набор большой; можно брать по ID | Около 290 кадров, нет |
| 11 | ImageNet-21k, кошачьи синсеты | До 31 794 вместе с дикими [И, неполно] | p50 375, >=512 2,3%, n=260 [И] | Разное | Класс | Некоммерческое использование [З] | Зеркало 3,9 ТБ целиком | Нет |
| 12 | Cats vs Dogs (PetImages) | Около 11,7 тыс. [О] | p50 371, >=512 0%, максимум 500, n=300 [И] | Приюты | Класс | На HF unknown | 722 МБ | Нет |

## 3. Лучшие кандидаты

### iNaturalist, Felis catus

- Вид [Felis catus, taxon 118552](https://api.inaturalist.org/v1/taxa/118552). Счётчики сняты через публичный API: [с фото](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&per_page=0) 197 075, [captive](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&captive=true&per_page=0) 91 677, [captive с любой CC](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&captive=true&photo_license=cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd&per_page=0) 59 374.
- Только CC0, CC BY и CC BY-SA: [12 851](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&photo_license=cc0,cc-by,cc-by-sa&per_page=0), из них captive 4 164.
- Размеры измерены по `original_dimensions` на 360 первых фото из 12 точек диапазона ID. Самая частая пара 1536x2048. Лицензии в той же выборке: CC BY-NC 233, без лицензии 96, CC BY 15, CC0 7, остальные CC 9.
- В другой выборке было около 1,5 фото на наблюдение (304 на 200), так что кадров больше, чем наблюдений.
- Скачивание: [бакет inaturalist-open-data](https://github.com/inaturalist/inaturalist-open-data), размеры original до 2048, large до 1024, medium до 500; метаданные в CSV обновляются ежемесячно. Брать нужно large или original: у medium сторона не больше 500.
- Минусы: боксов нет, значит нужен прогон детектора (в проекте уже есть `detect_cats.py` на Faster R-CNN). Кошка может быть мелкой и далеко. Фильтр captive отражает пометку пользователя, а не обстановку кадра.

### Open Images V7, Cat

- Класс `/m/01yrx` в [списке boxable-классов](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv).
- Посчитано потоково по официальным файлам: [train](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) 15 183 бокса на 12 516 кадрах, [validation](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) 345 кадров, [test](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv) 1 012.
- В train 831 кадр состоит только из изображений кошек (`IsDepiction`), то есть не фото. Кадров хотя бы с одним боксом без `IsDepiction` и `IsGroupOf` 11 597, с одной такой кошкой 10 186.
- Бокс одиночной кошки занимает медианно 50% площади (p25 25%, p75 76%). `IsTruncated` стоит у 51% одиночных кошек, поэтому не обрезанных кадром примерно половина.
- Без усечения и с шириной бокса не больше 0,75 кадра 46% боксов: это грубая прикидка того, что поместится в квадратный кроп у кадра 4:3.
- [Человеческие image-level метки](https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv): 56 747 положительных кадров Cat в train, из них с боксом 12 516.
- Породы, положительные человеческие метки в train: Himalayan 5 804, Siamese 722, Ragdoll 591, Birman 537, Persian 848, Domestic long-haired cat 258. Это нижняя граница: проход по файлу я прервал, не дойдя до конца. Среди этих кадров с боксом Cat только около 20% (Birman 119, Ragdoll 161, Himalayan 1 198). Точность меток пород глазами не проверена.
- Разрешение: [CVDF заявляет](https://github.com/cvdfoundation/open-images-dataset) не больше 1024 по длинной стороне. Замер по [зеркалу на HF](https://huggingface.co/datasets/vikhyatk/openimages-bbox) дал p50 686 и 99% >=512 на 200 случайных кадрах всех классов, а не только кошек.
- CVDF хостит только подмножество с боксами (train 513 ГБ, всего 561 ГБ). Полный набор около 18 ТБ, поэтому 44 тыс. кадров без бокса придётся брать по исходным URL Flickr, и их доступность я не проверял.
- Скачивание выборочно: [downloader.py](https://storage.googleapis.com/openimages/web/download_v7.html) принимает список `$SPLIT/$IMAGE_ID`.
- Лицензия: [аннотации CC BY 4.0, картинки указаны как CC BY 2.0](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), Google не гарантирует статус лицензии отдельных картинок.

### Cat Individuals

- [Карточка Kaggle](https://www.kaggle.com/datasets/timost1234/cat-individuals), [API](https://www.kaggle.com/api/v1/datasets/view/timost1234/cat-individuals): 13 536 фото 518 кошек, 11,2 ГБ, CC BY 4.0, снято камерами и смартфонами в приютах Тайбэя, часть из соцсетей.
- В первых 1 200 файлах медиана 889 КБ, p10 662 КБ, поэтому доля >=512 должна быть высокой, но это косвенный признак, а не замер.
- Назначение набора: распознавание морд, поэтому крупных морд может быть много. Глазами не проверял.

### LSUN Cat

- 1 657 266 кадров [по TFDS](https://www.tensorflow.org/datasets/catalog/lsun), [cat.zip 42 ГБ на официальном сервере](http://dl.yf.io/lsun/objects/).
- [README](https://github.com/fyu/lsun) говорит о меньшей стороне 256, но для объектов это неверно: чтение заголовков WEBP из четырёх участков архива (608 кадров) дало p50 399, максимум 400, >=384 около 60%, >=512 0%.
- Лицензия в README не указана. Разметка полуавтоматическая ([статья](https://arxiv.org/abs/1506.03365)), поэтому нужен свой фильтр.

### Pexels

- [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket): 567 597 записей с URL и размерами, [подписи InternVL2](https://huggingface.co/datasets/CaptionEmporium/pexels-568k-internvl2).
- Фильтр `caption ILIKE '% cat %'` дал 4 819 строк, из них в случайной дюжине 3 ложных. Тег `kitten` дал 2 433 строки. В выборках 80+80 короткая сторона везде >=1024.
- Запрет: [§8 условий Pexels](https://www.pexels.com/terms-of-service/) запрещает скрапинг и автоматический сбор, в том числе "for machine learning purposes", и массовое копирование без разрешения. [Лицензия](https://www.pexels.com/license/) про ML молчит, но условия сайта это не отменяет.

## 4. Что отброшено

- **ma7555/cat-breeds-dataset**: 67 пород, [2,1 ГБ](https://www.kaggle.com/api/v1/datasets/view/ma7555/cat-breeds-dataset), около 123 тыс. файлов [О], медиана файла всего 14 КБ, значит картинки мелкие. Породы проставлены [авторами объявлений](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset).
- **PetFinder Pawpularity**: [9 912 фото](https://www.kaggle.com/c/petfinder-pawpularity-score/rules), кошки и собаки без метки вида, только некоммерческое использование, данные нельзя передавать тем, кто не принял правила. Правила прочитаны по выдержке поиска.
- **HF-мелочь**: [cvdl/catfaces](https://huggingface.co/datasets/cvdl/catfaces) 64x64 [И]; [Rekzi/cat-breeds](https://huggingface.co/datasets/Rekzi/cat-breeds) >=512 0% [И]; [navhci/cat-breed-class](https://huggingface.co/datasets/navhci/cat-breed-class) >=512 0% [И].
- **Копии AFHQ**: [ziyang06315/cats_images_dataset](https://huggingface.co/datasets/ziyang06315/cats_images_dataset), [dimensi0n/afhq-512](https://www.kaggle.com/datasets/dimensi0n/afhq-512). Метка CC0 у второго противоречит [CC BY-NC оригинала](https://github.com/clovaai/stargan-v2).
- **huggan/cats**: [19 994 строки](https://huggingface.co/datasets/huggan/cats), >=512 24,7% [И]. Ровно вдвое больше строк и байт, чем [копия CAT dataset](https://huggingface.co/datasets/yashikota/cat-image-dataset), похоже на дубли; не проверял.
- **COCO, ImageNet, Cats vs Dogs**: не проходят по разрешению, см. таблицу.
- **Синтетика**: [facebook/PUG_Animals](https://huggingface.co/datasets/facebook/PUG_Animals), [Prgckwb/fake-animals](https://huggingface.co/datasets/Prgckwb/fake-animals) исключены по названию и назначению.
- **ImageNet-1k дополнительно**: доступ [только с принятием условий, некоммерческие исследования и обучение](https://huggingface.co/datasets/ILSVRC/imagenet-1k); разрешение оценено по тем же классам в [зеркале 21k](https://huggingface.co/datasets/gmongaras/Imagenet21K).

## 5. Что не удалось проверить

- **Вид кадров.** Изображения я глазами не смотрел. "Кошка целиком, дома" для iNaturalist, Open Images, LCW и Cat Individuals не подтверждено, для этого нужен просмотр выборки и прогон детектора.
- **Kaggle, разрешение.** Без токена API отдаёт карточки и списки файлов с размерами, но не сами файлы. Поэтому не измерены LCW, Cat Individuals, ma7555, [cat-faces-detection](https://www.kaggle.com/datasets/aleksandrdremov/cat-faces-detection) (50 тыс. фото с боксами голов, 7,2 ГБ), [pop-cats](https://www.kaggle.com/datasets/knucharat/pop-cats) (20 пород по 500, есть Birman, Himalayan, Ragdoll, Siamese), [yapwh1208/cats-breed-dataset](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset), [Gano](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection) и [Oxford Cats Extended-10k](https://www.kaggle.com/datasets/doctrinek/oxford-iiit-cats-extended-10k).
- **Open Images.** Хосты S3 не резолвились из песочницы, поэтому кадры именно с кошками не измерены; число дано по всем классам зеркала. Маски для Cat не считал. Подсчёт меток пород прерван до конца файла, числа занижены.
- **iNaturalist.** Страница для разработчиков с лимитами вернула 403. Прямое скачивание фото из S3 не проверялось. Перед массовой выгрузкой лимиты нужно прочитать.
- **ImageNet-21k.** 20 из 46 блоков не ответили, поэтому синсеты после Persian_cat не увидены, разделить домашних и диких в диапазоне не удалось.
- **PD12M, Unsplash, LAION.** Индекс поиска [PD12M](https://huggingface.co/datasets/Spawning/PD12M) (12,4 млн, CC0/PD) вернул ошибку, число кошек неизвестно. У [unsplash-lite](https://huggingface.co/datasets/1aurent/unsplash-lite) поиск упал, весь набор по тегу HF меньше 100 тыс. фото, число кошек неизвестно. [relaion2B](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) закрыт за принятием условий.
- **Не проверялись:** Objects365, SA-1B, Places (класса кошек там нет по замыслу, но подтверждения я не искал), конкурсный iNaturalist 2021.
- **Официальные страницы.** Сайт Oxford Pets и image-net.org отказали в соединении, лицензия Oxford взята из выдержки поиска по этой странице. Условия COCO и лицензия Microsoft PetImages не прочитаны.
- **Wikimedia Commons.** [41 192 растровых файла](https://commons.wikimedia.org/w/api.php?action=query&list=search&srnamespace=6&srsearch=haswbstatement:P180=Q146%20filetype:bitmap&srlimit=1&format=json) с пометкой "изображает кошку" [И]. Разрешение и лицензии не оценены, в выборку могут входить не фото.

## 6. Рекомендация

Комбинация для 512x512:

| Источник | Вход | >=512, оценка | Кадров с кошкой целиком, грубо |
|----------|------|---------------|-------------------------------|
| iNaturalist, captive + CC | 59 374 набл. | около 59 000 | неизвестно, зависит от детектора |
| Open Images, боксы | 13 873 | около 12 700 без изображений и групп | около 5 900 |
| Итого основа | | **около 72 000** | |
| iNaturalist, остальные CC (улица) | 80 133 набл. | около 79 600 | неизвестно |
| Cat Individuals, после замера | 13 536 | не измерено | неизвестно |

Порядок действий:

1. Скачать Open Images по ID, взять фото iNaturalist в размере large или original, сохранив лицензию и автора каждого фото.
2. Прогнать iNaturalist через `detect_cats.py`, оставить кадры, где бокс целиком входит в кроп и занимает заметную долю кадра. Если детектор отсеет половину, останется около 30 тыс. с iNaturalist и около 36 тыс. в основе.
3. Для файнтюна добавить Oxford Birman, Ragdoll, Siamese и Persian с апскейлом, отдельно от основы.
4. LSUN брать только для предобучения: апскейл x1,3 даст мягкие кадры, и это стоит держать отдельным этапом, а не смешивать с основой.

Ограничения по лицензиям:

- **Некоммерческие:** большинство фото iNaturalist (CC BY-NC), AFHQ (CC BY-NC 4.0), ImageNet (только исследования и обучение), Pawpularity. Для личного учебного проекта подходит, но модель и результаты нельзя продавать.
- **Атрибуция:** Open Images (CC BY 2.0), iNaturalist (все CC, кроме CC0), Cat Individuals.
- **ShareAlike и NoDerivatives:** Oxford-IIIT Pet (CC BY-SA 4.0), часть iNaturalist. Фото с ND разумнее исключить, раз кропы являются производной работой.
- **Прямой запрет:** Pexels, §8 условий, для ML и автоматического сбора. Рекомендую не включать Pexels в набор.
- **Нет права на публикацию:** Pawpularity нельзя передавать дальше. Метки лицензий от загрузивших на Kaggle не подтверждают прав на чужие фото (LCW, копии AFHQ, CAT dataset).
