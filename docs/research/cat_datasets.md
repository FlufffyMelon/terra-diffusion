# Open datasets with real photos of domestic cats for 512x512 diffusion

Date checked: 2026-11-02. Goal: joint training on other people's cats and Terra, latent 64x64x4, the crop must contain the whole cat box.

## 1. Conclusion

- Base: **iNaturalist, Felis catus** (short side >=512 for 99.4% of the sample; 59 374 observations marked captive with a CC license, 139 507 with any CC) plus **Open Images V7, class Cat** (13 873 frames with boxes, short side >=512 for 99% of the sample).
- Together that is about 72 thousand frames >=512 before the detector filter, and with outdoor iNaturalist cats about 150 thousand.
- No ready-made set of "tens of thousands of whole-cat photos, >=512, with boxes" was found: LSUN Cat is large (1.66 million), but the measured short side is at most 400; AFHQ is 512x512, but faces only.
- Closest to Terra by breed: the labels Birman (from 537 frames), Ragdoll (from 591), Himalayan (from 5 804) and Domestic long-haired cat (from 258) in Open Images and about 200 Birman frames in Oxford-IIIT Pet (only about 1% of Oxford is >=512).

Notation: **[M]** measured by me, **[C]** claimed by the source, **[E]** my estimate.

## 2. Ranked table

| # | Dataset | Cats | Resolution: short side | Frame content | Annotations | License | Size, download | Verdict |
|---|---------|------|------------------------|---------------|-------------|---------|----------------|---------|
| 1 | iNaturalist, Felis catus | 197 075 observations with photos; CC 139 507; captive+CC 59 374 [M] | p50 1536, >=512 99.4%, >=1024 90.3%, n=360 [M] | Not checked by eye. Includes outdoor cats; captive indirectly points to domestic ones | No boxes, the species is only partly confirmed by the community | Its own for each photo; most often CC BY-NC | On the order of 20-60 GB [E], S3 bucket by photo_id | **Take** |
| 2 | Open Images V7, Cat | 13 873 with boxes; 56 747 with a label in train [M] | p50 686, >=512 99.0%, n=200, all classes [M]; long side up to 1024 [C] | Everyday Flickr photos; median box 50% of the frame [M] | Boxes, breed labels at image level | Annotations CC BY 4.0, images CC BY 2.0 [C] | About 4.4 GB for 13.9 thousand [E], downloader.py by ID | **Take** |
| 3 | Cat Individuals (Kaggle) | 13 536, 518 individuals in total [C] | From 195x261 to 4608x3453 [C]; share >=512 not measured, median file 889 KB [M] | Shelters, smartphones [C]; intended for face recognition | Individual ID | CC BY 4.0 (uploader's label) | 11.2 GB, Kaggle account needed | Take after measuring |
| 4 | LSUN Cat | 1 657 266 [C] | p50 399, max 400, >=512 0%, n=608 [M] | Web search photos; not checked by eye | Class only | Not specified [C] | 42 GB, cat.zip | Reserve for pretraining with x1.3 upscaling |
| 5 | Pexels (photo-concept-bucket) | About 3.5-5 thousand [E] | >=1024 for 100%, n=160 [M] | High-quality photos, including home and sofa | Tags, captions | The site terms forbid automated download for ML | URLs only; a copy of 130 thousand photos is 205 GB | Best quality, but see the ban |
| 6 | LCW (Kaggle) | About 140 thousand individuals [C]; about 361 thousand files [E] | Not measured; median file 55 KB [M] | PetFinder listings | Individual ID by folder | Apache 2.0 label from the uploader, rights to the photos unclear | 22 GB, Kaggle account needed | Needs measuring |
| 7 | AFHQ / AFHQv2, Cat | About 5 000 [C] | 512x512 [C, M] | Faces only | None | CC BY-NC 4.0 [C] | Whole set 729 MB on HF | Not for the goal |
| 8 | Oxford-IIIT Pet, cats | About 2 400, 12 breeds [C] | p50 345, >=512 1.0%, n=300, all classes [M] | Whole pets and close-ups | Breeds, head box, trimap | CC BY-SA 4.0 [C] | About 800 MB | Only for fine-tuning by breed |
| 9 | CAT dataset (Kaggle crawford) | 9 993 on HF [M] | p50 375, >=512 29%, n=300 [M] | Various; annotated on heads | 9 head points | CC0 from the uploader | 2.2 GB | About 2.9 thousand frames [E], weak |
| 10 | COCO 2017, cat | 4 298, 4 968 instances [M] | All cat images: p50 449, >=512 6.8% train, 4.9% val; long side up to 640 [M] | Scenes; median box 21% of the frame [M] | Boxes, masks | Its own CC per photo, many NC [M] | The whole set is large; can be taken by ID | About 290 frames, no |
| 11 | ImageNet-21k, cat synsets | Up to 31 794 together with wild ones [M, incomplete] | p50 375, >=512 2.3%, n=260 [M] | Various | Class | Non-commercial use [C] | Mirror of 3.9 TB in full | No |
| 12 | Cats vs Dogs (PetImages) | About 11.7 thousand [E] | p50 371, >=512 0%, max 500, n=300 [M] | Shelters | Class | unknown on HF | 722 MB | No |

## 3. Best candidates

### iNaturalist, Felis catus

- Species [Felis catus, taxon 118552](https://api.inaturalist.org/v1/taxa/118552). The counts were taken through the public API: [with photos](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&per_page=0) 197 075, [captive](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&captive=true&per_page=0) 91 677, [captive with any CC](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&captive=true&photo_license=cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd&per_page=0) 59 374.
- Only CC0, CC BY and CC BY-SA: [12 851](https://api.inaturalist.org/v1/observations?taxon_id=118552&photos=true&photo_license=cc0,cc-by,cc-by-sa&per_page=0), of which captive 4 164.
- Sizes were measured by `original_dimensions` on the first 360 photos from 12 points of the ID range. The most common pair is 1536x2048. Licenses in the same sample: CC BY-NC 233, no license 96, CC BY 15, CC0 7, other CC 9.
- In another sample there were about 1.5 photos per observation (304 for 200), so there are more frames than observations.
- Download: [inaturalist-open-data bucket](https://github.com/inaturalist/inaturalist-open-data), sizes original up to 2048, large up to 1024, medium up to 500; the metadata in CSV is updated monthly. Take large or original: medium has a side of at most 500.
- Downsides: no boxes, so a detector run is needed (the project already has `detect_cats.py` on Faster R-CNN). The cat may be small and far away. The captive filter reflects the user's label, not the setting of the frame.

### Open Images V7, Cat

- Class `/m/01yrx` in the [list of boxable classes](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv).
- Counted by streaming the official files: [train](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) 15 183 boxes on 12 516 frames, [validation](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) 345 frames, [test](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv) 1 012.
- In train 831 frames consist only of images of cats (`IsDepiction`), i.e. not photos. Frames with at least one box without `IsDepiction` and `IsGroupOf`: 11 597, with a single such cat: 10 186.
- The box of a single cat takes a median of 50% of the area (p25 25%, p75 76%). `IsTruncated` is set for 51% of single cats, so about half are not cut off by the frame.
- Without truncation and with a box width of at most 0.75 of the frame: 46% of the boxes. This is a rough estimate of what will fit into a square crop of a 4:3 frame.
- [Human image-level labels](https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv): 56 747 positive Cat frames in train, of which 12 516 have a box.
- Breeds, positive human labels in train: Himalayan 5 804, Siamese 722, Ragdoll 591, Birman 537, Persian 848, Domestic long-haired cat 258. This is a lower bound: I interrupted the pass over the file before the end. Among these frames only about 20% have a Cat box (Birman 119, Ragdoll 161, Himalayan 1 198). The accuracy of the breed labels was not checked by eye.
- Resolution: [CVDF states](https://github.com/cvdfoundation/open-images-dataset) at most 1024 on the long side. A measurement on the [HF mirror](https://huggingface.co/datasets/vikhyatk/openimages-bbox) gave p50 686 and 99% >=512 on 200 random frames of all classes, not only cats.
- CVDF hosts only the subset with boxes (train 513 GB, 561 GB in total). The full set is about 18 TB, so the 44 thousand frames without a box would have to be taken from the original Flickr URLs, and I did not check their availability.
- Selective download: [downloader.py](https://storage.googleapis.com/openimages/web/download_v7.html) accepts a list of `$SPLIT/$IMAGE_ID`.
- License: [annotations CC BY 4.0, images listed as CC BY 2.0](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), Google does not guarantee the license status of individual images.

### Cat Individuals

- [Kaggle page](https://www.kaggle.com/datasets/timost1234/cat-individuals), [API](https://www.kaggle.com/api/v1/datasets/view/timost1234/cat-individuals): 13 536 photos of 518 cats, 11.2 GB, CC BY 4.0, taken with cameras and smartphones in Taipei shelters, some from social networks.
- In the first 1 200 files the median is 889 KB, p10 662 KB, so the share >=512 should be high, but this is an indirect sign, not a measurement.
- Purpose of the set: face recognition, so there may be many close-up faces. Not checked by eye.

### LSUN Cat

- 1 657 266 frames [per TFDS](https://www.tensorflow.org/datasets/catalog/lsun), [cat.zip 42 GB on the official server](http://dl.yf.io/lsun/objects/).
- The [README](https://github.com/fyu/lsun) speaks of a smaller side of 256, but this is wrong for objects: reading the WEBP headers from four parts of the archive (608 frames) gave p50 399, max 400, >=384 about 60%, >=512 0%.
- The license is not specified in the README. The annotation is semi-automatic ([paper](https://arxiv.org/abs/1506.03365)), so a filter of our own is needed.

### Pexels

- [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket): 567 597 records with URLs and sizes, [InternVL2 captions](https://huggingface.co/datasets/CaptionEmporium/pexels-568k-internvl2).
- The filter `caption ILIKE '% cat %'` gave 4 819 rows, of which 3 in a random dozen were false. The tag `kitten` gave 2 433 rows. In the 80+80 samples the short side is >=1024 everywhere.
- Ban: [section 8 of the Pexels terms](https://www.pexels.com/terms-of-service/) forbids scraping and automated collection, including "for machine learning purposes", and bulk copying without permission. The [license](https://www.pexels.com/license/) is silent about ML, but the site terms are not overridden by it.

## 4. What was rejected

- **ma7555/cat-breeds-dataset**: 67 breeds, [2.1 GB](https://www.kaggle.com/api/v1/datasets/view/ma7555/cat-breeds-dataset), about 123 thousand files [E], the median file is only 14 KB, so the images are small. Breeds were assigned by [the listing authors](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset).
- **PetFinder Pawpularity**: [9 912 photos](https://www.kaggle.com/c/petfinder-pawpularity-score/rules), cats and dogs without a species label, non-commercial use only, the data must not be passed to those who have not accepted the rules. The rules were read from a search excerpt.
- **Small HF sets**: [cvdl/catfaces](https://huggingface.co/datasets/cvdl/catfaces) 64x64 [M]; [Rekzi/cat-breeds](https://huggingface.co/datasets/Rekzi/cat-breeds) >=512 0% [M]; [navhci/cat-breed-class](https://huggingface.co/datasets/navhci/cat-breed-class) >=512 0% [M].
- **AFHQ copies**: [ziyang06315/cats_images_dataset](https://huggingface.co/datasets/ziyang06315/cats_images_dataset), [dimensi0n/afhq-512](https://www.kaggle.com/datasets/dimensi0n/afhq-512). The CC0 label of the second contradicts the [CC BY-NC of the original](https://github.com/clovaai/stargan-v2).
- **huggan/cats**: [19 994 rows](https://huggingface.co/datasets/huggan/cats), >=512 24.7% [M]. Exactly twice the rows and bytes of the [CAT dataset copy](https://huggingface.co/datasets/yashikota/cat-image-dataset), looks like duplicates; not checked.
- **COCO, ImageNet, Cats vs Dogs**: do not pass on resolution, see the table.
- **Synthetic**: [facebook/PUG_Animals](https://huggingface.co/datasets/facebook/PUG_Animals), [Prgckwb/fake-animals](https://huggingface.co/datasets/Prgckwb/fake-animals) are excluded by name and purpose.
- **ImageNet-1k in addition**: access [only after accepting the terms, non-commercial research and training](https://huggingface.co/datasets/ILSVRC/imagenet-1k); the resolution was estimated on the same classes in the [21k mirror](https://huggingface.co/datasets/gmongaras/Imagenet21K).

## 5. What could not be checked

- **Look of the frames.** I did not look at the images by eye. "Whole cat, at home" is not confirmed for iNaturalist, Open Images, LCW and Cat Individuals; for that a sample must be viewed and a detector run.
- **Kaggle, resolution.** Without a token the API returns the pages and file lists with sizes, but not the files themselves. So LCW, Cat Individuals, ma7555, [cat-faces-detection](https://www.kaggle.com/datasets/aleksandrdremov/cat-faces-detection) (50 thousand photos with head boxes, 7.2 GB), [pop-cats](https://www.kaggle.com/datasets/knucharat/pop-cats) (20 breeds of 500 each, includes Birman, Himalayan, Ragdoll, Siamese), [yapwh1208/cats-breed-dataset](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset), [Gano](https://www.kaggle.com/datasets/shawngano/gano-cat-breed-image-collection) and [Oxford Cats Extended-10k](https://www.kaggle.com/datasets/doctrinek/oxford-iiit-cats-extended-10k) were not measured.
- **Open Images.** The S3 hosts did not resolve from the sandbox, so frames with cats specifically were not measured; the number is given over all classes of the mirror. I did not count masks for Cat. The count of breed labels was interrupted before the end of the file, the numbers are underestimated.
- **iNaturalist.** The developer page with the limits returned 403. Direct download of photos from S3 was not checked. The limits must be read before a bulk download.
- **ImageNet-21k.** 20 of 46 blocks did not respond, so the synsets after Persian_cat were not seen, and domestic and wild ones could not be separated in the range.
- **PD12M, Unsplash, LAION.** The search index of [PD12M](https://huggingface.co/datasets/Spawning/PD12M) (12.4 million, CC0/PD) returned an error, the number of cats is unknown. For [unsplash-lite](https://huggingface.co/datasets/1aurent/unsplash-lite) the search failed, the whole set by the HF tag is under 100 thousand photos, the number of cats is unknown. [relaion2B](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) is gated behind accepting terms.
- **Not checked:** Objects365, SA-1B, Places (there is no cat class there by design, but I did not look for confirmation), the competition iNaturalist 2021.
- **Official pages.** The Oxford Pets site and image-net.org refused the connection, the Oxford license was taken from a search excerpt of that page. The COCO terms and the Microsoft PetImages license were not read.
- **Wikimedia Commons.** [41 192 raster files](https://commons.wikimedia.org/w/api.php?action=query&list=search&srnamespace=6&srsearch=haswbstatement:P180=Q146%20filetype:bitmap&srlimit=1&format=json) marked "depicts a cat" [M]. Resolution and licenses were not assessed, the sample may include non-photos.

## 6. Recommendation

Combination for 512x512:

| Source | Input | >=512, estimate | Frames with a whole cat, roughly |
|--------|-------|-----------------|----------------------------------|
| iNaturalist, captive + CC | 59 374 obs. | about 59 000 | unknown, depends on the detector |
| Open Images, boxes | 13 873 | about 12 700 without depictions and groups | about 5 900 |
| Base total | | **about 72 000** | |
| iNaturalist, other CC (outdoor) | 80 133 obs. | about 79 600 | unknown |
| Cat Individuals, after measuring | 13 536 | not measured | unknown |

Order of work:

1. Download Open Images by ID, take iNaturalist photos in large or original size, keeping the license and author of each photo.
2. Run iNaturalist through `detect_cats.py`, keep the frames where the box fits entirely into the crop and takes a noticeable share of the frame. If the detector discards half, about 30 thousand remain from iNaturalist and about 36 thousand in the base.
3. For fine-tuning add Oxford Birman, Ragdoll, Siamese and Persian with upscaling, separately from the base.
4. Take LSUN only for pretraining: x1.3 upscaling gives soft frames, and it is worth keeping as a separate stage rather than mixing with the base.

License restrictions:

- **Non-commercial:** most iNaturalist photos (CC BY-NC), AFHQ (CC BY-NC 4.0), ImageNet (research and training only), Pawpularity. Fine for a personal learning project, but the model and results must not be sold.
- **Attribution:** Open Images (CC BY 2.0), iNaturalist (all CC except CC0), Cat Individuals.
- **ShareAlike and NoDerivatives:** Oxford-IIIT Pet (CC BY-SA 4.0), part of iNaturalist. Photos with ND are better excluded, since crops are a derivative work.
- **Direct ban:** Pexels, section 8 of the terms, for ML and automated collection. I recommend not including Pexels in the set.
- **No right to publish:** Pawpularity must not be passed on. License labels from uploaders on Kaggle do not confirm the rights to other people's photos (LCW, AFHQ copies, CAT dataset).
