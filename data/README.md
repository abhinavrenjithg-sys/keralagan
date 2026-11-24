# Building the datasets

Each translation is its own unpaired dataset:

```
data/<name>/trainA   trainB   testA   testB
```

You never need matching pairs. A and B just need to share roughly the same *kind* of content.

## 1. Summer → Monsoon (`data/monsoon`) — start here

| Domain | What to collect | Target |
|---|---|---|
| A | Sunny Kerala landscapes: fields, roads, backwaters, coconut groves | 500–1,000 |
| B | The same kinds of scenes under rain and heavy cloud | 500–1,000 |

Sources: your own phone photos and videos (best), frames from your own clips
(`scripts/extract_frames.py`), and openly licensed weather-image datasets on Kaggle.

## 2. Day → Night (`data/night`)

| Domain | What to collect | Target |
|---|---|---|
| A | Streets, temples, backwaters, houses in daylight | 500+ |
| B | Similar places after dark, with lamps and lit windows | 500+ |

Tip: shoot the same street at noon and at 8 pm. The images don't need to line up,
but similar content makes training much easier.

## 3. Photo → Kerala mural (`data/mural`) — the showpiece

| Domain | What to collect | Target |
|---|---|---|
| A | Photos of people, dancers, Kathakali performers, full-body figures | 400+ |
| B | Kerala mural paintings, cropped into patches | 300–600 patches |

Murals mostly show figures, so keep domain A to figures too. CycleGAN struggles
when the two domains have very different content.

Mural sources: Wikimedia Commons (filter by licence), and your own photos of murals at
places such as Mattancherry Palace and Krishnapuram Palace, where photography is allowed.
Turn a few dozen large photos into hundreds of patches:

```bash
python scripts/make_patches.py --src raw/murals_full --dst raw/mural_patches --size 384 --stride 256
```

## 4. Clean, de-duplicate and split

```bash
python scripts/prepare_dataset.py --a raw/summer  --b raw/monsoon       --name monsoon
python scripts/prepare_dataset.py --a raw/day     --b raw/night         --name night
python scripts/prepare_dataset.py --a raw/people  --b raw/mural_patches --name mural
```

This drops broken, tiny and duplicate files, shrinks images (short side 320 px)
and holds out 10% for testing.

## 5. Record every source

Add one row per source to `sources.csv` (who made it, where it came from, its licence).
Only redistribute images whose licence allows it; the repo's `.gitignore` keeps
`data/` out of git by default.
