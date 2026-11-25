"""Import local catalog photos; this module never downloads images.

python -m dl.download_images --source /path/to/photos
The source must contain <product_id>/*.jpg (or png/jpeg/webp/avif).
Without --source, count the existing local photos. Kept under its old module
name so catalog-refresh commands remain compatible.
"""
import argparse
import shutil
from pathlib import Path
from dl import config


def main():
    parser = argparse.ArgumentParser(description='Import or check local product photos; no network requests.')
    parser.add_argument('--source', type=Path, help='Folder containing product-id image folders')
    args = parser.parse_args()
    source = (args.source or config.CATALOG_DIR).resolve()
    if args.source and not source.is_dir():
        parser.error('the local source folder does not exist')
    count = 0
    for folder in sorted(source.iterdir()) if source.is_dir() else []:
        if not folder.is_dir() or folder.is_symlink():
            continue
        for image in sorted(folder.iterdir()):
            if image.is_file() and not image.is_symlink() and image.suffix.lower() in config.IMG_EXTS | {'.avif'}:
                target = config.CATALOG_DIR / folder.name / image.name
                if image.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, target)
                count += 1
    print(f'{count} local catalog photos; no images downloaded')


if __name__ == '__main__':
    main()
