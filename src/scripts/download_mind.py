"""
MIND Dataset Downloader

This script downloads the Microsoft News Dataset (MIND) from official URLs,
extracts the zip files automatically, and verifies file integrity.

Usage:
    python -m src.scripts.download_mind --size small --output-dir Data/raw
    python -m src.scripts.download_mind --size large --verify
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger, setup_logger


# Official MIND dataset URLs
MIND_URLS = {
    "small": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    },
    "large": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
        "test": "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip",
    }
}

# Expected files in each dataset split
EXPECTED_FILES = {
    "train": ["news.tsv", "behaviors.tsv", "entity_embedding.vec", "relation_embedding.vec"],
    "dev": ["news.tsv", "behaviors.tsv", "entity_embedding.vec", "relation_embedding.vec"],
    "test": ["news.tsv", "behaviors.tsv"],
}


class DownloadProgressBar(tqdm):
    """Progress bar for download tracking"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar
        
        Args:
            b: Number of blocks transferred
            bsize: Size of each block (in bytes)
            tsize: Total size (in bytes)
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, logger) -> None:
    """
    Download a file from URL with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        logger: Logger instance
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Downloaded {output_path.name} ({file_size:.2f} MB)", file_size_mb=file_size)


def extract_zip(zip_path: Path, extract_to: Path, logger) -> None:
    """
    Extract a zip file with progress bar
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
        logger: Logger instance
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {zip_path.name} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc=f"Extracting {zip_path.name}") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)
    
    logger.info(f"Extracted {len(members)} files to {extract_to}", num_files=len(members))
    print(f"✅ Extracted to: {extract_to}")


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Compute hash of a file for integrity verification
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (md5, sha256, etc.)
    
    Returns:
        Hex digest of the file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def verify_extracted_files(extract_dir: Path, split: str, logger) -> Tuple[bool, list]:
    """
    Verify that all expected files exist after extraction
    
    Args:
        extract_dir: Directory where files were extracted
        split: Dataset split (train, dev, test)
        logger: Logger instance
    
    Returns:
        Tuple of (all_files_exist, missing_files)
    """
    expected = EXPECTED_FILES.get(split, [])
    missing_files = []
    
    logger.debug(f"Verifying {len(expected)} expected files in {extract_dir}")
    
    for filename in expected:
        file_path = extract_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
            logger.warning(f"Missing expected file: {filename}")
        else:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.debug(f"Found {filename} ({file_size:.2f} MB)")
    
    if len(missing_files) == 0:
        logger.info(f"All {len(expected)} expected files verified successfully")
    else:
        logger.error(f"Verification failed: {len(missing_files)} files missing")
    
    return len(missing_files) == 0, missing_files


def download_mind_dataset(
    size: str = "small",
    output_dir: Optional[Path] = None,
    verify: bool = False,
    keep_zip: bool = False,
    log_level: str = "INFO"
) -> None:
    """
    Download and extract MIND dataset
    
    Args:
        size: Dataset size ('small' or 'large')
        output_dir: Output directory for downloaded files
        verify: Whether to verify file integrity after extraction
        keep_zip: Whether to keep zip files after extraction
        log_level: Logging level
    """
    # Setup logger
    logger_instance = setup_logger(
        name="download_mind",
        log_level=log_level,
        console_output=True
    )
    logger = logger_instance.get_logger("download")
    
    if size not in MIND_URLS:
        logger.error(f"Invalid size: {size}. Must be 'small' or 'large'")
        raise ValueError(f"Invalid size: {size}. Must be 'small' or 'large'")
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path("Data") / "raw"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"Starting MIND-{size} dataset download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Verify files: {verify}")
    logger.info(f"Keep zip files: {keep_zip}")
    
    print(f"📥 Downloading MIND-{size} dataset...")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()
    
    urls = MIND_URLS[size]
    total_downloaded = 0
    total_extracted = 0
    
    for split, url in urls.items():
        logger.info(f"Processing {split} split")
        print(f"{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        # Download zip file
        zip_filename = f"MIND{size}_{split}.zip"
        zip_path = output_dir / zip_filename
        
        if zip_path.exists():
            logger.info(f"Zip file already exists: {zip_path}, skipping download")
            print(f"⚠️  Zip file already exists: {zip_path}")
            print(f"   Skipping download...")
        else:
            print(f"📥 Downloading from: {url}")
            try:
                download_file(url, zip_path, logger)
                total_downloaded += 1
                print(f"✅ Downloaded: {zip_path}")
            except Exception as e:
                logger.error(f"Error downloading {split}: {str(e)}")
                print(f"❌ Error downloading {split}: {e}")
                continue
        
        # Extract zip file
        extract_dir = output_dir / f"MIND{size}_{split}"
        
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Extract directory already exists: {extract_dir}, skipping extraction")
            print(f"⚠️  Extract directory already exists and is not empty: {extract_dir}")
            print(f"   Skipping extraction...")
        else:
            print(f"📦 Extracting {zip_filename}...")
            try:
                extract_zip(zip_path, extract_dir, logger)
                total_extracted += 1
            except Exception as e:
                logger.error(f"Error extracting {split}: {str(e)}")
                print(f"❌ Error extracting {split}: {e}")
                continue
        
        # Verify extracted files
        if verify:
            print(f"🔍 Verifying extracted files...")
            all_exist, missing = verify_extracted_files(extract_dir, split, logger)
            
            if all_exist:
                print(f"✅ All expected files present")
            else:
                print(f"⚠️  Missing files: {', '.join(missing)}")
        
        # Remove zip file if requested
        if not keep_zip and zip_path.exists():
            logger.info(f"Removing zip file: {zip_path}")
            print(f"🗑️  Removing zip file: {zip_path}")
            zip_path.unlink()
        
        print()
    
    logger.info("=" * 60)
    logger.info("MIND dataset download complete!")
    logger.info("=" * 60)
    logger.info(f"Splits downloaded: {total_downloaded}")
    logger.info(f"Splits extracted: {total_extracted}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)
    
    print(f"{'='*60}")
    print(f"✅ MIND-{size} dataset download complete!")
    print(f"📁 Files saved to: {output_dir.absolute()}")
    print(f"{'='*60}")


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Download Microsoft News Dataset (MIND)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download small dataset to default location (Data/raw)
  python -m src.scripts.download_mind --size small
  
  # Download large dataset to custom directory
  python -m src.scripts.download_mind --size large --output-dir /path/to/data
  
  # Download with file verification
  python -m src.scripts.download_mind --size small --verify
  
  # Keep zip files after extraction
  python -m src.scripts.download_mind --size small --keep-zip
        """
    )
    
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "large"],
        default="small",
        help="Dataset size to download (default: small)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded files (default: Data/raw)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify file integrity after extraction"
    )
    
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction (default: remove)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    try:
        download_mind_dataset(
            size=args.size,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            verify=args.verify,
            keep_zip=args.keep_zip,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
