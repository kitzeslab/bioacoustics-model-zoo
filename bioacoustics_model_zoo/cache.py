"""
Cache management utilities for bioacoustics model zoo.

Provides functionality to cache model files in OS-appropriate directories,
with user-configurable cache locations.
"""

from pathlib import Path
from bioacoustics_model_zoo.appdirs import user_cache_dir

# Global cache directory setting
_default_cache_dir = None


def get_default_cache_dir():
    """Get the default cache directory for the package.

    Returns:
        str: Path to default cache directory
    """
    global _default_cache_dir
    if _default_cache_dir is None:
        return user_cache_dir(appname="bioacoustics_model_zoo")
    return _default_cache_dir


def set_default_cache_dir(cache_dir):
    """Set the default cache directory for the package.

    Args:
        cache_dir (str or Path): Path to use as default cache directory
    """
    global _default_cache_dir
    _default_cache_dir = str(Path(cache_dir).resolve())


def get_model_cache_dir(model_name, model_version=None, cache_dir=None):
    """Get cache directory for a specific model.

    Args:
        model_name (str): Name of the model (e.g., 'birdnet', 'hawkears')
        model_version (str, optional): Version of the model.
            - if specified, does not consider model to be cached if version mismatches
            and stores model in [cache_dir]/[model_name]/[model_version]/
            - if not specified, assumes model has not changed versions and
            stores model in [cache_dir]/[model_name]/
        cache_dir (str or Path, optional): Override cache directory.
            If None, uses default cache directory.

    Returns:
        Path: Path to model-specific cache directory
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    if model_version is None:
        model_cache_path = Path(cache_dir) / model_name
    else:
        model_cache_path = Path(cache_dir) / model_name / str(model_version)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    return model_cache_path


def is_cached(filename, model_name, model_version=None, cache_dir=None):
    """Check if a file exists in the model's cache directory.

    Args:
        filename (str): Name of the file to check
        model_name (str): Name of the model
        model_version (str, optional): Version of the model.
            - if specified, does not consider model to be cached if version mismatches
            and stores model in [cache_dir]/[model_name]/[model_version]/
            - if not specified, assumes model has not changed versions since the last
            un-versioned download, and stores model in [cache_dir]/[model_name]/
        cache_dir (str or Path, optional): Override cache directory

    Returns:
        bool: True if file exists in cache, False otherwise
    """
    model_cache_path = get_model_cache_dir(
        model_name=model_name, model_version=model_version, cache_dir=cache_dir
    )
    file_path = model_cache_path / filename
    return file_path.exists()


def get_cached_file_path(filename, model_name, model_version=None, cache_dir=None):
    """Get the full path to a cached file.

    Args:
        filename (str): Name of the file
        model_name (str): Name of the model
        cache_dir (str or Path, optional): Override cache directory
        model_version (str, optional): Version of the model.
            - if specified, does not consider model to be cached if version mismatches
            and stores model in [cache_dir]/[model_name]/[model_version]/
            - if not specified, assumes model has not changed versions and
            stores model in [cache_dir]/[model_name]/

    Returns:
        Path: Full path to the cached file
    """
    model_cache_path = get_model_cache_dir(model_name, model_version, cache_dir)
    return model_cache_path / filename


def save_to_cache(
    source_path, filename, model_name, model_version=None, cache_dir=None
):
    """Save a file to the model's cache directory.

    Args:
        source_path (str or Path): Path to the source file
        filename (str): Name to save the file as in cache
        model_name (str): Name of the model
        model_version (str, optional): Version of the model.
            -
        cache_dir (str or Path, optional): Override cache directory

    Returns:
        Path: Path to the cached file
    """
    import shutil

    cached_file_path = get_cached_file_path(
        filename, model_name, model_version, cache_dir
    )

    # Copy file to cache if it doesn't already exist
    if not cached_file_path.exists():
        shutil.copy2(source_path, cached_file_path)

    return cached_file_path


def clear_cached_model(model_name, model_version=None, cache_dir=None):
    """Clear the cache for a specific model.

    If model_version is None, removes cached models for all versions.

    Args:
        model_name (str): Name of the model
        model_version (str, optional): Version of the model
        cache_dir (str or Path, optional): Override the default root cache directory
    """
    model_cache_path = get_model_cache_dir(model_name, model_version, cache_dir)
    if model_cache_path.exists():
        import shutil

        shutil.rmtree(model_cache_path)


def clear_all_cached_models(cache_dir=None):
    """Remove all cached models.

    Args:
        cache_dir (str or Path, optional): Override the default root cache directory
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()
    if Path(cache_dir).exists():
        import shutil

        shutil.rmtree(cache_dir)


# TODO: update tests to include model versioning
