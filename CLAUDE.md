# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Testing
- `pytest` - Run all tests (tests are in the `tests/` directory)
- `pytest tests/test_birdnet.py` - Run specific model tests

### Code Formatting  
- `black .` - Format code with Black formatter (configured in pyproject.toml)
- `black --check .` - Check formatting without making changes

### Package Management
- `poetry install` - Install dependencies using Poetry
- `poetry add <package>` - Add new dependency
- `pip install -e .` - Install package in development mode

### BirdSet Subproject (in src/birdset/)
- `python train.py` - Train BirdSet models using Hydra configs
- `python eval.py` - Evaluate trained models
- `python eval.py experiment=local/DT_example` - Run specific experiment configs

## Project Architecture

### High-Level Structure
This is a bioacoustics model zoo that provides pre-trained models for bird and other animal sound classification. The repository contains two main components:

1. **Main Model Zoo** (`bioacoustics_model_zoo/`) - Collection of wrapped pre-trained models with unified API
2. **BirdSet Integration** (`src/birdset/`) - Research framework for training and evaluating bird classification models (author note: incorrect - focus is just on point 1 here)

### Model Zoo Architecture

#### Model Registration System
Models are registered using the `@register_bmz_model` decorator and exposed through `bmz.list_models()`. Each model class implements:
- `predict()` - Generate predictions on audio files
- `embed()` - Generate embeddings/features
- `train()` - Fine-tune on custom datasets (for some models)

#### Conditional Imports
The `__init__.py` uses conditional imports to handle optional dependencies gracefully:
- TensorFlow models (Perch, YAMNet, SeparationModel) require `tensorflow`
- BirdNET requires `ai-edge-litert` (TFLite)
- HawkEars requires `timm` and `torchaudio`
- BirdSet models require `transformers` and `torchaudio`

#### Supported Models
- **BirdNET**: TFLite bird classification model (6K species)
- **Perch**: Google's bird embedding model
- **HawkEars**: Ensemble of 5 CNNs for 314 North American bird species
- **BirdSet ConvNeXT/EfficientNetB1**: PyTorch models trained on Xeno Canto
- **YAMNet**: AudioSet embedding model
- **SeparationModel**: MixIT bird audio separation
- **RanaSierraeCNN**: Frog vocalization detector

### BirdSet Integration
The `src/birdset/` directory contains a full BirdSet research framework alongside the model zoo. The `bioacoustics_model_zoo/bmz_birdset/` directory contains the integration layer:
- `bmz_birdset_convnext.py` and `bmz_birdset_efficientnetB1.py` implement BirdSet pre-trained models
- `birdset_preprocessing.py` handles data preprocessing for BirdSet models
- Config files provide model loading specifications

### Model Integration Patterns

#### OpenSoundscape Integration
Most models inherit from or integrate with OpenSoundscape's CNN class, providing:
- Consistent preprocessing pipelines
- Standardized prediction interfaces
- Built-in audio handling and spectrogram generation

#### Transfer Learning Support
Models support fine-tuning through:
- `freeze_feature_extractor()` - Freeze backbone, train only classifier
- `change_classes()` - Adapt to new class sets by changing the classification head output size and the .classes attribute
- Custom training loops with validation

### Resource Management
- Manually-downloaded model checkpoints stored in `resources/` directory
- Pre-trained models downloaded from HuggingFace Hub or TensorFlow Hub
- Automatic fallback classes when dependencies are missing

## Development Notes

### Adding New Models
1. Create model class in `bioacoustics_model_zoo/`
2. Implement `predict()`, `embed()`, and optionally `train()` methods
3. Add `@register_bmz_model` decorator
4. Import in `__init__.py` with appropriate dependency checking
5. Add model to README.md model list

### Testing Strategy
The test suite is comprehensive and includes:
- **Unit tests**: Test individual functions and components (cache, utils, model registration)
- **Integration tests**: Test model loading and basic prediction functionality
- **Dependency tests**: Ensure graceful handling when optional dependencies are missing
- **Slow tests**: Full model tests marked with `@pytest.mark.slow` for CI/CD flexibility

Current test coverage includes:
- `test_cache.py`: Cache functionality tests
- `test_utils.py`: Utility function tests including download and model registration
- `test_birdnet.py`: BirdNET model tests with dependency handling
- `test_*.py` files for each model with similar structure

### Dependency Management
The codebase supports multiple optional dependencies to avoid forcing users to install heavy packages they don't need. Always check for missing imports before using model classes.

## Current Status
The model zoo is ready for PyPI release with the following features:
- ✅ Model caching system implemented (`cache.py`) with OS-specific default locations
- ✅ Unified model registration system with `@register_bmz_model` decorator
- ✅ Comprehensive conditional import system for optional dependencies
- ✅ Integration with OpenSoundscape for consistent APIs
- ✅ Test suite covering major functionality
- ✅ Poetry-based packaging configuration

## PyPI Release Preparation
For production release, ensure:
1. All model tests pass with appropriate dependency checks
2. Documentation is up-to-date in README.md
3. Version number is updated in `pyproject.toml`
4. CI/CD pipeline handles optional dependencies correctly
5. Long-running integration tests are properly marked with `@pytest.mark.slow` 