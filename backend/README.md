# MyWardrobe Backend Development

This folder contains development and utility scripts for the MyWardrobe system.

## Structure

### `development/` - Core Development Scripts
- **`main.py`** - CLI interface for building embeddings, searching, and fine-tuning
- **`run_search.py`** - Wrapper script with environment setup and NumPy version checking
- **`clip_working_example.py`** - Working CLIP implementation example
- **`validate_setup.py`** - System validation and health checks
- **`setup.sh`** - Environment setup script

### `scripts/` - Utility Scripts
- Place additional utility scripts here

## Usage

### Building New Embeddings (when you get more data)
```bash
cd backend/development
python main.py prep --image_dir /path/to/new/images --mask_dir /path/to/masks
```

### Running Searches
```bash
cd backend/development
python main.py query --query_image /path/to/image --query_text "casual summer top" --top_k 10
```

### Fine-tuning
```bash
cd backend/development
python main.py finetune
```

### Validation
```bash
cd backend/development
python validate_setup.py
```

## Important Notes

- **Keep these files!** They're essential for future development
- When you get more clothing data, use `main.py prep` to create new embeddings
- The `finetune.py` in `src/mywardrobe/` is for model fine-tuning
- These scripts work with your existing `embeddings.npy` and `index_paths.txt`

## Future Development

- Add new data processing scripts here
- Experiment with different embedding strategies
- Test new models and fine-tuning approaches
- Keep the production API (`api/`) separate from development tools
