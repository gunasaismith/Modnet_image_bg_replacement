# Modnet Image Background Replacement

A minimal, local demo and helpers for portrait foreground extraction and
background replacement using MODNet. This repo provides a simple web demo
and example scripts to run inference on images.

## Features
- Lightweight Flask/demo app for uploading an image and applying background replacement.
- Integration with MODNet model code and example pipelines.
- Clear instructions to keep large model binaries and virtual environments
  out of the Git history (use Git LFS if you prefer tracking large files).

## Requirements
- Python 3.8+
- A CUDA-enabled GPU is optional but speeds up inference; CPU-only works too.

## Installation
1. Create and activate a virtual environment:

   ```powershell
   python -m venv myenv
   .\\myenv\\Scripts\\Activate.ps1
   ```

2. Upgrade pip and install requirements:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Obtaining the pretrained model
1. Download the MODNet checkpoint (photographic portrait matting) from the
   official release or the location you obtained it from.
2. Create a `pretrained/` directory in the project root and place the file
   there. The demo expects the filename:

   `pretrained/modnet_photographic_portrait_matting.ckpt`

## Running the demo
1. From the repository folder (the pushed, clean copy is `clean_push`):

   ```powershell
   cd "D:\Downloads\modnet\BG Replacement\clean_push"
   .\\myenv\\Scripts\\Activate.ps1   # if you used the suggested venv
   python app.py
   ```

2. Open a browser at http://127.0.0.1:5000 (or the printed host/port) and
   follow the UI to upload an image and try background replacement.

## Command-line usage
- `python modnet_pipeline.py --input <image> --output <out.png> --bg <bg.jpg>`
  (See `modnet_pipeline.py` for available flags and examples.)

## Repository layout
- `app.py` — demo web server
- `modnet_pipeline.py` — CLI pipeline for single-image processing
- `MODNet/` — included MODNet source code (model definitions, demos)
- `static/` — sample images and output examples
- `templates/` — web UI templates
- `requirements.txt` — Python deps (do not include virtualenv)

## Large files and Git LFS
- This repository purposefully excludes `myenv/`, `pretrained/`, and
  `modnet_workspace/`. If you need to version model checkpoints, enable
  Git LFS and track the model files:

  ```bash
  git lfs install
  git lfs track "pretrained/*.ckpt"
  git add .gitattributes
  ```

## Security & privacy
- Do not upload private or copyrighted images to public repos or demo
  deployments. Keep model checkpoints and data local where appropriate.

## Contributing
- Bug reports and pull requests are welcome. For substantial changes,
  open an issue first to discuss the design.

## License
- Add your preferred license file (e.g., `LICENSE`) to the repo. Tell me
  which license you want and I can add it.

## Acknowledgements
- MODNet project authors — this repo integrates MODNet sample code and
  model checkpoints (not distributed here).

## Contact
- If you'd like further README additions (examples, CI, badges, or a
  LICENSE), tell me which and I'll add them.
