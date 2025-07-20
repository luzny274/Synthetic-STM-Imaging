# Synthetic-STM-Imaging

## Overview

This repository provides tools for generating synthetic STM images of WSe₂ samples for ML tasks, and an example of training a U-Net model for defect localization.

---

## Environment Setup

This project was developed on a Windows 11 machine using an Anaconda environment. You can create the required environment using one of the provided `.yml` files:

### For NVIDIA GPUs (CUDA support)
```bash
conda env create -f environment__CUDA.yml
```

### For CPU-only systems
```bash
conda env create -f environment__CPU_only.yml
```
---

## WSe₂ Sample Generation

To generate synthetic WSe₂ samples:

1. Navigate to the `Sample_Generator` directory.
2. Run the batch script:
   ```bash
   generate_dataset.bat
   ```

> **Note:** Ensure your Conda environment is activated in the command line before running the script.

### Customizing the Sample Count

The number of samples to generate is controlled by the `sample_count` variable inside `generate_dataset.bat`.

### Modifying Simulation Parameters

Simulation parameters (e.g., defect types, scan resolution, drift settings) are set in:

```
gui_params/main_params.json
```

To modify these parameters through a graphical interface, run:

```bash
streamlit run gui_sample_gen.py
```

Make sure your Conda environment is activated before launching the GUI.

### Examples
1000 generated samples can be found in the `Sample_Generator/WSe2_dataset` folder. 100000 oof such samples were used for training the U-Net model

---

