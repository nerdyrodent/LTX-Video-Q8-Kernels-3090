
### LTXVideo Q8 – `q8_kernels`

This package implements the operations required to perform inference with the LTXVideo FP8-quantized model.

---

## 📚 Table of Contents

- [Installation](#-installation)
  - [Windows](#-windows)
    - [Cloned ComfyUI](#-cloned-comfyui)
    - [Portable ComfyUI](#-portable-comfyui)
  - [Linux](#-linux)
  - [Troubleshooting](#-troubleshooting)
- [ComfyUI Integration](#-comfyui-integration)

---

### 🔧 Installation

> **Note:** CUDA Toolkit **12.8 or later** is required. The instructions below assume CUDA 12.8 is installed.

---

#### 🪟 Windows

1. Install the **[Microsoft Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022)**.  
   During installation, select **"Desktop development with C++"**.

2. Make sure the **CUDA Toolkit** is installed by following the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

---

##### 📦 Cloned ComfyUI
1. Run the following commands inside your ComfyUI python environment.

   ```bash
   python.exe -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   python.exe -m pip install -U packaging wheel ninja setuptools

   python.exe -m pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git
   ```

##### 📦 Portable ComfyUI

1. Open a **Command Prompt** in the portable ComfyUI installation folder.

2. Check the version of the embedded Python:
   ```bash
   .\python_embeded\python.exe --version
   ```

3. Install the matching Python version (e.g., 3.12.10):
   - Download from the [official Python website](https://www.python.org/downloads/windows/),  
     or use [pyenv for Windows](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#installation).

4. Locate your Python installation.  
   If installed from official Python website, it is typically located at:  
   `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python312`

5. Copy the `include` and `libs` directories from the full Python installation into the `python_embeded` folder in your ComfyUI directory.

6. Run the following commands in the Command Prompt:

   ```bash
   .\python_embeded\python.exe -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   .\python_embeded\python.exe -m pip install -U packaging wheel ninja setuptools

   .\python_embeded\python.exe -m pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git
   ```

---

#### 🐧 Linux

1. Make sure the **CUDA Toolkit** is installed using the instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).  
   > Your CUDA version must match the version used by your installed PyTorch (e.g., CUDA 12.8).

2. Run the following commands in your ComfyUI Python environment:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   pip install packaging wheel ninja setuptools
   pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git
   ```

---

#### 🧩 Troubleshooting

- If you encounter issues with the `typing-extensions` package, run:

  ```bash
  pip install typing-extensions
  ```

  Then repeat the installation steps.

---

### 🎥 ComfyUI Integration

To run inference with ComfyUI:
 
1. Install the **[ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)** custom nodes pack.
2. Use the **LTXVideo Q8 Patcher** node on the loaded model.
3. See the example ComfyUI flow [here](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/fa234963e5b24d11085dfdb8a51cf6d38d73472c/example_workflows/ltxv-13b-i2v-base-fp8.json).