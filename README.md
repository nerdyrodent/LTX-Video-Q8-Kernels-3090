### LTXVideo Q8 – `q8_kernels`

This package implements the operations required to perform inference with the LTXVideo FP8-quantized model.

---

### 🔧 Installation

#### On Windows

First, make sure to install the **[Microsoft Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022)**.  
During installation, ensure you select **"Desktop development with C++"**.

#### On all platforms

Run the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install packaging wheel ninja setuptools   
python setup.py install
```

### 🎥 ComfyUI Integration

To run inference with ComfyUI:

1. Install the **ComfyUI-LTXVideo** custom nodes pack.
2. Use the **LTXVideo Q8 Patcher** node on the loaded model.
3. An example ComfyUI flow is available [here](https://github.com/Lightricks/ComfyUI-LTXVideo).