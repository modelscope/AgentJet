### Other Backbones

1. Trinity backbone: Click to read trinity backbone installation (Option 1)

```bash
# Create virtual environment
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/Trinity-RFT external/trinity

# Install dependencies
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements_trinity.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/trinity -i https://mirrors.aliyun.com/pypi/simple/ --no-deps

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```

2. VERL Backbone: Click to read VERL backbone installation (Option 2)


```bash
# Create virtual environment
uv venv --python=3.10.16
source .venv/bin/activate
git clone https://github.com/binary-husky/verl.git external/verl

# Install dependencies
uv pip install --upgrade pip setuptools packaging -i https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements_verl.txt -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --prerelease=allow
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/ --no-deps

# Install flash attention (must be installed last)
uv pip install --verbose flash-attn ring-flash-attn -i https://mirrors.aliyun.com/pypi/simple/ --no-deps --no-build-isolation
```


3. 🔄Switch Verl/Trinity backbone: cannot install both backbone at the same time, run the following command to toggle.

```bash
# verl -> trinity
cd external/verl && uv pip uninstall . && cd ../..

# trinity -> verl
uv pip install -e external/verl -i https://mirrors.aliyun.com/pypi/simple/ --no-deps
```
