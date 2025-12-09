## Setup
- Install NVIDIA cuda toolkit (make sure to add environment variables to .profile and .bashrc)
- Install pip requirements: `pip install -r requirements.txt`
- Clone Rust kernels project (https://github.com/NiekAukes/rust-kernels) and store it in the same folder as this project.
- Link Rust hybrid compiler (from source with: `rustup toolchain link rust-gpuhc ../rust-gpu-hybrid-compiler/build/host/stage1`)

### Troubleshooting
- When cargo is not recognized: `rustup default stable`
- When getting NVIDIA driver errors on linux: ensure that the driver is signed or secure boot is turned off.

