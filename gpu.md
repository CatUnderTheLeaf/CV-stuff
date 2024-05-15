# GPU programming with NVIDIA's CUDA on Ubuntu

## Ubuntu 22.04

> Tutorial [source](https://github.com/CisMine/Setup-as-Cuda-programmers)
### 1. Installing the CUDA Toolkit - CUDA Driver
The CUDA Driver acts as a bridge between the operating system and the GPU, enabling communication with the hardware. Without it, the CUDA Toolkit, which provides software tools for GPU programming, cannot function properly. So, before installing the CUDA Toolkit, it's essential to install the CUDA Driver to ensure proper GPU utilization.

Verify that your NVIDIA driver is compatible with your computer. This can be confirmed by [Nvidia Driver](https://www.nvidia.com/drivers)

My suitable Nvidia Driver version is: 545.29.02

Let's open Software & Updates and install Nvidia Driver. Notice that install driver metapackage "Using NVIDIA driver metapackage from nvidia-driver-545 (proprietary)"

After installing, click restart

You have successfully installed the CUDA Driver. You can verify this by open terminal and run this command
```
$nvidia-smi
```
You'll notice a recommendation for the CUDA Toolkit version at the top right corner. Now, you need to find the appropriate CUDA Toolkit to download in this [link](https://developer.nvidia.com/cuda-toolkit-archive
).

I need to isntall Cuda Toolkit version: 12.3, so I followed the [Installation Instructions](https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
When done, run these commands to add your path

```
$gedit /home/$USER/.bashrc
```

Then scroll until the end and add these ( **replace cuda-12.3 into your version** )

```
export PATH="/usr/local/cuda-12.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH"
```
Save and exit editor. You have successfully installed the CUDA Toolkit. You can verify this by open terminal and run this command:

```
$nvcc -V
```
### 2. Installing CuDNN

Going this [link](https://developer.nvidia.com/rdp/cudnn-archive?fbclid=IwAR1Wl9U3uTFgihg49VkO-kyXihTqr0M1rtkCp9lwgM1G9SWE29WiNLRDg3Q_aem_AXAgXED-yDF8TPZ1KFasp4tA78932KiQ-plbcM1Vn2k2KGipmdYxfkQ4Y5FfOyz6Ygx9TNFpfLVTjDleBveNADdx) to download CuDNN v8.9.7

Download the **Local Installer for Linux x86_64 (Tar)**

Next, open the terminal in the directory where you downloaded the CuDNN files.

run these commands:

```
$ sudo apt-get install zlib1g
$ tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

You have successfully installed the CuDNN. You can verify this by open terminal and run this command

```
$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
Which means the version is 8.9.7

### 3. Install Tensorflow

Install, create and activate `venv` (more user-friendly option to open your project in VS Code and there make virtualenvs)
```
$ sudo apt install python3.10-venv

$ python3 -m venv ~/.env/tf_env

$ source ~/.env/tf_env/bin/activate
```
Install tensorflow with cuda support. It will install TensorFlow 2.16, as can be seen in release notes:
> TensorFlow pip packages are now built with CUDA 12.3 and cuDNN 8.9.7
```
$ pip install tensorflow[and-cuda]~=2.16.1
```
Check installation,`GPU` device should be listed
```
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## WSL2

Configuration for GPU usage: Ubuntu 22.04(WSL2), Python 3.10, CUDA Toolkit 12.2, cuDNN 8.9.7.29, tensorflow 2.15

1. Install [WSL2](https://docs.microsoft.com/windows/wsl/install) and Ubuntu 22.04
2. Open Ubuntu 22.04 in WSL2 and update Python and pip
```
$ sudo apt-get update && sudo apt-get upgrade
$ sudo apt-get install python3 #version 3.10
$ sudo apt-get install pip
$ pip install pip==21.3.1 # this version doesn't hate installation
```
3. Install fresh [NVIDIA® GPU drivers](https://www.nvidia.com/drivers)
4. Install [CUDA Toolkit 12.2 for WSL2](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
5. Download [cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x for Ubuntu 22.04](https://developer.nvidia.com/rdp/cudnn-archive)
6. Install cudnn
```
$ sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
$ sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get install libcudnn8=8.9.7.29-1+cuda12.2
$ sudo apt-get install libcudnn8-dev=8.9.7.29-1+cuda12.2
```
7. Install, create and activate `venv` (more user-friendly option to open your project in VS Code and there make virtualenvs)
```
$ sudo apt install python3.10-venv

$ python3 -m venv tensorflow2.15

$ source tensorflow2.15/bin/activate
```
8. Install tensorrt with `python3 -m pip install --upgrade tensorrt` to install tensorrt 8.6.1 (without it there was an installation error)
9. Finally install tensorflow with `pip install -U tensorflow[and-cuda]==2.15.0`
10. Edit `.bashrc`
```
# Export path variables !!! VERY IMPORTANT !!!
# Need to adapt to your python version:
$ export CUDNN_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn"
$ export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64"
# ...
$ export PATH="$PATH":"/usr/local/cuda/bin"
```
11. Check installation, should be 2.15.0 and `GPU` device listed
```
$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
12. Install Keras and check its version, should be >= 3.0.0
```
$ pip install --upgrade keras
$ python3 -c "import keras; print(keras.__version__)"
```
> There are still errors with this installation:
> 
> E "Unable to register cuDNN/cuFFT/cuBLAS factory: Attempting to register factory for plugin cuDNN/cuFFT/cuBLAS when one has already been registered"
> 
> W "TF-TRT Warning: Could not find TensorRT"
> 
> I "could not open file to read NUMA node" - WSL2 is not build with NUMA support, as I understand
> 
> Previously needed this with Tensorflow 2.13
> ```
> # make sim link to cuda library for GPU usage
> # because it needs sim link instead of direct access to library file
> $ cd /usr/lib/wsl/lib
> $ sudo rm libcuda.so libcuda.so.1
> $ sudo ln -s libcuda.so.1.1 libcuda.so.1
> $ sudo ln -s libcuda.so.1 libcuda.so
> $ sudo ldconfig
> $ cd
> ```
