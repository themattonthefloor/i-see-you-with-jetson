# Jetson Xavier Setup Instructions

### 1. Check available disk space and create work folder

- `df -h` Ensure that there is at least ***16GB*** free disk capacity

- Navigate to the i-see-you-with-jetson directory and create a directory for installation files
- Note: if the i-see-you-with-jetson folder does not exist, create it.

- `cd ~/Desktop/i-see-you-with-jetson/`

- `mkdir installation-files`

### 2. Download and install Miniforge
- `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -P installation-files`

- `bash installation-files/Miniforge3-Linux-aarch64.sh`

- Enter "yes" for conda init, then close and start a new terminal

### 4. Setup inference environment
#### 4.1. Run the following commands in the following order to create the basic inference environment:
- `conda create --name inference python=3.6.15`

- `conda activate inference`

- `pip install opencv-python==4.5.5.64`

- `pip install pynvml==11.4.1`

- `conda install matplotlib=3.3.4`

- `pip install termcolor==1.1.0`

- `pip install scikit_learn==0.24.2`

- `pip install tqdm==4.64.0`

- `pip install cvlib==0.2.6`

#### 4.2. Download and install pytorch & torchvision using the following wheel installation commands:

- `wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O installation-files/torch-1.8.0-cp36-cp36m-linux_aarch64.whl`

- `sudo apt-get install python3-pip libopenblas-base libopenmpi-dev `

- `pip3 install numpy installation-files/torch-1.8.0-cp36-cp36m-linux_aarch64.whl`

- `pip install torchvision==0.9.1`

#### 4.3. Download and install pycocotools:

- `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

- `export LD_PRELOAD=~/miniforge3/envs/inference/lib/python3.6/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0`

#### 4.4. Install pycuda for cuda/cudnn:

The .bashrc files needs to be edited to proceed with the installation.
- Use the command `vi ~/.bashrc` to open the file editor.
- Scroll to the end of the file and add the following to the script:
~~~
# <<< AISG modifications <<<
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# <<< AISG modifications <<<
~~~
- run `source ~/.bashrc` to restart the terminal
- run `nvcc --version`
- The console should print the following:
> nvcc: NVIDIA (R) Cuda compiler driver
>
> Copyright (c) 2005-2021 NVIDIA Corporation
>
> Built on Sun_Feb_28_22:34:44_PST_2021
>
> Cuda compilation tools, release 10.2, V10.2.300
>
> Build cuda_10.2_r440.TC440_70.29663091_0

To activate the inference environment and install pycuda:

- Run `conda activate inference`

- Run `pip install pycuda==2020.1`

#### 4.5. Install torch2trt
The .bashrc files needs to be further edited to proceed with the installation.
- Run `vi ~/.bashrc` to open the file editor.
- Further amend the end of the script to match the following:
~~~
# <<< AISG modifications <<<
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
export LD_PRELOAD=~/miniforge3/envs/inference/lib/python3.6/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
# <<< AISG modifications <<<
~~~
- Run `source ~/.bashrc` to restart the terminal
- Run `conda activate inference` to reactivate the inference environment

Run the following to download and install torch2trt

- `cd ~/Desktop/aisg/installation-files/`

- `git clone https://github.com/NVIDIA-AI-IOT/torch2trt`

- `cd torch2trt`

- `git checkout 2732b35ac4dbe3d6e93cd74910af4e4729f1d93b`

- `python setup.py install`

### 5. Camera check
- Check if the camera is functioning properly by running this `simply_cam.py` script.
    ```
    $ cd ~/Desktop/test
    $ python3 simply_cam.py
    ```

- If camera is unable to open, may require reboot of edge device. Repeat this section until camera is working before proceeding to next section.
- If camera is able to show live view of station, hit `q` to close camera and quit. Proceed to next section to test live inference.

### 6. Inference
- Copy the entire project folder `whyre-tech-dev` including all source codes into the `~/Desktop/aisg/` directory.
- Do not amend the folder/file structure as this may result in inference to not work.
- Run `cd ~/Desktop/aisg/whyre-tech-dev/` to navigate to the project folder
- Run `conda activate inference` to activate the inference environment
- Follow 5.1 or 5.2 to perform inference.
- Note that it is normal for the model to print warning messages during tensor RT preparation.

#### 6.1. Sample command for live inference using camera capture
- For ***bumpy*** station, run

    ```
    # debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/bumpy/ye/yolact_edge_mobilenetv2_wt_100_46258.pth --cnn_model_path=weights/champion_0.18/bumpy/cnn/mobilenetv2_bumpy_day_2022-05-12_18-15-457004_epochs039_SD.pth --station=bumpy --output_pred_img --debug --show_window

    # non-debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/bumpy/ye/yolact_edge_mobilenetv2_wt_100_46258.pth --cnn_model_path=weights/champion_0.18/bumpy/cnn/mobilenetv2_bumpy_day_2022-05-12_18-15-457004_epochs039_SD.pth --station=bumpy --output_pred_img --debug --show_window
    ```

- For ***plank day*** station, run

    ```
    # debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/plank_day/ye/yolact_edge_mobilenetv2_wt_100_41208.pth --cnn_model_path=weights/champion_0.18/plank_day/cnn/mobilenetv2_plank_day_2022-05-24_12-16-955437_epochs034_SD.pth --station=plank --output_pred_img --debug --show_window

    # non-debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/plank_day/ye/yolact_edge_mobilenetv2_wt_100_41208.pth --cnn_model_path=weights/champion_0.18/plank_day/cnn/mobilenetv2_plank_day_2022-05-24_12-16-955437_epochs034_SD.pth --station=plank
    ```

- For ***plank night*** station, run

    ```
    # debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/plank_night/ye/yolact_edge_mobilenetv2_wt_100_41309.pth --cnn_model_path=weights/champion_0.18/plank_night/cnn/mobilenetv2_plank_night_2022-05-25_00-51-313679_epochs012_SD.pth --station=plank --output_pred_img --debug --show_window

    # non-debug mode
    python -m src.jetson_live_inference --weights=/weights/champion_0.18/plank_night/ye/yolact_edge_mobilenetv2_wt_100_41309.pth --cnn_model_path=weights/champion_0.18/plank_night/cnn/mobilenetv2_plank_night_2022-05-25_00-51-313679_epochs012_SD.pth --station=plank
    ```


#### 6.2. Live inference script
- When using the inference class in your own script, it can be called as such:
~~~ 
from src.inference import WhyreTechInference

...

model = WhyreTechInference(
                weights=args.weights,
                station=args.station,
                cnn_model_path=args.cnn_model_path
                )
...

prediction, timestamp, _, _, _, _, _, _ = model.predict(frame)

- For further reference, refer to the `jetson_live_inference.py` script in `src` directory for the example.