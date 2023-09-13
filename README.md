# Installation


The code was tested on Windows 10, with [Anaconda](https://www.anaconda.com/download) Python 3.9 and [PyTorch]((http://pytorch.org/)) v2.0.1. 
[Optional but recommended] NVIDIA GPUs can be used for both training and testing.
After install Anaconda:

0. [Optional but recommended] create a new conda environment. 

    ~~~
    conda create --name pytorch_HW python=3.9
    ~~~
    And activate the environment.
    
    ~~~
    conda activate pytorch_HW
    ~~~

1. Install pytorch v2.0.1:

    ~~~
    conda install pytorch=2.0.1 torchvision -c pytorch
    ~~~
    
     
     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 
     
2. Install git:

    ~~~
	https://git-scm.com/download/
    ~~~

3. Clone this repo:

    ~~~
    HW4_ROOT=/path/to/clone/HW4
    git clone https://github.com/hrlblab/DS5660_HW4_TransferLearning_FineTuning $HW4_ROOT
    ~~~


4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
