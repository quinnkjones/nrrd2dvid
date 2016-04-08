
sudo apt-get update
sudo apt-get install python-pyrrd

wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh

CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root

conda create libdvid

conda install -n libdvid -c flyem libdvid-cpp

echo "$var" >> /home/$user/.bashrc && . ~/.bashrc

export PATH="/home/$USER/miniconda/env/dvidlib/lib:$PATH"

conda install -n libdvid numpy


source activate libdvid

conda install -c https://conda.anaconda.org/auto pynrrd

python setup_tifffile.py build_ext --inplace
