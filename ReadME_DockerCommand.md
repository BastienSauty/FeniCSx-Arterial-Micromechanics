Build the docker image using dockerfile

docker build -t spyder-python-env .

Run the docker image 

docker run -it --rm   -v $(pwd):/workspace -w /workspace   spyder-python-env

Alias to use 

fenicsx_0_9_0

Then run the spyder kernel : 

python -m spyder_kernels.console —-ip=0.0.0.0 -f=./kernel_info.json

In spyder connect to an existing kernel and point to kernel_info.json
Then to run a file. Change accordingly the file name

%runfile '/workspace/ChV_MultiscaleActiveStressRegulation/chv_3_1_ReachingBasalState.py' --args 'python -m ChV_MultiscaleActiveStressRegulation.chv_3_1_ReachingBasalState' --wdir /workspace
%runfile '/workspace/ChV_MultiscaleActiveStressRegulation/chv_3_2_Pipeline_SA.py' --args 'python -m ChV_MultiscaleActiveStressRegulation.chv_3_2_Pipeline_SA' --wdir /workspace

Run Jupyter lab
jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root
