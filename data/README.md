The Data Preparation 

1. First clone the python-tools (https://github.com/lightfield-analysis/python-tools)
   repository, which supports read/write/process
   light field data (lightfield-analysis.net/benchmark/downloads/full_data.zip)
2. Run the function save_data() to save the needed vertical
   and horizontal crossing epipolar volumes 
   NOTE: run in python2 or change the scripts of python-tools
3. Run create_data_splited() function (in python3) to get 
   dataset in 'test' and 'train' splits
  
Already prepared data is available on IWR Compute Server
/export/home/tmuradya/3dcv
Or in github
git@github.com:ffeldmann/Light-Field-View-Synthesis.git
   
NOTE: this script is mostly hardcoded.
Be carefully by making any changes.