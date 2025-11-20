Coursework 2: Comp 60012
Artificial Neural Networks
Ivan Severinov & Janhavi Revashetti
Imperial College London
November 21st 2024


Introduction:
Artificial Neural Networks Coursework 2 is a programming assignment for Introduction to Machine Learning where in Part 1  
we implment a our own Neural Network Mini Library. In Part 2 of the project, we used PyTorch to create a Neural Network that predicts 
California House Prices.
The python files are called part1_nn_lib.py and part2_house_value_regression.py

How to Run the Code:
To run this code, you must have at least python 3.12.3 on your workstation.
In addition you may use the following Python environment to run your code.
This virutal environment contains all the correct packages.
    Code:
        $ source /vol/lab/ml/intro2ml/bin/activate
        (intro2ml) $ python3 -c "import numpy as np; import torch; print(np); print(torch)"
        <module 'numpy' from '/vol/lab/ml/intro2ml/lib/python3.12/site-packages/numpy/__init__.py'>
        <module 'torch' from '/vol/lab/ml/intro2ml/lib/python3.12/site-packages/torch/__init__.py'> 
    To deactivate environment:
        (intro2ml) $ deactivate
    Tired of typing the full path to the virtual environment every time? Create a symbolic link (i.e. shortcut)
    from your home directory to point to the activate script. The following creates a symbolic link named
    intro2ml in your home directory. You only need to do this once.
        $ ln -s /vol/lab/ml/intro2ml/bin/activate ~/intro2ml

Code, Input, and Output:
The main files in this coursework are part1_nn_lib.py and part2_house_value_regression.py
The two datasets used in this project are iris.dat and housing.csv
To run this code, you just have to run the code. 
The output in part 1 is Validation Accuracy, Train Loss, and Validation Loss for the iris.dat dataset.
The output in part 2 is the Regression Error and the best model saved to the pickle file called part2_model.pickle