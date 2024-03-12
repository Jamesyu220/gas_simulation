# CSE6730_Gas_Simulation

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Getting Started](#getting-started)
<!-- * [Demos and Examples](#demos-and-examples) -->
* [License](#license)
* [Reference](#reference)
* [Contributors](#contributors)
<!-- * [Evaluation and Results](#evaluation-and-results) -->

## Introduction
In this project, we plan to establish a system incorporating various types of ideal gas particles, potentially featuring two distinct kinds, such as Nitrogen and Oxygen. The goal is to simulate a fluid representation that mirrors the smooth behavior characteristic of gases, as to observe the intricate movements of gas molecules and, importantly, to assess whether the ideal gas law remains applicable when considering some other factors. Additionally, we will introduce several other operations into the system extending beyond the ideal gas model. These operations will involve adjusting the total volume, inserting and extracting particles strategically, and introducing external factors such as a heat source. We intend to use an agent-based model to simulate particles individually, while using numerical methods and high-performance libraries to optimize the performance of the simulation.  

## Technologies
Project is created with:
* Python 3.9
* Jupyter Notebook
* Python libraries (see /requirements.txt)
* VSCode

## Getting Started
To run this project, 
1. Clone the repo:
    ```sh
    git clone https://github.gatech.edu/jyu678/CSE6730_Gas_Simulation.git
    ```

2. Set up the virtual environment[packages](#technologies)
    ```sh
    conda env create -f gas_env_win.yml
    conda activate gas_simul
    ```

3. Install python libraries
    ```sh
    pip install -r requirements.txt
    ```
4. Please go to the directory src and run:  
    ```sh
    python test_taichi.py
    ```   

<!-- ## Demos and Examples
To be done ...   -->

## License
Distributed under the Apache License. See LICENSE for more information.

## Reference 
[1] The ideal gas diffusion simulator developed by the University of Colorado. https://phet.colorado.edu/en/simulations/gas-properties  

[2] Y. Zeng and J. Fang, “Numerical simulation and experimental study on gas mixing in a gas chamber for sensor evaluation,” Measurement: Sensors 18, 100338 (2021). https://doi.org/10.1016/j.measen.2021.100338  

[3] Scott Van Bramer. The Kinetic-Molecular Theory, Effusion, and Diffusion. https://chem.libretexts.org/Courses/Widener_University/Widener_University%3A_Chem_135/05%3A_Gases/5.04%3A_The_Kinetic-Molecular_Theory_Effusion_and_Diffusion  

[4] Simulation of an Ideal Gas to Verify Maxwell-Boltzmann distribution. https://github.com/rafael-fuente/Ideal-Gas-Simulation-To-Verify-Maxwell-Boltzmann-distribution.git  

[5] Ideal gas simulation in a 3D system. https://github.com/labay11/ideal-gas-simulation.git  

[6] Skiverse: A SKI universe. https://github.com/mountain/skiverse.git  

[7] Python Real Gas FROzen SHock (RGFROSH) https://github.com/VasuLab/RGFROSH.git  

[8] Thermodynamic Cycles. https://github.com/geokosto/Thermodynamic-Cycles.git  

[9] Liu, M.B., Liu, G.R. Smoothed Particle Hydrodynamics (SPH): an Overview and Recent Developments. Arch Computat Methods Eng 17, 25–76 (2010). https://doi.org/10.1007/s11831-010-9040-7  

[10] Pereira, P., Cruz, F., Carvalho, D. Pombo, I. A Smooth Introduction to Smoothed Particle Hydrodynamics (SPH). https://inductiva.ai/blog/article/sph-2-a-smooth-introduction  

[11] Ren, B., Yan, X., Yang, T. et al. Fast SPH simulation for gaseous fluids. Vis Comput 32, 523–534 (2016). https://doi.org/10.1007/s00371-015-1086-y  

## Contributors
* [Kuo, Chun-Fu](https://github.gatech.edu/ckuo67)
* [Lin, Ping-Chuan](https://github.gatech.edu/plin302)
* [Liu, Ziming](https://github.gatech.edu/zliu874)
* [Nielson, Felicity](https://github.gatech.edu/Fnielson3)
* [Yu, Andy](https://github.gatech.edu/ayu303)
* [Yu, James](https://github.gatech.edu/jyu678)
---