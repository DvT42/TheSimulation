# TheSimulation

## Installation and Setup

This guide outlines the installation process and data preparation steps for using EarthPy's functionalities for spatial vector LiDAR analysis.

### Recommended Installation Method

We highly recommend installing EarthPy through GitHub instead of just copying the code files. The Git repository stores pre-trained models (visible in the file tree) that you can directly load into your code, eliminating the need to train the system from scratch. Of course, the traditional installation method is still an option.

[GitHub repository link](https://github.com/DvT42/TheSimulation)

### Other Installations

#### 1. Install Required Libraries:

Regardless of the installation method, ensure you install all the libraries listed in the `requirements.txt` file. Refer to your preferred package manager's documentation for installation instructions.

#### 2. Data Preparation

#### Manual download of Earthpy's data:

To abstain from needlessly installing earthpy as a dependency, it is required to download the shapely files that are at their database manually. this can be done easily using the following link:

[link to figshare download](https://ndownloader.figshare.com/files/12459464)

The full path to which the downloaded files should be extracted is:
```
DRIVER\Users\USER\earth-analytics\data\spatial-vector-lidar\
```
It should be noted that the files are relatively massive, so they take time to download. Downloading them manually will take less time than downloading through earthpy's built-in interface according to my experience.

## System Operation
Currently, there is no built-in command that claims a specific save from the data, but the load operations draw from the files that are placed in the folder itself and not in its subfolders. What this means is that it is necessary to manually copy the files that are inside the desired folder inside data into the data folder.

Once the desired files have been placed in the data folder, the simulation can be started. Any of the allowed commands can be entered into the system, which are shown in the following list.

After the desired files have been placed in the data folder, the simulation can be started. Any of the allowed commands can be entered into the system, which are shown in the following list.
### List of possible commands:
 - **load** - loads the pairs of the minds of the successful people (best), their number according to the SELECTED COUPLES rule.
 - **load alive** - loads all the saved alive people separately (men and women).
 - **load children** - loads all the people who give birth to the saved children.
 - **load both** - loads the same amount of people living and giving birth to children.
 - **save** - saves the most successful (best) into saved brains.pkl.
 - **save alive** - saves all saved alive people separately (men and women).
 - **save children** - saves all the people who give birth to the children into saved brains.pkl.
 - **save both** - Saves all people who give birth to children and lives, if they are more than those that were saved before.
 - **best** - displays the identifying numbers of the most genetically successful couples in the current simulation, numbered according to the SELECTED COUPLES constant.
 - **i + id** - displays general information about the person in the current simulation with the entered ID number.
 - **il + id** - displays the movement history of the person in the current simulation with the entered ID number.
 - **ia + id** - shows the relationship of the person in the current simulation with the entered ID number.
 - **s + num** - advances the simulation by the given number of months.
 - **y + num** - advances the simulation similarly to s + num, but in as many years as the given number.
 - **display** - shows the current simulation in terms of regions and living people.
 - **visualize** - opens up a matplotlib representation of the current simulation.
 - **x** - stops the program.
