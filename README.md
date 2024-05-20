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

#### Running the Data Acquisition Code:

```python
data = et.data.get_data('spatial-vector-lidar', verbose=True)
```

This code snippet imports the necessary maps and Shapely files for the "Map" class to function. Since it accesses the internet, users with firewalled computers will need to download the files manually:
#### Manual download of Earthpy's data:

[link to figshare download](https://help.figshare.com/article/usage-metrics)

Identify the path stored in the `et.io.HOME` constant. You can achieve this by entering a simple `print` statement in any Python console connected to your virtual environment where EarthPy is installed:

```python
print(et.io.HOME)
```
The full path to which the downloaded files should be extracted is:
```
et.io.HOME\earth-analytics\data\spatial-vector-lidar\
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
 - **x** - stops the program.