# maps_extraction
This is a software to extract the surface and color maps of tree barks from their 3D point cloud.

## Dependencies List
OpenCV- Tested on version 3.4.4 and
PCL -Tested on version 1.3

## Installation

Clone the repository 
```bash
git clone https://github.com/anon454/maps_extraction.git
```
Compilation
```bash
cd maps_extraction
mkdir build && cd build
cmake ..
make
cd ..
chmod +x process_pc.sh
```
## Dataset Preparation
Place all the point clouds of the trees in a single folder in .ply format. The code assumes that the y co-ordinates of the point cluds represent the height of the trees. By default the software starts extracting the maps from y=0 upto y=1.1 (measured in metres). The values can be modified in src/pc_tree.cpp, lines 405 and 406 (d_min and d_max).

## Extracting the maps
Run the script process.pc.sh to extract the surface and color maps from all the point clouds. The script takes the path to the folder containing the point clouds as an argument. 
```bash
./process_pc.sh [PATH_TO_POINTCLOUD_DIRECTORY]
```
The extracted maps are stored in 'dataset/surface' and 'dataset/color'.
