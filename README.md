#  AI PROJECT 2022
## Task: Combining Video Class Solution with Face Recognition Solution
Name: Kim Ngan Ngan

### Set up environment
+ Create a python new_env environment using conda or other tools.

+ Activate new_env environment
```bash
conda activate new_env
```
+ Instead packages in requirements.txt
```bash
pip install -r requirements.txt
```

### How to use?

You create a directory named **data** that contains the **inputs** and **outputs** sub-folders. 

+ The **input** sub-folder contains mp4 files for video recording. That means you must select the mp4 file when the system runs the recognition based on the video recording. 

+ The **outputs** sub-folder contains **webcam** and **video_recorder** sub-folders. Each sub-folder contains sub-folders according to datetime at the system runs. They contain mp4 and h264 result files for both video recording and webcam.

You can see  example structure for the final report as below:
```
data
├── data
│   ├── inputs
│   └── outputs
│       ├── video_recorder
│           ├── 2022-06-13_16-52-11 <-- created by running system
│             ├── 2022-06-13_16-52-11.mp4 <-- created by running system
│             ├── h264_2022-06-13_16-52-11.mp4 <-- created by running system
│       ├── webcam
│           ├── 2022-06-13_16-40-43 #created by running system
│             ├── 2022-06-13_16-40-43.mp4 <-- created by running system
│             ├── h264_2022-06-13_16-40-43.mp4 <-- created by running system
│           ├── 2022-06-13_16-48-33 <-- created by running system
│             ├── 2022-06-13_16-48-33.mp4 <-- created by running system
│             ├── h264_2022-06-13_16-48-33.mp4 <-- created by running system
├── ...
```
