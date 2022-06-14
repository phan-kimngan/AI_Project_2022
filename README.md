#  AI PROJECT 2022
## Task: Combining Video Class Solution with Face Recognition Solution
Name: Kim Ngan Ngan

### Set up environment
+ Create a python project_ai_env environment using conda or other tools.

+ Activate project_ai_env environment
```bash
conda activate project_ai_env
```
+ Instead packages in requirements.txt
```bash
pip install -r requirements.txt
```

### Structure of Project

- Create a directory named **data** that contains the **inputs** and **outputs** directories. 

+ The **input** directory contains mp4 files for video recording. That means you must select the mp4 file when the system runs the recognition based on the video recording. 

+ The **outputs** directory contains **webcam** and **video_recorder** sub-directories. Each sub-directory continuously contains sub-directories according to datetime at the system runs. They contain mp4 and h264 result files for both video recording and webcam.

See  example structure for *the final report* as below:
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
- Read the README at recog for more instructions

### Project Demo
+ Make inputs and outputs directories at data

+ Make students directory at recog

+ Put demo video into ./data/inputs such as test.mp4



+ Activate project_ai_env environment

+ Run streamlit run main.py

<img src="interface.png" alt="output" style="zoom:25%;" />

At Update Database: Create ID -> Take picture -> **Add to database** for each picture -> **Finish**

At Video Recoding: Choose option -> Load video of zoom at ./data/inputs -> **Start** and wait a few minutes -> **I want** for *Do you want to add the profile of new students?* question to add a new student if available

At Webcam: choose option -> **Start** -> **Analysis** to show attendance report at this time -> **Stop** -> **I want** for *Do you want to add the profile of new students?* question to add a new student if available

At System Cloud: choose option -> Choose Date -> choose directories to show video result. You can edit path_csv line in zoom.py where ./Downloads/ of your path to show the CSV reports at At System Cloud page when you downloaded the CSV reports
