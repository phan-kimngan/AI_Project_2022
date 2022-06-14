#  AI PROJECT 2022
Prof: Hyung–Jeong Yang
## Task: Combining Video Class Solution with Face Recognition Solution
Name: Kim Ngan Ngan

ID: 217161


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
│           ├── datetime1 <-- created by running system
│             ├── datetime1.mp4 <-- created by running system
│             ├── h264_datetime1.mp4 <-- created by running system
│       ├── webcam
│           ├── datetime2 #created by running system
│             ├── datetime2.mp4 <-- created by running system
│             ├── h264_datetime2.mp4 <-- created by running system
│           ├── datetime3 <-- created by running system
│             ├── datetime3.mp4 <-- created by running system
│             ├── h264_datetime3.mp4 <-- created by running system
├── ...
```
- Read the README at recog for more instructions

### Project Demo
+ Make **inputs** and **outputs** directories at **data**

+ Make **students** directory at **recog** and make sure as instructed

+ Put demo video into ./data/inputs such as test.mp4

+ Activate project_ai_env environment

+ Run command: *python initialize.py* to create **dataset-embeddings.npz** file at **recog**

+ Run command: *streamlit run main.py*

<img src="interface.png" alt="output" style="zoom:25%;" />

At Update Database page: Create ID -> Take picture -> **Add to database** for each picture -> **Finish**

At Video Recoding page: Choose option -> Load video of zoom at ./data/inputs -> **Start** and wait a few minutes -> **I want** for *Do you want to add the profile of new students?* question to add a new student if available

At Webcam page: choose option -> **Start** -> **Analysis** to show attendance report at this time -> **Stop** -> **I want** for *Do you want to add the profile of new students?* question to add a new student if available

At System Cloud page: choose option -> Choose Date -> choose directories to show video result. You can edit path_csv line in zoom.py where ./Downloads/ of your path to show the CSV reports at At System Cloud page when you downloaded the CSV reports

You can see video demonstration at https://ejnu-my.sharepoint.com/:v:/g/personal/217161_jnu_ac_kr/EQ9TuSjaGTVEnl89nf_5DfgBH1bLfU2K4Y5-xptfKSIJqg?e=0EGGVo
