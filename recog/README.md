You can download the weight of Facenet at the link and put into here: https://drive.google.com/file/d/1hi7UhUN5klGcF_6SA_zhAb3O_rUI3COk/view?usp=sharing

In this folder, you create a new directory named **students** where can store sub-directories according to the ID of students. Each sub-folder contains pictures of each student. 

In students directories:
+ Create a sub-directories named "unknown" for detecting new students
+ Initialize requires at least 2 sub-directories with student photos available, other directories can be created by the system

You can see below structure:
```
recog
├── students
│   ├── unknown <- make available
│   ├── student_ID1 <- make available
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
│   ├── student_ID2 <- make available
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
│   └── ... <- add by running system
├── ...
```
