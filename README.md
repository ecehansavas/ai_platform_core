# ESTRA
Easy Streaming Data Analysis Tool (ESTRA) is designed with the aim of creating an easy-to-use data stream analysis platform that serves the purpose of a quick and efficient tool to explore and prototype machine learning solutions on various datasets. ESTRA is developed as a web-based, scalable, extensible, and open-source data analysis tool with a user-friendly and easy to use user interface

ESTRA consist of 4 main components:
```
    User Interface -> Javascript / ReactJs (ai_platform_ui)
    Web Server -> Python / Django (ai_platform_backend)
    Database -> PostgreSQL
    Background worker -> Python (ai_platform_core)
```

ESTRA  provides  a  flexible  deployment  structure  depending  on  the  use  case.   For personal use, ESTRA can be run on a regular personal computer.  For a large scale use, every component can be deployed into their own servers and they can even be deployed as a load-balanced multi-instance fashion.

Details shared in the following link https://open.metu.edu.tr/handle/11511/89668

## ai_platform_core
The  logic  of  the  application  is  defined  in  theWorkercomponent  implemented  in Python.
Initially, the worker checks the jobs in the database, and if there is any job in the queue that isready to be processed, then the worker takes the job, and the state of that process changes to "in progress". The worker starts running the selected algorithm with the dataset specified by the user. When the process is successfully completed, the worker then updates the database with the results and looks for another job.

# Installation (Ubuntu 18.04):

## Pre-requisites:
```
echo 'export PATH="${HOME}/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
sudo apt install python3-pip
python3 -m pip install --user pipenv
```

## Preparing the Code:
```
git clone https://github.com/ecehansavas/ai_platform_core
cd ai_platform_core
pipenv shell
pipenv install
```

## Preparing the R environment
Install R version 4, not 3.5
For Ubuntu 18.04: https://askubuntu.com/a/1287824
For Ubuntu 20.04: https://rtask.thinkr.fr/installation-of-r-4-0-on-ubuntu-20-04-lts-and-tips-for-spatial-packages/

```
sudo apt-get install r-base r-base-dev
sudo apt update -y
sudo apt-get install openjdk-8-jdk
sudo R CMD javareconf JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
sudo apt-get install libpcre2-8-0

R
    install.packages("proxy")
    install.packages("mclust")
    install.packages("rJava")
    install.packages("https://cran.r-project.org/src/contrib/Archive/stream/stream_1.3-2.tar.gz", repos=NULL, type="source")

    https://cran.r-project.org/src/contrib/Archive/streamMOA/streamMOA_1.2-2.tar.gz
    install.packages("streamMOA")
    install.packages("funtimes")
```

## Running the code
```
# Connection string format: postgres://[username]:[password]@[hostname]:[port]/[dbname]
DATABASE_URL="<connection string for the PostgreSQL database to be used>" python ai_core/Main.py
# Optional: get the connection string from heroku for the backend app via `heroku config`
```
