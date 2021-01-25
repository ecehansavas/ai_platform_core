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
