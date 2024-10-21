function deploy_doc(){
    if [ ! -z "$1" ]
    then
        git checkout $1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "Creating doc at commit" $COMMIT "and pushing to folder $2"
    # Hotfix
    if [ "$1" \< "dcbb21f" ]
        then
            pip install -U ..
            pip install rapidfuzz==2.15.1
            pip install tensorflow-addons>=0.17.1
        else
            pip install -U ..
        fi
    if [ ! -z "$2" ]
    then
        if [ "$2" == "latest" ]; then
            echo "Pushing main"
            sphinx-build source _build -a && mkdir build && mkdir build/$2 && cp -a _build/* build/$2/
        elif [ -d build/$2 ]; then
            echo "Directory" $2 "already exists"
        else
            echo "Pushing version" $2
            cp -r _static source/ && cp _conf.py source/conf.py
            sphinx-build source _build -a
            mkdir build/$2 && cp -a _build/* build/$2/ && git checkout source/ && git clean -f source/
        fi
    else
        echo "Pushing stable"
        cp -r _static source/ && cp _conf.py source/conf.py
        sphinx-build source build -a && git checkout source/ && git clean -f source/
    fi
}

# You can find the commit for each tag on https://github.com/mindee/doctr/tags
if [ -d build ]; then rm -Rf build; fi
cp -r source/_static .
cp source/conf.py _conf.py
git fetch --all --tags --unshallow
deploy_doc "" latest
deploy_doc "571af3d" v0.1.0
deploy_doc "6248b0b" v0.1.1
deploy_doc "650c4ad" v0.2.0
deploy_doc "1bbdb07" v0.2.1
deploy_doc "3f05134" v0.3.0
deploy_doc "369a787" v0.3.1
deploy_doc "51663dd" v0.4.0
deploy_doc "74ff9ff" v0.4.1
deploy_doc "b9d8feb" v0.5.0
deploy_doc "9d03085" v0.5.1
deploy_doc "dcbb21f" v0.6.0
deploy_doc "75bddfc" v0.7.0
deploy_doc "67d1087" v0.8.0
deploy_doc "62d94ff" v0.8.1
deploy_doc "894eafd" v0.9.0
deploy_doc "d5dbc73" # v0.10.0 Latest stable release
rm -rf _build _static _conf.py
