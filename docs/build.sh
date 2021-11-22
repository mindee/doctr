function deploy_doc(){
    if [ ! -z "$1" ]
    then
        git checkout $1
    fi
    COMMIT=$(git rev-parse --short HEAD)
    echo "Creating doc at commit" $COMMIT "and pushing to folder $2"
    pip install -U ..
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
deploy_doc "571af3dc" v0.1.0
deploy_doc "6248b0bd" v0.1.1
deploy_doc "650c4ad4" v0.2.0
deploy_doc "1bbdb072" v0.2.1
deploy_doc "3f051346" v0.3.0
deploy_doc "369a787d" v0.3.1
deploy_doc "51663ddf" v0.4.0
deploy_doc "74ff9ffb" # v0.4.1 Latest stable release
rm -rf _build _static _conf.py
