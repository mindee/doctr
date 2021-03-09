function deploy_doc(){
    echo "Creating doc at commit $1 and pushing to folder $2"
    git checkout $1
    pip install -U ..
    if [ ! -z "$2" ]
    then
        if [ "$2" == "main" ]; then
            echo "Pushing main"
            sphinx-build source _build -a && mkdir build && mkdir build/$2 && cp -a _build/* build/$2/
            # sphinx-build source _build -a && scp -r -oStrictHostKeyChecking=no _build/* $doc:$dir/$2/
            # cp -r _build/_static .
        elif ssh -oStrictHostKeyChecking=no $doc "[ -d build/$2 ]"; then
            echo "Directory" $2 "already exists"
            # scp -r -oStrictHostKeyChecking=no _static/* $doc:$dir/$2/_static/
            # cp -a _static/* build/$2/_static/
        else
            echo "Pushing version" $2
            sphinx-build source _build -a
            # cp -r _static _build/_static
            mkdir build/$2
            cp -a _build/* build/$2
            # scp -r -oStrictHostKeyChecking=no _build $doc:$dir/$2
        fi
    else
        echo "Pushing stable"
        sphinx-build source build -a
        # cp -r _static build/_static
        # scp -r -oStrictHostKeyChecking=no _build/* $doc:$dir
    fi
    # rm -r _static _build
}

# You can find the commit for each tag on https://github.com/mindee/doctr/tags
deploy_doc "main" main
deploy_doc "571af3dc" # v0.1.0 Latest stable release
