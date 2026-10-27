# Build the multi-version documentation deployed on https://mindee.github.io/doctr
#
#   build/          -> latest stable release
#   build/latest/   -> main branch
#   build/vX.Y.Z/   -> previous releases
#
# Every version is rendered with the configuration, theme and static files of the main branch, so that they all share
# the same look and version switcher (`source/_static/switcher.json`, to update at each release).
#
# deploy_doc <commit> <folder> <version>
#   commit:  commit to build ("" for the current one)
#   folder:  output folder in build/ ("latest", "vX.Y.Z", or "" for the stable version at the root)
#   version: version shown as selected in the version switcher (must match an entry of switcher.json)
function deploy_doc(){
    local commit=$1 folder=$2 version=$3
    if [ -n "$commit" ]; then
        git checkout "$commit"
    fi
    echo "Creating doc at commit $(git rev-parse --short HEAD) (version: $version)"
    pip install -U ..

    # Older commits are rendered with the current configuration and assets
    local legacy=0
    if [ -n "$commit" ]; then
        legacy=$([ -f source/index.md ] && echo 0 || echo 1)
        rm -rf source/_static source/_templates
        cp -r _static _templates source/ && cp _conf.py source/conf.py
    fi

    local out=build
    if [ -n "$folder" ]; then
        out=build/$folder
    fi
    if [ -d "$out" ] && [ -n "$folder" ]; then
        echo "Directory $out already exists"
    else
        DOCTR_DOCS_VERSION=$version DOCTR_DOCS_LEGACY=$legacy sphinx-build source _build -a
        mkdir -p "$out" && cp -a _build/. "$out/" && rm -rf _build
    fi

    if [ -n "$commit" ]; then
        git checkout source/ && git clean -fd source/
    fi
}

# You can find the commit for each tag on https://github.com/mindee/doctr/tags
if [ -d build ]; then rm -Rf build; fi
cp -r source/_static source/_templates . && cp source/conf.py _conf.py
git fetch --all --tags --unshallow
deploy_doc "" latest latest
deploy_doc "1c9ce92" v0.11.0 v0.11.0
deploy_doc "97d4006" v0.12.0 v0.12.0
deploy_doc "7dabbe1" v1.0.0 v1.0.0
deploy_doc "6541a6e" v1.0.1 v1.0.1
deploy_doc "2d8e244" "" v1.1.0  # Latest stable release
rm -rf _build _static _templates _conf.py
