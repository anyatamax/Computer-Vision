set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip        \
        python3-tk         \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     matplotlib                         \
    #     numpy                              \
    #     pillow                             \
    #     pytest                             \
    #     scikit-image                       \
    #     scikit-learn                       \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        contourpy==1.3.0                   \
        cycler==0.12.1                     \
        fonttools==4.53.1                  \
        imageio==2.35.1                    \
        iniconfig==2.0.0                   \
        joblib==1.4.2                      \
        kiwisolver==1.4.7                  \
        lazy_loader==0.4                   \
        matplotlib==3.9.2                  \
        networkx==3.3                      \
        numpy==2.1.1                       \
        packaging==24.1                    \
        pillow==10.4.0                     \
        pluggy==1.5.0                      \
        pyparsing==3.1.4                   \
        pytest==8.3.2                      \
        python-dateutil==2.9.0.post0       \
        scikit-image==0.24.0               \
        scikit-learn==1.5.1                \
        scipy==1.14.1                      \
        six==1.16.0                        \
        threadpoolctl==3.5.0               \
        tifffile==2024.8.30                \
        ;
}

setup_checker() {
    python3 --version # Python 3.12.3
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"