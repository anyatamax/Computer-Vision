set -o xtrace

setup_root() {
    apt-get install -qq -y \
        ffmpeg             \
        git                \
        python3-pip        \
        python3-tk         \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     albumentations                     \
    #     albumentations_experimental        \
    #     imgaug                             \
    #     kornia                             \
    #     lightning[extra]                   \
    #     matplotlib                         \
    #     moviepy                            \
    #     opencv-python-headless             \
    #     pandas                             \
    #     pytest                             \
    #     scikit-image                       \
    #     scikit-learn                       \
    #     timm                               \
    #     torch                              \
    #     torchvision                        \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        aiohappyeyeballs==2.4.0            \
        aiohttp==3.10.5                    \
        aiosignal==1.3.1                   \
        albucore==0.0.17                   \
        albumentations==1.4.16             \
        albumentations-experimental==0.0.1 \
        annotated-types==0.7.0             \
        antlr4-python3-runtime==4.9.3      \
        attrs==24.2.0                      \
        bitsandbytes==0.44.1               \
        certifi==2024.8.30                 \
        charset-normalizer==3.3.2          \
        contourpy==1.3.0                   \
        cycler==0.12.1                     \
        decorator==4.4.2                   \
        docstring_parser==0.16             \
        eval_type_backport==0.2.0          \
        filelock==3.16.0                   \
        fonttools==4.53.1                  \
        frozenlist==1.4.1                  \
        fsspec==2024.9.0                   \
        huggingface-hub==0.24.6            \
        hydra-core==1.3.2                  \
        idna==3.8                          \
        imageio==2.35.1                    \
        imageio-ffmpeg==0.5.1              \
        imgaug==0.4.0                      \
        importlib_resources==6.4.5         \
        iniconfig==2.0.0                   \
        Jinja2==3.1.4                      \
        joblib==1.4.2                      \
        jsonargparse==4.33.2               \
        kiwisolver==1.4.7                  \
        kornia==0.7.3                      \
        kornia_rs==0.1.5                   \
        lazy_loader==0.4                   \
        lightning==2.4.0                   \
        lightning-utilities==0.11.7        \
        markdown-it-py==3.0.0              \
        MarkupSafe==2.1.5                  \
        matplotlib==3.9.2                  \
        mdurl==0.1.2                       \
        moviepy==1.0.3                     \
        mpmath==1.3.0                      \
        multidict==6.0.5                   \
        networkx==3.3                      \
        numpy==2.1.1                       \
        nvidia-cublas-cu12==12.1.3.1       \
        nvidia-cuda-cupti-cu12==12.1.105   \
        nvidia-cuda-nvrtc-cu12==12.1.105   \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.0.70        \
        nvidia-cufft-cu12==11.0.2.54       \
        nvidia-curand-cu12==10.3.2.106     \
        nvidia-cusolver-cu12==11.4.5.107   \
        nvidia-cusparse-cu12==12.1.0.106   \
        nvidia-nccl-cu12==2.20.5           \
        nvidia-nvjitlink-cu12==12.6.68     \
        nvidia-nvtx-cu12==12.1.105         \
        omegaconf==2.3.0                   \
        opencv-python==4.10.0.84           \
        opencv-python-headless==4.10.0.84  \
        packaging==24.1                    \
        pandas==2.2.2                      \
        pillow==10.4.0                     \
        pluggy==1.5.0                      \
        proglog==0.1.10                    \
        protobuf==5.28.2                   \
        pydantic==2.9.1                    \
        pydantic_core==2.23.3              \
        Pygments==2.18.0                   \
        pyparsing==3.1.4                   \
        pytest==8.3.2                      \
        python-dateutil==2.9.0.post0       \
        pytorch-lightning==2.4.0           \
        pytz==2024.1                       \
        PyYAML==6.0.2                      \
        requests==2.32.3                   \
        rich==13.9.2                       \
        safetensors==0.4.5                 \
        scikit-image==0.24.0               \
        scikit-learn==1.5.1                \
        scipy==1.14.1                      \
        setuptools==74.1.2                 \
        shapely==2.0.6                     \
        six==1.16.0                        \
        sympy==1.13.2                      \
        tensorboardX==2.6.2.2              \
        threadpoolctl==3.5.0               \
        tifffile==2024.8.30                \
        timm==1.0.9                        \
        torch==2.4.1                       \
        torchmetrics==1.4.1                \
        torchvision==0.19.1                \
        tqdm==4.66.5                       \
        triton==3.0.0                      \
        typeshed_client==2.7.0             \
        typing_extensions==4.12.2          \
        tzdata==2024.1                     \
        urllib3==2.2.2                     \
        yarl==1.11.0                       \
        ;
}

setup_checker() {
    python3 --version # Python 3.12.3
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"