<h2 align="center">Resolving Scale Ambiguity in Multi-view 3D Reconstruction using Dual-Pixel Sensors</h2>
<h4 align="center">
    <strong>Kohei Ashida</strong>
    ·
    <a href="https://sites.google.com/view/hiroaki-santo/"><strong>Hiroaki Santo</strong></a>
    ·
    <a href="http://cvl.ist.osaka-u.ac.jp/user/okura/"><strong>Fumio Okura</strong></a>
    ·
    <a href="http://www-infobiz.ist.osaka-u.ac.jp/en/member/matsushita/"><strong>Yasuyuki Matsushita</strong></a>
</h3>
This is the project page for 'Resolving Scale Ambiguity in Multi-view 3D Reconstruction using Dual-Pixel Sensors'.
<h4 align="center">ECCV 2024</h3>

## 

### Estimate blur size

Download the [datasets](https://huggingface.co/datasets/kohei-ashida/dp-sfm/tree/main), unzip the file and place it as follows.

<pre>
datasets/
├── DSLR
└── PHONE
</pre>

Install the MATLAB engine API in python and run `python3 estimate_blur.py`.


###  Estimate the scale
```
cd ./docker/ && ./docker_build.sh && ./docker_run.sh
docker exec -it dp-sfm-cupy bash
python3 run_eccv_DSLR.py --scene scene_id
python3 run_eccv_PHONE.py --scene scene_id
```
