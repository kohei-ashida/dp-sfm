import glob
import numpy as np

from Viewclass import DataLoader2
import estimate as est
import os
import re
from concurrent.futures import ProcessPoolExecutor

def multiview(datas):
    Xs = []
    Ys = []
    fs = []
    fnums = []
    for data in datas:
        Xs.append(data.x)
        Ys.append(data.y)
        fs.append(data.focal_length)
        fnums.append(data.fnum)

    # s, gs = est.calc_s_and_gs(Ys, Xs, datas[0].focal_length, datas[0].fnum, way=way)
    s, gs = est.optmize_s_gs(Ys, Xs, fs, fnums)
    return s, gs

def proccess_view(blurest_type,
                f,
                fnum,
                coc_path,
                dep_path,
                imgb_path,
                af_path,
                FOR_SUPP):
    view = DataLoader2(
        blurest_type,
        focal_length=f,
        fnum=fnum,
        estblur_path=coc_path,
        depth_path=dep_path,
        imgb_path=imgb_path,
        afpoint_path=af_path,
        sensor_p=1.4e-3 * 2
    )
    # load
    print("load")
    view.idnum = int(re.search(rf"{f}F{fnum}_(\d+)", view.cocPath).group(1))
    # if view.idnum != 22:
    #     return None
    view.load_data()
    # filtering
    if not FOR_SUPP:
        confidence = view.out_sobel *\
                        np.exp(-view.out_fval * 10**6)\
                        / (view.depth_haba**2 + 1e-10)
        if view.error_map is not None:
            confidence[view.error_map] = np.nan
        confidence[(np.isnan(view.depth)) | (view.depth==0)] = np.nan

        # confidence[np.isnan(confidence)] = 0
        sorted_conf = sorted(set(confidence[~np.isnan(confidence)]), reverse=True)
        T_c = sorted_conf[ round(len(sorted_conf) * 0.1) - 1 ]
    else:
        assert view.GTfocusdis is not None, "out! (GTg)"
        GTblur = est.fn(
            view.GTfocusdis, view.depth, view.focal_length, view.fnum, s=1
        )
        abs_error = np.abs(view.blur_radius_opt*2 - GTblur)

        confidence = 1 / (abs_error + 1e-10) \
                        / (view.depth_haba**2 + 1e-10)
        if view.error_map is not None:
            confidence[view.error_map] = np.nan
        confidence[(np.isnan(view.depth)) | (view.depth==0)] = np.nan

        # confidence[np.isnan(confidence)] = 0
        sorted_conf = sorted(set(confidence[~np.isnan(confidence)]), reverse=True)
        T_c = sorted_conf[ round(len(sorted_conf) * 0.01) - 1 ]



    confidence[np.isnan(confidence)] = 0
    # confidence[np.isnan(confidence)] = confidence[~np.isnan(confidence)].min()


    filtering_error = confidence < T_c

    view.filtering(filtering_error)

    s, _ = view.calc_s_and_gs()

    print(f"est s: {s}")

    return view


def main(coc_paths, depth_paths, imgb_paths, af_paths, f, fnum, FOR_SUPP=False):
    views = []

    # sort paths
    coc_paths = sorted(coc_paths)
    depth_paths = sorted(depth_paths)
    imgb_paths = sorted(imgb_paths)
    ids_coc_paths = list(map(lambda x: int(re.search(rf"{f}F{fnum}_(\d+)", x).group(1)), coc_paths))
    ids_depth_paths = list(map(lambda x: int(re.search(rf"{f}F{fnum}_(\d+)", x).group(1)), depth_paths))
    ids_imgb_paths = list(map(lambda x: int(re.search(rf"{f}F{fnum}_(\d+)", x).group(1)), imgb_paths))

    for id_coc_path, id_depth_path, id_imgb_path in zip(ids_coc_paths, ids_depth_paths, ids_imgb_paths):
        assert id_coc_path == id_depth_path == id_imgb_path ,"out!"
    print("ok")




    # for coc_path, dep_path, imgb_path in zip(coc_paths, depth_paths, imgb_paths):
    #     print(coc_path, dep_path, imgb_path)

    # exit()



    if True:
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = []
            for coc_path, dep_path, imgb_path, af_path\
                in zip(coc_paths, depth_paths, imgb_paths, af_paths):
                futures.append(executor.submit(proccess_view, 
                                               "p_51_k_21_s_27_r_1",
                                               f,
                                               fnum,
                                               coc_path,
                                               dep_path,
                                               imgb_path,
                                               af_path,
                                               FOR_SUPP))

            for future in futures:
                view = future.result()
                if view is not None:
                    views.append(view)
            executor.shutdown()

        views_selected = list(filter(lambda x: x.s is not None, views))
    else:
        for coc_path, dep_path, imgb_path, af_path\
            in zip(coc_paths, depth_paths, imgb_paths, af_paths):
            # print(coc_path, dep_path, ref_path)
            # view = DataLoader2(
            #    "p_51_k_21_s_27_r_1",
            ##    "p_111_k_41_s_47_r_0.5",
            #     focal_length=f,
            #     fnum=fnum,
            #     estblur_path=coc_path,
            #     depth_path=dep_path,
            #     imgb_path=imgb_path,
            #     afpoint_path=None,
            # )

            views.append(proccess_view("p_111_k_41_s_47_r_0.5",
                f,
                fnum,
                coc_path,
                dep_path,
                imgb_path,
                af_path,
                FOR_SUPP))

        views_selected = list(filter(lambda x: x.s is not None, views))





    for view in views_selected:
        print(view.idnum, view.s, view.g)


    views_selected = sorted(views_selected, key=lambda x: x.s)

    views_selected = views_selected[
        len(views_selected) // 2 - 3: len(views_selected) // 2 + 4
    ]

    print("=========================================")

    for view in views_selected:
        print(view.idnum, view.s, view.g)
    s, gs = multiview(views_selected)
    print(f"s: {s}, gs: {gs}")


    os.makedirs(f"../resutls/{CAMERA_TYPE}_{scene_name}/", exist_ok=True)

    # save result to text
    with open(f"../resutls/{CAMERA_TYPE}_{scene_name}/{f}F{fnum}_s.txt", "w", encoding="utf-8") as f:
        for view in views_selected:
            f.write(f"{view.idnum} {view.s} {view.g}\n")
        f.write(f"multiview| s: {s}  gs: {gs}\n")
        soutia = (s-1)*100
        soutai = round(soutia, 3)
        f.write(f"{soutai}")



CAMERA_TYPE = "PHONE"

# argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=int, default=1)

scene_id = parser.parse_args().scene

#workspace2021ashidadp_matomeQualitativeAFpointDSLRScene2/35.0F1.4_001.npy
for f in [4.38][:]:
    for fnum in [1.73][:]:
        for scene_name in ["Scene1", "Scene2", "Scene3", "Scene4", "Scene5", "Scene6"][(scene_id-1)*2:scene_id*2]:
            # coc_paths = glob.glob(f"../Results_qualitative/{CAMERA_TYPE}/{scene_name}/{f}F{fnum}/*")
            coc_paths = glob.glob(f"../blur_results/{CAMERA_TYPE}/{scene_name}/{f}F{fnum}/*")

            depth_paths = glob.glob(f"../Qualitative/{CAMERA_TYPE}/{scene_name}/depth/{f}F{fnum}_*.tif")
            imgb_paths = glob.glob(f"../Qualitative/{CAMERA_TYPE}/{scene_name}/{f}F{fnum}/{f}F{fnum}_*_B.JPG")
            if len(coc_paths) == len(depth_paths) == len(imgb_paths) ==0:
                continue
            assert len(coc_paths) == len(depth_paths) == len(imgb_paths), f"out!\n{len(coc_paths)}\n{len(depth_paths)}\n{len(imgb_paths)}"


            # dummy
            AF_paths = [None for _ in range(len(coc_paths))]

            assert len(coc_paths) == len(depth_paths) == len(imgb_paths) == len(AF_paths), "out!"

            main(coc_paths, depth_paths, imgb_paths, AF_paths, f, fnum, FOR_SUPP=not True)
