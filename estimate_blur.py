import os
import glob

# from run_matlab import run_matlab

select_folder = "Qualitative"

save_path = "blur_results"


def exe_matlab(img_path, camera_type, select_folder, save_path):
    return os.system(f"python run_matlab.py --img_path {img_path} --mode {camera_type} --select_folder {select_folder} --save_path {save_path}")

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    
    scene_name = "Scene1"
    for scene_name in ["Scene1", "Scene2", "Scene3", "Scene4", "Scene5", "Scene6"][:]:
        print(scene_name)
        for camera_type in ["DSLR"]:
            f = 35.0
            for f in [35.0, 50.0, 85.0]:
                fnum = 1.4
                for fnum in [1.4, 1.8, 4.0, 8.0]:
                    print(scene_name, camera_type, f, fnum)
                
                    img_paths = glob.glob(f"{select_folder}/{camera_type}/{scene_name}/{f}F{fnum}/*_B.JPG")
                    print(len(img_paths))
                    with ProcessPoolExecutor(1) as executor:
                        for img_path in img_paths[:]:
                            executor.submit(exe_matlab, img_path, camera_type, select_folder, save_path)

            
    
    for scene_name in ["Scene1", "Scene2", "Scene3", "Scene4", "Scene5", "Scene6"][:]:
        for camera_type in ["PHONE"]:
            for f in [4.38]:
                for fnum in [1.73]:
                    print(scene_name, camera_type, f, fnum)
                
                    img_paths = glob.glob(f"{select_folder}/{camera_type}/{scene_name}/{f}F{fnum}/*_B.JPG")
                    print(len(img_paths))

                    with ProcessPoolExecutor(1) as executor:
                        for img_path in img_paths[:]:
                            executor.submit(exe_matlab, img_path, camera_type, select_folder, save_path)
