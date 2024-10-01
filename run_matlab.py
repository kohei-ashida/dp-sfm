import argparse
import os
import glob
import matlab.engine
import io



def run_matlab(img_path, mode, select_folder, save_path):
    eng = matlab.engine.start_matlab()
    if os.path.exists(img_path):
        pass
        # print(f"{img_path} found")
    else:
        print(f"{img_path} not found")
        return False
    
    if mode == "DSLR":
        image_resize_val = 0.5
        patch_size = 111
        ker_size = 41.0
        stride = 47
        border = 25
    elif mode == "PHONE":
        image_resize_val = 1
        patch_size = 51
        ker_size = 21.0
        stride = 27
        border = 15
    assert patch_size % 2 == 1, "patch_size is not odd"
    assert ker_size % 2 == 1, "ker_size is not odd"
    assert stride % 2 == 1, "stride is not odd"
    
    if image_resize_val != 1:
        result_name = f"{img_path.replace('_B.JPG', '')}_p_{int(patch_size)}_k_{int(ker_size)}_s_{int(stride)}_r_{image_resize_val}"
    else:
        result_name = f"{img_path.replace('_B.JPG', '')}_p_{int(patch_size)}_k_{int(ker_size)}_s_{int(stride)}_r_{int(image_resize_val)}"
    output_path = result_name.replace(select_folder, save_path)
    
    if not os.path.exists(f"{output_path}/raw.mat"):
        result_name = eng.step1(
            img_path.replace(select_folder, ""),
            image_resize_val,
            stride,
            patch_size,
            ker_size,
            border,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
    else:
        print("already processed")
    eng.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="DSLR")
    parser.add_argument("--select_folder", type=str, default="Qualitative")
    parser.add_argument("--save_path", type=str, default="blur_results")

    args = parser.parse_args()

    run_matlab(args.img_path, args.mode, args.select_folder, args.save_path)