import numpy as np
import cv2
import re
import os
from ddd_cut import crop_for_ddd
import estimate as est
from scipy.io import loadmat
from numba import njit
def calc_std_p(depth, stride, patch_size):
    depth = crop_for_ddd(depth, patch_size, stride)
    @njit
    def main(depth, stride, patch_size):
        r, c = depth.shape
        haba = np.zeros((r, c))
        for i in range(0, r, stride):
            for j in range(0, c, stride):
                haba[i : i + stride, j : j + stride] = np.nanstd(
                    depth[i : i + patch_size, j : j + patch_size]
                )
        return haba


    return main(depth, stride, patch_size)
def expand_mask(mask, expand_size):
    # mask is True or False
    mask = mask.copy()
    mask = mask.astype(np.uint8)
    kernel = np.ones((expand_size, expand_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask != 0
def optType2args(optType):
    """
    patch_size, ker_sig, stride, resize
    """
    optType = optType.replace("r_1", "r_1.0")
    r = re.compile(r"p_(\d+)_k_(\d+)_s_(\d+)_r_(\d+.\d+)")
    patch_size, ker_sig, stride, resize = r.findall(optType)[0]
    patch_size, ker_sig, stride, resize = (
        int(patch_size),
        int(ker_sig),
        int(stride),
        float(resize),
    )
    return patch_size, ker_sig, stride, resize



class DataLoader2:
    def __init__(
        self,
        optType,
        focal_length,
        estblur_path,
        depth_path,
        fnum,
        imgb_path,
        show=False,
        # target=None,
        split=None,
        afpoint_path=None,
        sensor_p=5.36e-3
    ):
        self.sensor_p = sensor_p
        self.idnum = None
        self.imgb_path = imgb_path
        self.alpha = 1.0
        self.error_map = None
        self.cocPath = estblur_path
        self.optType = optType
        self.show = show
        self.focal_length = focal_length
        self.fnum = fnum
        self.do_dddcut = False
        patch_size, ker_sig, stride, resize = optType2args(self.optType)
        self.resize = resize
        self.depthname = depth_path

        if afpoint_path is not None:
            if os.path.exists(afpoint_path):
                print(f"found AFpoint: {afpoint_path}")
                self.afpoint = np.load(afpoint_path)
            else:
                assert False, f"not found AFpoint: {afpoint_path}"
        else:
            self.afpoint = None

        # self.__load_data()

        # self.__crop_data()

        self.s = None
    def pixel2mm(self, pixel, resize):

        return pixel * self.sensor_p / resize


    def mm2pixel(self, mm, resize):
        return mm / self.pixel2mm(1, resize)

    def load_data(self):
        patch_size, ker_sig, stride, resize = optType2args(self.optType)

        assert os.path.exists(
            os.path.join(self.cocPath)
        ), f"cocPath: {self.cocPath} is not exist"

        self.depth = cv2.imread(self.depthname, cv2.IMREAD_ANYDEPTH)
        print(self.depthname)
        self.depth = self.depth.astype(np.float32)

        self.depth = self.depth * 1000
        self.depth[self.depth == 0] = np.nan

        self.depth_haba = calc_std_p(self.depth, stride=stride, patch_size=patch_size)

        # Qualitativeフォルダへのパス
        qual_path = self.cocPath.replace("Results_qualitative", "Qualitative")
        # 最後のフォルダ名を削除
        qual_path = qual_path[: qual_path.rfind("/")]
        err_img_path = self.imgb_path.replace("_B.JPG", "_error.JPG")

        if os.path.exists(err_img_path):
            self.error_img = cv2.imread(err_img_path)
            # resize
            self.error_img = cv2.resize(
                self.error_img,
                dsize=None,
                fx=self.resize,
                fy=self.resize,
                interpolation=cv2.INTER_NEAREST,
            )

            self.error_map = self.error_img[:, :, 0] != 0
            self.error_map = expand_mask(self.error_map, patch_size)


        # if os.path.exists(self.depthname.replace("_B.tif", "_B_render.tif")):
        #     # # render_depthの読み込み
        #     self.render_depth = cv2.imread(self.depthname, cv2.IMREAD_ANYDEPTH)
        #     self.render_depth = self.render_depth.astype(np.float32)
        #     # self.render_depth[self.render_depth == 0] = np.nan
        #     self.render_depth = self.render_depth * 1000

        # load reference image
        self.refIMG = cv2.imread(os.path.join(self.cocPath, "reference.png"))
        self.refIMG = cv2.cvtColor(self.refIMG, cv2.COLOR_BGR2RGB)

        if self.afpoint is not None:
            # print(self.afpoint)
            x1, y1, x2, y2 = self.afpoint[0]
            # resize
            x1, y1, x2, y2 = (
                int(x1 * self.resize),
                int(y1 * self.resize),
                int(x2 * self.resize),
                int(y2 * self.resize),
            )
            print(f"[{x1}, {y1}], [{x2}, {y2}]")
            afarea = self.depth[y1:y2, x1:x2]
            self.GTfocusdis = np.nanmean(afarea)
            print(f"GTfocusdis: {self.GTfocusdis}")
        else:
            print(f"not found GT focus distance")

        # load confidence
        raw = loadmat(os.path.join(self.cocPath, "raw.mat"))
        self.conf = raw["confidence"]
        self.out_fval = raw["out_fval"]
        self.out_sobel = raw["out_sobel"]

        self.__do_dddcut()
        assert (
            self.conf.shape == self.out_fval.shape
        ), f"depth.shape != out_fval.shape\n{self.depth.shape} != {self.out_fval.shape}"
        assert (
            self.conf.shape == self.out_sobel.shape
        ), f"depth.shape != out_sobel.shape\n{self.depth.shape} != {self.out_sobel.shape}"

        # # load blur radius
        self.blur_radius_opt = raw["target"]
        self.blur_radius_opt = self.pixel2mm(self.blur_radius_opt, self.resize)


        self.blur_radius_opt = -1 * self.blur_radius_opt

    def __do_dddcut(self):
        if not self.do_dddcut:
            patch_size, ker_sig, stride, resize = optType2args(self.optType)
            self.do_dddcut = True
            self.depth = crop_for_ddd(self.depth, patch_size, stride)
            self.out_fval = crop_for_ddd(self.out_fval, patch_size, stride)
            self.out_sobel = crop_for_ddd(self.out_sobel, patch_size, stride)
            # if self.render_depth is not None:
            #     self.render_depth = crop_for_ddd(self.render_depth, patch_size, stride)
            if self.error_map is not None:
                self.error_map = crop_for_ddd(self.error_map, patch_size, stride)

    def filtering(self, error_map):
        if self.error_map is not None:
            assert (
                self.error_map.shape == error_map.shape
            ), f"self.error_map.shape != error_map.shape\n{self.error_map.shape} != {error_map.shape}"
            
        error_map = (
            ((self.error_map) | (error_map)) if self.error_map is not None else error_map
        )
        error_map = error_map | np.isnan(self.depth) | (self.depth == 0)


        # if self.target == "231110_2" and self.focal_length == 35.0:
        #     error_map = error_map | (self.depth < 450)
        self.x = self.depth[~error_map]
        self.y = self.blur_radius_opt[~error_map] * 2 * self.alpha  # =1
        # y_pixel = self.y.copy()
        self.conf = self.conf[~error_map]
        self.out_fval = self.out_fval[~error_map]
        # reshape
        self.x = self.x.reshape(-1, 1)
        self.y = self.y.reshape(-1, 1)
        self.conf = self.conf.reshape(-1, 1)

    def close(self):
        # self.x, self.y, self.conf, self.focal_length, self.fnum, self.resize以外を削除
        del self.depth, self.blur_radius_opt

    def calc_s_and_gs(self, way="L1"):
        self.g = None
        if way in ["L1", "lstsq"]:
            pass
        else:
            raise ValueError("way must be L1 or lstsq")

        s, g = est.calc_s_and_gs(
            blurs=[self.y],
            depths=[self.x],
            focal_length=self.focal_length,
            fnum=self.fnum,
            way=way,
        )

        est_blur = est.fn(g, self.x, self.focal_length, self.fnum, s)
        est_blur = self.mm2pixel(est_blur, resize=self.resize)



        if s <= 0 or g < 0 or max(est_blur) - min(est_blur) <= 2:
            self.s, self.g = None, None
        else:
            self.s, self.g = s, g
        return s, g

