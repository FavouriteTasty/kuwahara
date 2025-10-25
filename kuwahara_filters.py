import argparse
import cv2
import numpy as np
from math import pi, cos, sin
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm


############################
# 1. SIMPLE KUWAHARA FILTER
############################
def kuwahara_simple(img, radius=5):
    """
    经典4象限桑原滤波:
    - 对于每个像素的 (2r+1)x(2r+1) 窗口
    - 划分成4个重叠子区: 左上/右上/左下/右下
    - 计算每个子区的均值和方差
    - 取方差最小子区的均值作为输出
    参数:
        img: HxWx3 RGB float32[0,1]
        radius: 邻域半径
    返回:
        out: HxWx3 float32[0,1]
    """
    h, w, c = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    # 为了高效，我们先准备积分图(积分图能O(1)求任意子区域均值/方差)
    def integral_image(I):
        # I: HxW
        # return S where S[y,x] = sum_{0<=i<=y,0<=j<=x} I[i,j]
        return I.cumsum(axis=0).cumsum(axis=1)

    # 构建每个通道的积分图 & 积分图(平方)
    int_img = [integral_image(img[..., i]) for i in range(c)]
    int_img2 = [integral_image(img[..., i] ** 2) for i in range(c)]

    def rect_stats(ii, ii2, x0, y0, x1, y1):
        """
        从积分图计算矩形区域均值和方差
        (x0,y0) inclusive, (x1,y1) inclusive
        会自动clip到图像边界
        """
        x0c = max(x0, 0)
        y0c = max(y0, 0)
        x1c = min(x1, w-1)
        y1c = min(y1, h-1)
        area = (x1c - x0c + 1) * (y1c - y0c + 1)
        if area <= 0:
            # fallback
            return np.zeros(c, dtype=np.float32), np.ones(c, dtype=np.float32)

        # helper to get sum from integral image
        def isum(ii_single):
            A = ii_single[y1c, x1c]
            B = ii_single[y0c-1, x1c] if y0c > 0 else 0.0
            C = ii_single[y1c, x0c-1] if x0c > 0 else 0.0
            D = ii_single[y0c-1, x0c-1] if (y0c > 0 and x0c > 0) else 0.0
            return A - B - C + D

        mean = np.zeros(c, dtype=np.float32)
        var = np.zeros(c, dtype=np.float32)
        for ch in range(c):
            s = isum(int_img[ch])
            s2 = isum(int_img2[ch])
            mean[ch] = s / area
            var[ch] = s2 / area - mean[ch] ** 2
        # 我们用颜色方差的和来选 (也可以用亮度方差)
        return mean, var

    # 遍历像素
    for y in tqdm(range(h), desc=f"Simple Kuwahara (r={radius})"):
        for x in range(w):
            x0 = x - radius
            x1 = x + radius
            y0 = y - radius
            y1 = y + radius
            xm = x
            ym = y

            # 四个子区
            regions = [
                (x0, y0, xm, ym),  # 左上
                (xm, y0, x1, ym),  # 右上
                (x0, ym, xm, y1),  # 左下
                (xm, ym, x1, y1),  # 右下
            ]

            best_mean = None
            best_score = None
            for (rx0, ry0, rx1, ry1) in regions:
                mean, var = rect_stats(int_img, int_img2, rx0, ry0, rx1, ry1)
                score = var.sum()  # 总方差
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_mean = mean
            out[y, x, :] = best_mean

    return np.clip(out, 0.0, 1.0)


#############################################
# 2. GENERALIZED KUWAHARA FILTER (ISOTROPIC)
#############################################
def generate_sector_samples(radius, n_sectors=8, samples_per_sector=6):
    """
    预生成扇区采样模板 (极坐标方式)。
    - n_sectors: 分成多少个角扇区，比如8
    - samples_per_sector: 每个扇区里多少半径分层采样
    返回:
        samples[sector] = [(dx,dy), ...]  (float偏移)
    """
    sectors = []
    for s in range(n_sectors):
        theta0 = (2 * pi / n_sectors) * s
        theta1 = (2 * pi / n_sectors) * (s + 1)

        # 为了更平滑，我们在该扇区内均匀选一些角度 + 半径
        pts = []
        for r_i in range(1, radius + 1):
            # 在当前扇区内随机/均匀选角度 (这里用均匀2点也行)
            for ang_i in range(samples_per_sector):
                # ang between theta0..theta1
                t = (ang_i + 0.5) / samples_per_sector
                ang = theta0 * (1 - t) + theta1 * t
                dx = r_i * cos(ang)
                dy = r_i * sin(ang)
                pts.append((dx, dy))
        sectors.append(pts)
    return sectors


def sample_region_stats(img, x, y, pts):
    """
    从img在(x,y)附近根据pts列表[(dx,dy),...]采样RGB,
    返回: mean(3,), var(3,)
    """
    h, w, _ = img.shape
    # 采样
    vals = []
    for (dx, dy) in pts:
        xx = int(round(x + dx))
        yy = int(round(y + dy))
        xx = min(max(xx, 0), w-1)
        yy = min(max(yy, 0), h-1)
        vals.append(img[yy, xx])
    if len(vals) == 0:
        v = img[y, x]
        return v, np.zeros_like(v)
    vals = np.stack(vals, axis=0)  # N x 3
    mean = vals.mean(axis=0)
    var = vals.var(axis=0)
    return mean, var


def kuwahara_generalized(
    img,
    radius=6,
    n_sectors=8,
    samples_per_sector=6,
    q=8.0,
):
    """
    generalized Kuwahara:
    - 把邻域分成多个方向扇区
    - 对每个扇区计算均值 m_i 和方差 sigma_i^2
    - 用权重 w_i = (1 / (sigma_i^2 + eps))^q 来加权融合
      (更常见写法是 exp(-lambda * sigma_i^2), 这里用(1/σ²)^q风格/等价锐度控制)

    参数:
        img: HxWx3 float32[0,1]
        radius: 采样半径(像素)
        n_sectors: 扇区数量, 4/6/8/12...
        samples_per_sector: 扇区内角度/半径采样点密度
        q: 边缘增强(锐度). 大 -> 更倾向低方差扇区 -> 更锐利
    返回:
        out: HxWx3
    """
    h, w, _ = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    sectors = generate_sector_samples(radius, n_sectors, samples_per_sector)

    eps = 1e-6
    for y in tqdm(range(h), desc=f"Generalized Kuwahara (r={radius}, q={q})"):
        for x in range(w):
            means = []
            sigmas = []
            for pts in sectors:
                mean, var = sample_region_stats(img, x, y, pts)
                means.append(mean)
                # 用亮度方差或RGB总方差都可以
                sigma2 = var.mean()  # 简化: RGB方差平均
                sigmas.append(sigma2)

            means = np.stack(means, axis=0)   # S x 3
            sigmas = np.array(sigmas)         # S

            # 权重: (1/(σ^2+eps))^q
            sigmas_clamped = np.clip(sigmas, 1e-6, 1e3)
            weights = np.exp(-q * sigmas_clamped / sigmas_clamped.mean())
            weights /= (weights.sum() + eps)

            out[y, x, :] = (means * weights[:, None]).sum(axis=0)

    return np.clip(out, 0.0, 1.0)


#####################################################
# 3. ANISOTROPIC / ORIENTATION-ADAPTIVE KUWAHARA
#####################################################
def estimate_structure_orientation(gray, sigma=2.0):
    """
    用结构张量估计每个像素的主方向 (edge方向的切线方向)
    步骤:
    1) sobel 求 Ix, Iy
    2) J = [Ix^2, IxIy; IxIy, Iy^2] 做 gaussian 平滑
    3) 主特征向量方向给出主方向theta
    返回:
        theta_map: HxW float32 方向(弧度)，范围(-pi/2, pi/2)
    """
    # sobel会给近似梯度
    Ix = sobel(gray, axis=1, mode='reflect')  # d/dx
    Iy = sobel(gray, axis=0, mode='reflect')  # d/dy

    # 结构张量分量
    Jxx = gaussian_filter(Ix * Ix, sigma=sigma)
    Jxy = gaussian_filter(Ix * Iy, sigma=sigma)
    Jyy = gaussian_filter(Iy * Iy, sigma=sigma)

    # 主方向: 0.5 * atan2(2Jxy, Jxx - Jyy)
    theta = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy + 1e-12))
    # theta 是主轴方向(最大特征值向量方向). 我们可以把它当作“局部方向”
    return theta.astype(np.float32)


def generate_anisotropic_sector_samples(
    radius,
    n_sectors=8,
    samples_per_sector=6,
    anisotropy=2.0,
    theta=0.0,
):
    """
    给定局部主方向theta，生成各向异性的采样扇区:
    - 我们先在局部坐标系 (u,v):
        u沿theta方向, v沿theta+90°
    - 以 (u/aniso, v*aniso) 的椭圆缩放，类似在主方向拉长
    - 然后再分成n_sectors个扇区(在u,v平面上按角度分区)

    注意: 这里是简单近似。
    """
    # 我们做一套局部坐标变换:
    ct = cos(theta)
    st = sin(theta)

    sectors = []
    for s in range(n_sectors):
        theta0 = (2 * pi / n_sectors) * s
        theta1 = (2 * pi / n_sectors) * (s + 1)

        pts = []
        for r_i in range(1, radius + 1):
            for ang_i in range(samples_per_sector):
                t = (ang_i + 0.5) / samples_per_sector
                ang = theta0 * (1 - t) + theta1 * t

                # 在局部各向异性椭圆半径里取点
                # 椭圆拉伸：主方向放大anisotropy，法向缩小1/anisotropy
                # 我们先生成一个圆点 (ru, rv) 再缩放
                ru = r_i * cos(ang)
                rv = r_i * sin(ang)

                ru_scaled = ru * anisotropy
                rv_scaled = rv / anisotropy

                # 再从局部(u=ct*x+st*y, v=-st*x+ct*y)反变换回全局(dx,dy)
                # 我们现在有 (u,v) = (ru_scaled, rv_scaled)
                # 逆变换:
                dx = ru_scaled * ct - rv_scaled * st
                dy = ru_scaled * st + rv_scaled * ct

                pts.append((dx, dy))
        sectors.append(pts)

    return sectors


def kuwahara_anisotropic(
    img,
    radius=6,
    n_sectors=8,
    samples_per_sector=6,
    q=8.0,
    anisotropy=2.0,
    sigma_grad=2.0,
):
    """
    anisotropic Kuwahara:
    - 先估计每个像素的主方向theta(y,x)
    - 对每个像素, 根据theta生成各向异性扇区采样模板
      (在主方向上拉长, 法向上压扁)
    - 类似generalized: 对每个扇区算均值+方差 -> 根据方差加权融合

    参数:
        img: HxWx3 float32[0,1]
        radius
        n_sectors
        samples_per_sector
        q: 锐度
        anisotropy: 各向异性拉伸系数 (>1 更细腻，主方向保留更好)
        sigma_grad: 估计结构张量时的平滑σ
    返回:
        out: HxWx3
    """
    h, w, _ = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    gray = img.mean(axis=2).astype(np.float32)
    theta_map = estimate_structure_orientation(gray, sigma=sigma_grad)

    eps = 1e-6

    for y in tqdm(range(h), desc=f"Anisotropic Kuwahara (r={radius}, aniso={anisotropy})"):
        for x in range(w):
            theta = theta_map[y, x]
            sectors = generate_anisotropic_sector_samples(
                radius,
                n_sectors=n_sectors,
                samples_per_sector=samples_per_sector,
                anisotropy=anisotropy,
                theta=theta,
            )

            means = []
            sigmas = []
            for pts in sectors:
                mean, var = sample_region_stats(img, x, y, pts)
                means.append(mean)
                sigma2 = var.mean()
                sigmas.append(sigma2)

            means = np.stack(means, axis=0)
            sigmas = np.array(sigmas)

            sigmas_clamped = np.clip(sigmas, 1e-6, 1e3)
            weights = np.exp(-q * sigmas_clamped / sigmas_clamped.mean())
            weights /= (weights.sum() + eps)
            out[y, x, :] = (means * weights[:, None]).sum(axis=0)

    return np.clip(out, 0.0, 1.0)


###########################
# UTIL: I/O + CLI EXAMPLE
###########################
def load_image_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    return rgb_f


def save_image_rgb(path, img_f):
    img8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    parser = argparse.ArgumentParser("Kuwahara filter demo")

    parser.add_argument("--input", required=True, help="input image (jpg/png)")
    parser.add_argument("--output", default="out.png", help="output image")

    parser.add_argument(
        "--mode",
        default="simple",
        choices=["simple", "generalized", "anisotropic"],
        help="which Kuwahara variant to apply",
    )

    # shared-ish params
    parser.add_argument("--radius", type=int, default=6, help="neighborhood radius")

    # generalized params
    parser.add_argument("--sectors", type=int, default=8, help="num angular sectors")
    parser.add_argument("--samples", type=int, default=6, help="samples per sector")
    parser.add_argument("--q", type=float, default=8.0, help="sharpness (higher=sharper edges)")

    # anisotropic params
    parser.add_argument("--aniso", type=float, default=2.0, help="anisotropy stretch factor (>1)")
    parser.add_argument("--sigma-grad", type=float, default=2.0, help="gaussian sigma for gradient smoothing")

    args = parser.parse_args()

    img = load_image_rgb(args.input)

    if args.mode == "simple":
        result = kuwahara_simple(img, radius=args.radius)

    elif args.mode == "generalized":
        result = kuwahara_generalized(
            img,
            radius=args.radius,
            n_sectors=args.sectors,
            samples_per_sector=args.samples,
            q=args.q,
        )

    elif args.mode == "anisotropic":
        result = kuwahara_anisotropic(
            img,
            radius=args.radius,
            n_sectors=args.sectors,
            samples_per_sector=args.samples,
            q=args.q,
            anisotropy=args.aniso,
            sigma_grad=args.sigma_grad,
        )

    else:
        raise ValueError("unknown mode")

    save_image_rgb(args.output, result)
    print(f"Saved {args.mode} Kuwahara result to {args.output}")


if __name__ == "__main__":
    main()