/// KeyPoint type and the three stages of keypoint processing:
///   1. `find_scale_space_extrema`  — detect DoG local maxima/minima
///   2. `adjust_local_extrema`      — sub-pixel localization + filtering
///   3. `calc_orientation_hist` /
///      `assign_orientations`       — dominant orientation(s) per keypoint
///

use crate::image::Image;
use crate::pyramid::{DogPyramid, GaussPyramid};


const SIFT_IMG_BORDER:      usize = 5;
const SIFT_MAX_INTERP_STEPS: usize = 5;
const SIFT_ORI_HIST_BINS:   usize = 36;
const SIFT_ORI_SIG_FCTR:    f32   = 1.5;
const SIFT_ORI_RADIUS:      f32   = 4.5; // 3 * SIFT_ORI_SIG_FCTR
const SIFT_ORI_PEAK_RATIO:  f32   = 0.8;


/// A detected keypoint.
#[derive(Debug, Clone)]
pub struct KeyPoint {
    /// sub-pixel position in the original image coordinate space
    pub x: f32,
    pub y: f32,
    /// diameter of the keypoint neighborhood
    pub size: f32,
    /// dominant orientation in degrees
    pub angle: f32,
    /// DoG response value
    pub response: f32,
    /// octave index
    pub octave: i32,
    /// layer within the octave
    pub layer: i32,
}

/// scan every internal layer of the DoG pyramid for
/// pixels that are strict local maxima or minima in the 3×3×3 neighbourhood.
///
/// `threshold` = floor(0.5 * contrast_threshold / n_octave_layers * 255)
pub fn find_scale_space_extrema(
    dog_pyr: &DogPyramid,
    gauss_pyr: &GaussPyramid,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
) -> Vec<KeyPoint> {
    let threshold = (0.5 * contrast_threshold / n_octave_layers as f64 * 255.0).floor() as f32;
    let mut keypoints: Vec<KeyPoint> = Vec::new();

    for (o, octave_dog) in dog_pyr.iter().enumerate() {
        // Iterate over internal layers only: skip first and last DoG image.
        for l in 1..=n_octave_layers {
            if l >= octave_dog.len() - 1 {
                break;
            }
            let prev  = &octave_dog[l - 1];
            let curr  = &octave_dog[l    ];
            let next  = &octave_dog[l + 1];
            let h = curr.height;
            let w = curr.width;

            for r in SIFT_IMG_BORDER..(h - SIFT_IMG_BORDER) {
                for c in SIFT_IMG_BORDER..(w - SIFT_IMG_BORDER) {
                    let val = curr.get(r, c);

                    // Quick threshold check before the full 26-neighbor test
                    if val.abs() <= threshold {
                        continue;
                    }

                    if is_extremum(val, prev, curr, next, r, c) {
                        // Attempt sub-pixel localization; may produce refined keypoint
                        if let Some(kp) = adjust_local_extrema(
                            dog_pyr, gauss_pyr,
                            o, l, r, c,
                            n_octave_layers,
                            contrast_threshold,
                            edge_threshold,
                            sigma,
                        ) {
                            keypoints.push(kp);
                        }
                    }
                }
            }
        }
    }

    keypoints
}


fn pix(img: &Image, r: usize, c: usize) -> f32 { img.get(r, c) }

fn is_extremum(val: f32, prev: &Image, curr: &Image, next: &Image, r: usize, c: usize) -> bool {
    macro_rules! neighbors {
        ($img:expr) => {
            [
                pix($img, r-1, c-1), pix($img, r-1, c), pix($img, r-1, c+1),
                pix($img, r,   c-1),                    pix($img, r,   c+1),
                pix($img, r+1, c-1), pix($img, r+1, c), pix($img, r+1, c+1),
            ]
        }
    }
    let prev_n = neighbors!(prev);
    let curr_n = neighbors!(curr);
    let next_n = neighbors!(next);

    if val > 0.0 {
        // Must be greater than all 26 neighbours
        prev_n.iter().chain(curr_n.iter()).chain(next_n.iter())
            .chain(&[pix(prev, r, c), pix(next, r, c)])
            .all(|&n| val > n)
    } else {
        // Must be less than all 26 neighbours
        prev_n.iter().chain(curr_n.iter()).chain(next_n.iter())
            .chain(&[pix(prev, r, c), pix(next, r, c)])
            .all(|&n| val < n)
    }
}


/// Taylor-expand the DoG around the candidate extremum,
/// solve for the sub-pixel offset, reject keypoints that fail contrast or edge tests,
/// and emit orientation-assigned keypoints.
fn adjust_local_extrema(
    dog_pyr: &DogPyramid,
    gauss_pyr: &GaussPyramid,
    octave: usize,
    mut layer: usize,
    mut row: usize,
    mut col: usize,
    n_octave_layers: usize,
    contrast_threshold: f64,
    edge_threshold: f64,
    sigma: f64,
) -> Option<KeyPoint> {
    let img_w = dog_pyr[octave][layer].width  as i32;
    let img_h = dog_pyr[octave][layer].height as i32;

    let mut xi = 0f32; // sub-pixel offset in scale
    let mut xr = 0f32; // sub-pixel row offset
    let mut xc = 0f32; // sub-pixel col offset

    for _ in 0..SIFT_MAX_INTERP_STEPS {
        let prev = &dog_pyr[octave][layer - 1];
        let curr = &dog_pyr[octave][layer    ];
        let next = &dog_pyr[octave][layer + 1];

        let r = row; let c = col;

        // First-order finite differences (central differences)
        let dx = 0.5 * (curr.get(r, c+1) - curr.get(r, c-1));
        let dy = 0.5 * (curr.get(r+1, c) - curr.get(r-1, c));
        let ds = 0.5 * (next.get(r, c)   - prev.get(r, c));

        // Second-order derivatives (diagonal/cross)
        let v2  = 2.0 * curr.get(r, c);
        let dxx = curr.get(r, c+1) + curr.get(r, c-1) - v2;
        let dyy = curr.get(r+1, c) + curr.get(r-1, c) - v2;
        let dss = next.get(r, c)   + prev.get(r, c)   - v2;
        let dxy = 0.25 * (curr.get(r+1, c+1) - curr.get(r+1, c-1)
            - curr.get(r-1, c+1) + curr.get(r-1, c-1));
        let dxs = 0.25 * (next.get(r, c+1) - next.get(r, c-1)
            - prev.get(r, c+1) + prev.get(r, c-1));
        let dys = 0.25 * (next.get(r+1, c) - next.get(r-1, c)
            - prev.get(r+1, c) + prev.get(r-1, c));

        // Solve H * x = -g using Cramer's rule on the 3×3 Hessian.
        // H = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]
        let (ox, or_, os) = solve_3x3(
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss,
            -dx, -dy, -ds,
        );

        xc = ox; xr = or_; xi = os;

        // If offset < 0.5 in all dimensions, we've converged
        if xc.abs() < 0.5 && xr.abs() < 0.5 && xi.abs() < 0.5 {
            break;
        }

        // Move the candidate pixel if the offset is large
        col  = (col  as i32 + xc.round() as i32).clamp(SIFT_IMG_BORDER as i32, img_w - SIFT_IMG_BORDER as i32 - 1) as usize;
        row  = (row  as i32 + xr.round() as i32).clamp(SIFT_IMG_BORDER as i32, img_h - SIFT_IMG_BORDER as i32 - 1) as usize;
        layer = (layer as i32 + xi.round() as i32).clamp(1, n_octave_layers as i32) as usize;
    }

    // If still not converged, discard
    if xc.abs() >= 0.5 || xr.abs() >= 0.5 || xi.abs() >= 0.5 {
        return None;
    }

    // ── Contrast check
    let curr = &dog_pyr[octave][layer];
    let prev = &dog_pyr[octave][layer - 1];
    let next = &dog_pyr[octave][layer + 1];
    let r = row; let c = col;

    let dx = 0.5 * (curr.get(r, c+1) - curr.get(r, c-1));
    let dy = 0.5 * (curr.get(r+1, c) - curr.get(r-1, c));
    let ds = 0.5 * (next.get(r, c)   - prev.get(r, c));
    // Interpolated response
    let contr = curr.get(r, c) + 0.5 * (dx * xc + dy * xr + ds * xi);
    if contr.abs() * (n_octave_layers as f32) < (contrast_threshold as f32) {
        return None;
    }

    // ── Edge check (ratio of principal curvatures)
    let v2  = 2.0 * curr.get(r, c);
    let dxx = curr.get(r, c+1) + curr.get(r, c-1) - v2;
    let dyy = curr.get(r+1, c) + curr.get(r-1, c) - v2;
    let dxy = 0.25 * (curr.get(r+1, c+1) - curr.get(r+1, c-1)
        - curr.get(r-1, c+1) + curr.get(r-1, c-1));

    let tr  = dxx + dyy;         // trace
    let det = dxx * dyy - dxy * dxy; // determinant

    // If det <= 0 or ratio > threshold, discard edge-like extremum
    let thr = (edge_threshold as f32 + 1.0).powi(2) / edge_threshold as f32;
    if det <= 0.0 || tr * tr / det >= thr {
        return None;
    }

    // ── Build keypoint and assign orientations
    // Scale factor: octave 0 is at 2x resolution (upscaled base)
    let scale = 1.0 / (1 << octave) as f32;
    let kp_x  = (col as f32 + xc) * scale * 2.0; // *2 because we upscaled 2x
    let kp_y  = (row as f32 + xr) * scale * 2.0;
    let kp_size = sigma as f32
        * 2f32.powf((layer as f32 + xi) / n_octave_layers as f32)
        * (1 << octave) as f32 * 2.0; // ×2 for upscale

    // Encode octave and layer
    let kp_octave = octave as i32 | (layer as i32) << 8;

    // Orientation assignment uses the Gaussian layer closest to the keypoint scale
    let gauss_img = &gauss_pyr[octave][(layer + 1).min(gauss_pyr[octave].len() - 1)];

    let mut kps = assign_orientations(
        gauss_img,
        row, col,
        kp_size,
        kp_x, kp_y,
        kp_octave, layer as i32,
        contr,
        sigma,
        n_octave_layers,
        octave,
        xi,
    );

    if kps.is_empty() { None } else { Some(kps.remove(0)) }
}

/// Solve a 3×3 linear system A*x = b using Cramer's rule.
/// Returns (x0, x1, x2) or (0,0,0) if singular.
fn solve_3x3(
    a00: f32, a01: f32, a02: f32,
    a10: f32, a11: f32, a12: f32,
    a20: f32, a21: f32, a22: f32,
    b0: f32,  b1: f32,  b2: f32,
) -> (f32, f32, f32) {
    let det = a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
    if det.abs() < 1e-10 {
        return (0.0, 0.0, 0.0);
    }
    let inv = 1.0 / det;
    let x0 = inv * (b0 * (a11 * a22 - a12 * a21)
        - a01 * (b1 * a22 - a12 * b2)
        + a02 * (b1 * a21 - a11 * b2));
    let x1 = inv * (a00 * (b1 * a22 - a12 * b2)
        - b0  * (a10 * a22 - a12 * a20)
        + a02 * (a10 * b2  - b1  * a20));
    let x2 = inv * (a00 * (a11 * b2  - b1  * a21)
        - a01 * (a10 * b2  - b1  * a20)
        + b0  * (a10 * a21 - a11 * a20));
    (x0, x1, x2)
}


/// Computes the gradient orientation histogram in a circular window around the
/// keypoint and emits one KeyPoint per dominant orientation peak.
fn assign_orientations(
    img: &Image,
    row: usize,
    col: usize,
    size: f32,
    kp_x: f32,
    kp_y: f32,
    kp_octave: i32,
    kp_layer: i32,
    response: f32,
    sigma: f64,
    n_octave_layers: usize,
    octave: usize,
    xi: f32,
) -> Vec<KeyPoint> {
    let scale = (sigma as f32)
        * 2f32.powf((kp_layer as f32 + xi) / n_octave_layers as f32)
        * (1 << octave) as f32;

    let radius = (SIFT_ORI_RADIUS * scale / SIFT_ORI_SIG_FCTR).round() as i32;
    let sigma_ori = scale * SIFT_ORI_SIG_FCTR;
    let expf_scale = -1.0 / (2.0 * sigma_ori * sigma_ori);

    let mut hist = vec![0f32; SIFT_ORI_HIST_BINS];

    let h = img.height as i32;
    let w = img.width  as i32;

    for i in (-radius)..=radius {
        let y = row as i32 + i;
        if y <= 0 || y >= h - 1 { continue; }
        for j in (-radius)..=radius {
            let x = col as i32 + j;
            if x <= 0 || x >= w - 1 { continue; }

            let dx = img.get(y as usize, (x+1) as usize)
                - img.get(y as usize, (x-1) as usize);
            let dy = img.get((y-1) as usize, x as usize)
                - img.get((y+1) as usize, x as usize);

            let weight = ((i*i + j*j) as f32 * expf_scale).exp();
            let mag    = (dx*dx + dy*dy).sqrt();
            let ori    = dy.atan2(dx).to_degrees().rem_euclid(360.0);

            let bin = (ori * SIFT_ORI_HIST_BINS as f32 / 360.0) as usize;
            let bin = bin.min(SIFT_ORI_HIST_BINS - 1);
            hist[bin] += weight * mag;
        }
    }

    // Smooth histogram
    for _ in 0..6 {
        let prev_last = hist[SIFT_ORI_HIST_BINS - 1];
        let mut prev  = hist[0];
        hist[0] = (prev_last + 2.0 * hist[0] + hist[1]) * 0.25;
        for k in 1..SIFT_ORI_HIST_BINS - 1 {
            let tmp = hist[k];
            hist[k] = (prev + 2.0 * hist[k] + hist[k + 1]) * 0.25;
            prev = tmp;
        }
        let last = SIFT_ORI_HIST_BINS - 1;
        hist[last] = (prev + 2.0 * hist[last] + hist[0]) * 0.25;
    }

    let max_val = hist.iter().cloned().fold(0f32, f32::max);

    let mut keypoints = Vec::new();
    for k in 0..SIFT_ORI_HIST_BINS {
        let left  = hist[(k + SIFT_ORI_HIST_BINS - 1) % SIFT_ORI_HIST_BINS];
        let right = hist[(k + 1) % SIFT_ORI_HIST_BINS];
        let cur   = hist[k];

        if cur > left && cur > right && cur >= SIFT_ORI_PEAK_RATIO * max_val {
            // Parabolic interpolation of peak
            let bin = k as f32 + 0.5 * (left - right) / (left - 2.0 * cur + right);
            let bin = bin.rem_euclid(SIFT_ORI_HIST_BINS as f32);
            let angle = 360.0 - bin * 360.0 / SIFT_ORI_HIST_BINS as f32;
            let angle = if (angle - 360.0).abs() < 1e-4 { 0.0 } else { angle };

            keypoints.push(KeyPoint {
                x: kp_x,
                y: kp_y,
                size,
                angle,
                response,
                octave: kp_octave,
                layer: kp_layer,
            });
        }
    }

    keypoints
}

/// After all keypoints are found, retain the top `n_features` by response magnitude.
pub fn retain_best(mut kps: Vec<KeyPoint>, n_features: usize) -> Vec<KeyPoint> {
    if n_features == 0 || kps.len() <= n_features {
        return kps;
    }
    kps.sort_by(|a, b| b.response.abs().partial_cmp(&a.response.abs()).unwrap());
    kps.truncate(n_features);
    kps
}