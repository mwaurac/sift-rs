use crate::image::Image;
/// SIFT descriptor computation.
///
/// For each keypoint:
///   - Lay a 4×4 grid of cells around the keypoint (scaled + rotated).
///   - Each cell accumulates an 8-bin gradient orientation histogram.
///   - Concatenate → 128-element vector.
///   - Normalize, clamp at 0.2, renormalize
///   - Convert to u8 in [0, 255] via ×512, clamp.
use crate::keypoint::KeyPoint;
use crate::pyramid::GaussPyramid;

const SIFT_DESCR_WIDTH: usize = 4; // d: number of cells per side
const SIFT_DESCR_HIST_BINS: usize = 8; // n: orientation bins per cell
const SIFT_DESCR_SCL_FCTR: f32 = 3.0; // descriptor window scale factor
const SIFT_DESCR_MAG_THR: f32 = 0.2; // normalization clamp threshold
const SIFT_INT_DESCR_FCTR: f32 = 512.0;

/// A 128-byte SIFT descriptor.
pub type Descriptor = [u8; 128];

/// Compute descriptors for all keypoints.
pub fn compute_descriptors(keypoints: &[KeyPoint], gauss_pyr: &GaussPyramid) -> Vec<Descriptor> {
    keypoints
        .iter()
        .map(|kp| {
            let octave = kp.octave & 0xFF;
            let layer = (kp.octave >> 8) & 0xFF;

            // Gaussian image used for the descriptor: the layer above the detection layer
            let gauss_layer = ((layer + 1) as usize).min(gauss_pyr[octave as usize].len() - 1);
            let img = &gauss_pyr[octave as usize][gauss_layer];

            // Scale: keypoint size / (SIFT_DESCR_SCL_FCTR * d) * 0.5 gives the pixel
            // radius of each cell, then × (1<<octave) maps back to pyramid resolution.
            let scale_factor = 2.0 / (1 << octave) as f32;
            // Position in pyramid coordinates
            let ptx = kp.x * scale_factor;
            let pty = kp.y * scale_factor;

            let scl = kp.size * scale_factor * 0.5 * SIFT_DESCR_SCL_FCTR / SIFT_DESCR_WIDTH as f32;

            calc_sift_descriptor(img, ptx, pty, kp.angle, scl)
        })
        .collect()
}

/// core descriptor computation for one keypoint.
fn calc_sift_descriptor(
    img: &Image,
    ptx: f32,
    pty: f32,
    ori: f32, // dominant orientation in degrees
    scl: f32, // scale (pixels per descriptor cell)
) -> Descriptor {
    let d = SIFT_DESCR_WIDTH;
    let n = SIFT_DESCR_HIST_BINS;

    let cos_t = ori.to_radians().cos();
    let sin_t = ori.to_radians().sin();

    // Radius of the descriptor region (before rotation)
    let radius = ((d as f32 + 1.0) * scl * std::f32::consts::SQRT_2 * 0.5 + 0.5) as i32;

    // Accumulator: d×d cells × n bins
    let mut hist = vec![0f32; d * d * n];

    let rows = img.height as i32;
    let cols = img.width as i32;

    for i in (-radius)..=radius {
        for j in (-radius)..=radius {
            // Rotate sample offset into descriptor space
            let rot_y = (j as f32 * cos_t - i as f32 * sin_t) / scl;
            let rot_x = (j as f32 * sin_t + i as f32 * cos_t) / scl;

            // Shift so 0 is center of the d×d grid
            let rbin = rot_y + d as f32 / 2.0 - 0.5;
            let cbin = rot_x + d as f32 / 2.0 - 0.5;

            if rbin <= -1.0 || rbin >= d as f32 {
                continue;
            }
            if cbin <= -1.0 || cbin >= d as f32 {
                continue;
            }

            // Pixel coordinates
            let r = (pty + i as f32).round() as i32;
            let c = (ptx + j as f32).round() as i32;
            if r <= 0 || r >= rows - 1 {
                continue;
            }
            if c <= 0 || c >= cols - 1 {
                continue;
            }

            let dx = img.get(r as usize, (c + 1) as usize) - img.get(r as usize, (c - 1) as usize);
            let dy = img.get((r - 1) as usize, c as usize) - img.get((r + 1) as usize, c as usize);

            // Weight by Gaussian centered on descriptor window
            let rr = rot_y - (d as f32 / 2.0 - 0.5);
            let cc = rot_x - (d as f32 / 2.0 - 0.5);
            let w = (-(rr * rr + cc * cc) / (2.0 * (d as f32 / 2.0).powi(2))).exp();

            let mag = (dx * dx + dy * dy).sqrt() * w;
            // Rotate gradient orientation relative to keypoint orientation
            let grad_ori = dy.atan2(dx).to_degrees() - ori;
            let obin = grad_ori * n as f32 / 360.0;

            // Trilinear interpolation into histogram
            trilinear_add(&mut hist, rbin, cbin, obin, mag, d, n);
        }
    }

    // Normalize and convert to u8
    let norm: f32 = hist.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-7);
    let mut descriptor = [0u8; 128];
    for (i, h) in hist.iter().enumerate() {
        // Clamp, renormalize, scale to [0, 255]
        let v = (h / norm).min(SIFT_DESCR_MAG_THR);
        descriptor[i] = (v * SIFT_INT_DESCR_FCTR).round().min(255.0) as u8;
    }

    // Second normalization pass
    let norm2: f32 = descriptor
        .iter()
        .map(|&x| (x as f32).powi(2))
        .sum::<f32>()
        .sqrt()
        .max(1e-7);
    for v in &mut descriptor {
        *v = ((*v as f32 / norm2) * SIFT_INT_DESCR_FCTR)
            .round()
            .min(255.0) as u8;
    }

    descriptor
}

/// Add `val` to `hist` using trilinear interpolation across (rbin, cbin, obin).
fn trilinear_add(
    hist: &mut Vec<f32>,
    rbin: f32,
    cbin: f32,
    obin: f32,
    val: f32,
    d: usize,
    n: usize,
) {
    let r0 = rbin.floor() as i32;
    let c0 = cbin.floor() as i32;
    let o0 = obin.floor() as i32;
    let dr = rbin - r0 as f32;
    let dc = cbin - c0 as f32;
    let mut do_ = obin - o0 as f32;
    if do_ < 0.0 {
        do_ += n as f32;
    }

    // 2×2×2 interpolation
    for (ri, rv) in [(0, 1.0 - dr), (1, dr)] {
        let r = r0 + ri;
        if r < 0 || r >= d as i32 {
            continue;
        }
        for (ci, cv) in [(0, 1.0 - dc), (1, dc)] {
            let c = c0 + ci;
            if c < 0 || c >= d as i32 {
                continue;
            }
            for (oi, ov) in [(0i32, 1.0 - do_ % 1.0), (1, do_ % 1.0)] {
                let o = ((o0 + oi).rem_euclid(n as i32)) as usize;
                let idx = (r as usize * d + c as usize) * n + o;
                hist[idx] += rv * cv * ov * val;
            }
        }
    }
}
