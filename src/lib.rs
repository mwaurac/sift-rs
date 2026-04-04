use crate::descriptor::Descriptor;
pub use crate::image::Image;
use crate::keypoint::KeyPoint;

mod descriptor;
mod image;
mod keypoint;
mod pipeline;
mod pyramid;

pub struct SiftParams {
    /// max keypoints retained
    pub n_features: usize,
    /// layers per octave in the DoG pyramid
    pub n_octave_layers: usize,
    /// contrast threshold for filtering weak extrema
    pub contrast_threshold: f64,
    /// threshold for filtering edge responses
    pub edge_threshold: f64,
    /// sigma for gaussian blur
    pub sigma: f64,
}

impl Default for SiftParams {
    fn default() -> Self {
        Self {
            n_features: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
        }
    }
}

pub fn detect_and_compute(image: &Image, params: &SiftParams) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    pipeline::detect_and_compute(image, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal image size for the pyramid to have at least one usable octave.
    const W: usize = 128;
    const H: usize = 128;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn default_params() -> SiftParams {
        SiftParams::default()
    }

    /// A blank image — should produce zero keypoints.
    fn blank_image() -> Image {
        Image::new(W, H, vec![0f32; W * H])
    }

    /// A horizontal gradient: pixel value = column index as f32.
    /// Smooth ramps have no DoG extrema → zero keypoints.
    fn gradient_image() -> Image {
        let data = (0..H).flat_map(|_| (0..W).map(|c| c as f32)).collect();
        Image::new(W, H, data)
    }

    /// A checkerboard with cell size `cell`: sharp edges create many extrema.
    fn checkerboard(cell: usize) -> Image {
        let data = (0..H)
            .flat_map(|r| {
                (0..W).map(move |c| {
                    if (r / cell + c / cell) % 2 == 0 {
                        200.0f32
                    } else {
                        50.0f32
                    }
                })
            })
            .collect();
        Image::new(W, H, data)
    }

    /// Gaussian blob at (cx, cy) with radius `r`.  Creates a single isolated
    /// blob extremum — one of the simplest structures SIFT should detect.
    fn blob_image(cx: f32, cy: f32, r: f32) -> Image {
        let data = (0..H)
            .flat_map(|row| {
                (0..W).map(move |col| {
                    let d2 = (col as f32 - cx).powi(2) + (row as f32 - cy).powi(2);
                    200.0f32 * (-d2 / (2.0 * r * r)).exp()
                })
            })
            .collect();
        Image::new(W, H, data)
    }

    // ── correctness tests ─────────────────────────────────────────────────────

    #[test]
    fn blank_produces_no_keypoints() {
        let (kps, descs) = detect_and_compute(&blank_image(), &default_params());
        assert!(kps.is_empty(), "blank image should have no keypoints");
        assert_eq!(kps.len(), descs.len());
    }

    #[test]
    fn gradient_produces_no_keypoints() {
        // A perfect linear ramp has no DoG local extrema
        let (kps, descs) = detect_and_compute(&gradient_image(), &default_params());
        assert!(kps.is_empty(), "pure gradient should have no keypoints");
        assert_eq!(kps.len(), descs.len());
    }

    #[test]
    fn checkerboard_produces_keypoints() {
        let (kps, descs) = detect_and_compute(&checkerboard(8), &default_params());
        assert!(!kps.is_empty(), "checkerboard should produce keypoints");
        assert_eq!(kps.len(), descs.len(), "one descriptor per keypoint");
    }

    #[test]
    fn blob_produces_keypoints() {
        let (kps, descs) = detect_and_compute(&blob_image(64.0, 64.0, 10.0), &default_params());
        assert!(
            !kps.is_empty(),
            "Gaussian blob should produce at least one keypoint"
        );
        assert_eq!(kps.len(), descs.len());
    }

    // ── descriptor sanity tests ───────────────────────────────────────────────

    #[test]
    fn descriptors_are_128_bytes() {
        let (kps, descs) = detect_and_compute(&checkerboard(8), &default_params());
        assert!(!descs.is_empty());
        for d in &descs {
            assert_eq!(d.len(), 128);
        }
        let _ = kps;
    }

    #[test]
    fn descriptors_are_not_all_zero() {
        let (_, descs) = detect_and_compute(&checkerboard(8), &default_params());
        assert!(!descs.is_empty());
        for d in &descs {
            assert!(
                d.iter().any(|&v| v != 0),
                "descriptor should not be all zeros"
            );
        }
    }

    // ── keypoint field tests ──────────────────────────────────────────────────

    #[test]
    fn keypoint_positions_are_within_image() {
        let (kps, _) = detect_and_compute(&checkerboard(8), &default_params());
        for kp in &kps {
            assert!(kp.x >= 0.0 && kp.x < W as f32, "x={} out of range", kp.x);
            assert!(kp.y >= 0.0 && kp.y < H as f32, "y={} out of range", kp.y);
        }
    }

    #[test]
    fn keypoint_angles_are_in_range() {
        let (kps, _) = detect_and_compute(&checkerboard(8), &default_params());
        for kp in &kps {
            assert!(
                kp.angle >= 0.0 && kp.angle < 360.0,
                "angle={} not in [0, 360)",
                kp.angle
            );
        }
    }

    #[test]
    fn keypoint_sizes_are_positive() {
        let (kps, _) = detect_and_compute(&checkerboard(8), &default_params());
        for kp in &kps {
            assert!(kp.size > 0.0, "keypoint size must be positive");
        }
    }

    #[test]
    fn keypoint_responses_are_non_negative() {
        let (kps, _) = detect_and_compute(&checkerboard(8), &default_params());
        for kp in &kps {
            assert!(
                kp.response >= 0.0,
                "response={} should be non-negative",
                kp.response
            );
        }
    }

    #[test]
    fn rectangular_images_keep_keypoints_in_bounds() {
        let w = 160;
        let h = 96;
        let data = (0..h)
            .flat_map(|r| {
                (0..w).map(move |c| {
                    if (r / 8 + c / 8) % 2 == 0 {
                        200.0f32
                    } else {
                        50.0f32
                    }
                })
            })
            .collect();
        let img = Image::new(w, h, data);
        let (kps, _) = detect_and_compute(&img, &default_params());
        for kp in &kps {
            assert!(kp.x >= 0.0 && kp.x < w as f32, "x={} out of range", kp.x);
            assert!(kp.y >= 0.0 && kp.y < h as f32, "y={} out of range", kp.y);
        }
    }

    // ── n_features cap test ───────────────────────────────────────────────────

    #[test]
    fn n_features_cap_is_respected() {
        let params = SiftParams {
            n_features: 5,
            ..default_params()
        };
        let (kps, descs) = detect_and_compute(&checkerboard(8), &params);
        assert!(kps.len() <= 5, "got {} keypoints, expected ≤5", kps.len());
        assert_eq!(kps.len(), descs.len());
    }

    // ── from_u8 constructor test ──────────────────────────────────────────────

    #[test]
    fn from_u8_constructor_works() {
        let data: Vec<u8> = (0..W * H).map(|i| (i % 256) as u8).collect();
        let img = Image::from_u8(W, H, &data);
        assert_eq!(img.width, W);
        assert_eq!(img.height, H);
        assert_eq!(img.data[0], 0.0f32);
        assert_eq!(img.data[255], 255.0f32);
    }
}
