use crate::descriptor;
/// Pipeline:
///   1. Create base image (upscale 2x + initial blur)
///   2. Build Gaussian pyramid
///   3. Build DoG pyramid
///   4. Find scale-space extrema → raw keypoints
///   5. (Optionally) retain top-N by response
///   6. Compute descriptors
use crate::image::Image;
use crate::keypoint;
use crate::pyramid;
use crate::{Descriptor, KeyPoint, SiftParams};

pub fn detect_and_compute(img: &Image, params: &SiftParams) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    let base = pyramid::create_base_image(img, params.sigma);

    // n_octaves derived from original image size
    let n_octaves = pyramid::n_octaves(img.width, img.height);

    let gauss_pyr =
        pyramid::build_gaussian_pyramid(&base, n_octaves, params.n_octave_layers, params.sigma);

    let dog_pyr = pyramid::build_dog_pyramid(&gauss_pyr);

    let mut kps = keypoint::find_scale_space_extrema(
        &dog_pyr,
        &gauss_pyr,
        params.n_octave_layers,
        params.contrast_threshold,
        params.edge_threshold,
        params.sigma,
    );

    kps = keypoint::retain_best(kps, params.n_features);

    let descs = descriptor::compute_descriptors(&kps, &gauss_pyr);

    (kps, descs)
}
