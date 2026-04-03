/// Gaussian and Difference-of-Gaussian scale-space pyramids.
///

use crate::image::Image;

/// `gauss_pyr[octave][layer]` — the blurred images.
pub type GaussPyramid = Vec<Vec<Image>>;
/// `dog_pyr[octave][layer]` — DoG = gauss[l+1] - gauss[l].
pub type DogPyramid   = Vec<Vec<Image>>;

/// Build the Gaussian pyramid.
///
/// logic:
///   - The base image is upscaled 2x and blurred with `sigma_diff` so that the
///     effective blur is params.sigma.
///   - Each octave has `n_octave_layers + 3` images.
///   - Between octaves the image is downsampled 2x.
///   - Sigmas per layer follow: sig[k] = sigma * 2^(k/n_octave_layers).
pub fn build_gaussian_pyramid(
    base: &Image,
    n_octaves: usize,
    n_octave_layers: usize,
    sigma: f64,
) -> GaussPyramid {
    // Per-layer sigmas, relative to the previous layer (incremental).
    // sig[0] = sigma (applied to the base); sig[k] = diff so that total = sigma*2^(k/s)
    let n_layers = n_octave_layers + 3;
    let mut sig = vec![0f64; n_layers];
    sig[0] = sigma;
    let k = 2f64.powf(1.0 / n_octave_layers as f64);
    for i in 1..n_layers {
        let sig_prev = sigma * k.powi(i as i32 - 1);
        let sig_total = sig_prev * k;
        // incremental sigma needed
        sig[i] = (sig_total * sig_total - sig_prev * sig_prev).sqrt();
    }

    let mut pyr: GaussPyramid = Vec::with_capacity(n_octaves);

    for o in 0..n_octaves {
        let mut octave: Vec<Image> = Vec::with_capacity(n_layers);

        if o == 0 {
            // Base octave starts from the (already sigma-blurred) base image
            octave.push(base.gaussian_blur(sig[0]));
        } else {
            // Downsample the third-from-top layer of the previous octave
            // (index n_octave_layers in the previous octave = 2 * sigma^o)
            let prev_octave = pyr.last().unwrap();
            let src = &prev_octave[n_octave_layers];
            octave.push(src.downsample2x());
        }

        for l in 1..n_layers {
            let blurred = octave[l - 1].gaussian_blur(sig[l]);
            octave.push(blurred);
        }

        pyr.push(octave);
    }

    pyr
}

/// Build the DoG pyramid from the Gaussian pyramid.
/// `dog_pyr[o][l] = gauss_pyr[o][l+1] - gauss_pyr[o][l]`
pub fn build_dog_pyramid(gauss_pyr: &GaussPyramid) -> DogPyramid {
    gauss_pyr.iter().map(|octave| {
        octave.windows(2)
            .map(|pair| pair[1].subtract(&pair[0]))
            .collect()
    }).collect()
}

/// Number of octaves formula:
/// `nOctaves = round(log2(min(rows, cols))) - 1`
pub fn n_octaves(width: usize, height: usize) -> usize {
    let min_dim = width.min(height) as f64;
    (min_dim.log2().round() as usize).saturating_sub(1).max(1)
}

/// Prepare the base image for the pyramid.
pub fn create_base_image(img: &Image, sigma: f64) -> Image {
    const INIT_SIGMA: f64 = 0.5;
    // After 2x upscale, the effective sigma doubles: 2 * INIT_SIGMA = 1.0
    // We need to add `sigma_diff` to bring it up to `sigma`.
    let upscaled = img.upscale2x();
    let sigma_diff = (sigma * sigma - (2.0 * INIT_SIGMA) * (2.0 * INIT_SIGMA)).sqrt().max(0.01);
    upscaled.gaussian_blur(sigma_diff)
}