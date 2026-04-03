/// Row-major grayscale f32 image.
#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }

    /// Construct from u8 pixels, converting to f32 in [0,255].
    pub fn from_u8(width: usize, height: usize, data: &[u8]) -> Self {
        assert_eq!(data.len(), width * height);
        Self::new(width, height, data.iter().map(|&p| p as f32).collect())
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.width + col]
    }

    #[inline]
    pub fn get_clamped(&self, row: i32, col: i32) -> f32 {
        let r = row.clamp(0, self.height as i32 - 1) as usize;
        let c = col.clamp(0, self.width as i32 - 1) as usize;
        self.data[r * self.width + c]
    }

    /// Bilinear upscale by factor 2 (used to double the base image before pyramid).
    pub fn upscale2x(&self) -> Image {
        let new_w = self.width * 2;
        let new_h = self.height * 2;
        let mut out = vec![0f32; new_w * new_h];
        for r in 0..new_h {
            for c in 0..new_w {
                // map back to source coords
                let sr = (r as f32 + 0.5) / 2.0 - 0.5;
                let sc = (c as f32 + 0.5) / 2.0 - 0.5;
                let r0 = sr.floor() as i32;
                let c0 = sc.floor() as i32;
                let dr = sr - r0 as f32;
                let dc = sc - c0 as f32;
                let p00 = self.get_clamped(r0,     c0    );
                let p01 = self.get_clamped(r0,     c0 + 1);
                let p10 = self.get_clamped(r0 + 1, c0    );
                let p11 = self.get_clamped(r0 + 1, c0 + 1);
                out[r * new_w + c] =
                    p00 * (1.0 - dr) * (1.0 - dc)
                        + p01 * (1.0 - dr) * dc
                        + p10 * dr * (1.0 - dc)
                        + p11 * dr * dc;
            }
        }
        Image::new(new_w, new_h, out)
    }

    /// Gaussian blur in-place. Separable 1D kernel
    pub fn gaussian_blur(&self, sigma: f64) -> Image {
        let kernel = gaussian_kernel_1d(sigma);
        let tmp = convolve_rows(self, &kernel);
        convolve_cols(&tmp, &kernel)
    }

    /// Downsample by factor 2 (nearest even pixel)
    pub fn downsample2x(&self) -> Image {
        let new_w = self.width / 2;
        let new_h = self.height / 2;
        let mut out = vec![0f32; new_w * new_h];
        for r in 0..new_h {
            for c in 0..new_w {
                out[r * new_w + c] = self.get(r * 2, c * 2);
            }
        }
        Image::new(new_w, new_h, out)
    }

    /// Subtract another image pixel-by-pixel (for DoG).
    pub fn subtract(&self, other: &Image) -> Image {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        let data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Image::new(self.width, self.height, data)
    }
}


fn gaussian_kernel_1d(sigma: f64) -> Vec<f32> {
    let radius = ((sigma * 3.0 + 0.5) as usize).max(1);
    let size = 2 * radius + 1;
    let mut kernel = vec![0f32; size];
    let s2 = (2.0 * sigma * sigma) as f32;
    let mut sum = 0f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        kernel[i] = (-x * x / s2).exp();
        sum += kernel[i];
    }
    for k in &mut kernel {
        *k /= sum;
    }
    kernel
}

fn convolve_rows(img: &Image, kernel: &[f32]) -> Image {
    let radius = kernel.len() / 2;
    let w = img.width;
    let h = img.height;
    let mut out = vec![0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let col_src = (c as i32 + ki as i32 - radius as i32)
                    .clamp(0, w as i32 - 1) as usize;
                acc += kv * img.data[r * w + col_src];
            }
            out[r * w + c] = acc;
        }
    }
    Image::new(w, h, out)
}

fn convolve_cols(img: &Image, kernel: &[f32]) -> Image {
    let radius = kernel.len() / 2;
    let w = img.width;
    let h = img.height;
    let mut out = vec![0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let row_src = (r as i32 + ki as i32 - radius as i32)
                    .clamp(0, h as i32 - 1) as usize;
                acc += kv * img.data[row_src * w + c];
            }
            out[r * w + c] = acc;
        }
    }
    Image::new(w, h, out)
}