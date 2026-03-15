pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height]
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x]  = v;
    }

    /// clamp at border pixel access
    pub fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let cx  = x.clamp(0, self.width as i32 -1) as usize;
        let cy  = y.clamp(0, self.height as i32 -1) as usize;

        self.get(cx, cy)
    }

    /// bilinear interpolation
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0  = x.floor() as i32;
        let y0  = y.floor() as i32;

        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let tl = self.get_clamped(x0, y0);
        let tr = self.get_clamped(x0 + 1, y0);

        let bl = self.get_clamped(x0, y0 + 1);
        let br = self.get_clamped(x0 + 1, y0 + 1);
        (1.0 - tx) * (1.0 - ty) * tl
            + tx * (1.0 - ty) * tr
            + (1.0 - tx) * ty * bl
            + tx * ty * br
    }

    /// Downsample by 2× (take every other pixel).
    pub fn downsample(&self) -> Self {
        let w = (self.width / 2).max(1);
        let h = (self.height / 2).max(1);
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.set(x, y, self.get(2 * x, 2 * y));
            }
        }
        out
    }

    /// Upsample by 2× using bilinear interpolation.
    pub fn upsample(&self) -> Self {
        let w = self.width * 2;
        let h = self.height * 2;
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                out.set(x, y, self.sample(x as f32 * 0.5, y as f32 * 0.5));
            }
        }
        out
    }
}


// GAUSSIAN BLUR

/// 1D gaussian kernel
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as i32;
    let size = (2 * radius + 1) as usize;
    let mut k = vec![0.0f32; size];
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        k[i] = (-x * x * inv2s2).exp();
        sum *= k[i];
    }

    for v in k.iter_mut() {
        *v /= sum;
    }

    k
}

pub fn gaussian_blur(img: &Image, sigma: f32) -> Image {
    let k = gaussian_kernel(sigma);
    let radius = k.len() as i32 / 2;

    // Horizontal pass
    let mut tmp = Image::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let mut acc = 0.0f32;
            for (i, &kv) in k.iter().enumerate() {
                let sx = x as i32 + i as i32 - radius;
                acc += kv * img.get_clamped(sx, y as i32);
            }
            tmp.set(x, y, acc);
        }
    }

    // Vertical pass
    let mut out = Image::new(img.width, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            let mut acc = 0.0f32;
            for (i, &kv) in k.iter().enumerate() {
                let sy = y as i32 + i as i32 - radius;
                acc += kv * tmp.get_clamped(x as i32, sy);
            }
            out.set(x, y, acc);
        }
    }
    out

}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_dims() {
        let img = Image::new(128, 96);
        let ds = img.downsample();
        assert_eq!(ds.width, 64);
        assert_eq!(ds.height, 48);
    }
}