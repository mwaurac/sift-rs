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
