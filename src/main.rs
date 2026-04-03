/// Usage:
///   sift <input_image> [output_image]
///
/// Reads any image format (JPEG, PNG, etc.), converts to grayscale,
/// runs SIFT, draws keypoints as circles with orientation lines,
/// and writes the annotated image.
///
/// If output_image is omitted, writes to <input_stem>_keypoints.png.

use std::env;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use image::Rgb;
use image::RgbImage;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: sift <input_image> [output_image]");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path: PathBuf = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        let stem = input_path.file_stem().unwrap_or_default().to_string_lossy();
        input_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}_keypoints.png", stem))
    };

    // ── Load and convert to grayscale f32 ────────────────────────────────────
    let dyn_img = image::open(input_path).unwrap_or_else(|e| {
        eprintln!("Failed to open '{}': {}", input_path.display(), e);
        std::process::exit(1);
    });

    let (width, height) = dyn_img.dimensions();
    let gray = dyn_img.to_luma8();
    let pixels: Vec<f32> = gray.pixels().map(|p| p[0] as f32).collect();

    let sift_img = sift_rs::Image::new(width as usize, height as usize, pixels);

    // ── Run SIFT ──────────────────────────────────────────────────────────────
    let params = sift_rs::SiftParams::default();

    eprintln!(
        "Running SIFT on {}x{} image (contrast_threshold={}, edge_threshold={}, sigma={}) ...",
        width, height,
        params.contrast_threshold,
        params.edge_threshold,
        params.sigma,
    );

    let (keypoints, _) = sift_rs::detect_and_compute(&sift_img, &params);

    eprintln!("Found {} keypoints.", keypoints.len());

    // ── Draw annotated output ─────────────────────────────────────────────────
    // Start from the original colour image
    let mut out: RgbImage = dyn_img.to_rgb8();

    for kp in &keypoints {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;
        let radius = (kp.size / 2.0).max(3.0).round() as i32;

        // Circle in lime green
        draw_circle(&mut out, cx, cy, radius, Rgb([50, 230, 80]));

        // Orientation line: from center outward along kp.angle
        let angle_rad = kp.angle.to_radians();
        let ex = cx + (radius as f32 * angle_rad.cos()).round() as i32;
        let ey = cy - (radius as f32 * angle_rad.sin()).round() as i32; // y flipped
        draw_line(&mut out, cx, cy, ex, ey, Rgb([255, 220, 0]));
    }

    out.save(&output_path).unwrap_or_else(|e| {
        eprintln!("Failed to save '{}': {}", output_path.display(), e);
        std::process::exit(1);
    });

    eprintln!("Saved annotated image to '{}'.", output_path.display());

    // ── Print keypoint table to stdout ────────────────────────────────────────
    println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>12}", "idx", "x", "y", "size", "angle", "response");
    println!("{}", "-".repeat(60));
    for (i, kp) in keypoints.iter().enumerate() {
        println!(
            "{:<6} {:>8.2} {:>8.2} {:>8.2} {:>10.2} {:>12.6}",
            i, kp.x, kp.y, kp.size, kp.angle, kp.response
        );
    }
}

// ── Drawing helpers ───────────────────────────────────────────────────────────

fn put_pixel_safe(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
        img.put_pixel(x as u32, y as u32, color);
    }
}

/// Midpoint circle algorithm (Bresenham).
fn draw_circle(img: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>) {
    let mut x = r;
    let mut y = 0i32;
    let mut err = 1 - r;
    while x >= y {
        for &(px, py) in &[
            (cx+x, cy+y), (cx-x, cy+y), (cx+x, cy-y), (cx-x, cy-y),
            (cx+y, cy+x), (cx-y, cy+x), (cx+y, cy-x), (cx-y, cy-x),
        ] {
            put_pixel_safe(img, px, py, color);
        }
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Bresenham line.
fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let (mut x, mut y) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put_pixel_safe(img, x, y, color);
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}