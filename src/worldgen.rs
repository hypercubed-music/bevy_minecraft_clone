use ndarray::*;
use noise::{NoiseFn, Perlin, BasicMulti, Curve, Seedable, MultiFractal};
use itertools::*;
use std::cmp;

const CHUNK_WIDTH:i32 = 16;
const CHUNK_HEIGHT:i32 = 16;

pub struct ChunkGenerator;

impl ChunkGenerator {
    pub fn generate(position : [i32;3], seed:u32) -> Option<Array3::<u8>> {
        let cavenoise = BasicMulti::default().set_octaves(1).set_frequency(1.0/16.0).set_seed(seed);
        let lownoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/128.0).set_seed(seed+1);
        let highnoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/128.0).set_seed(seed+2);
        let _tnoise = BasicMulti::default().set_octaves(5).set_frequency(0.5).set_seed(seed+3);
        let treenoise : noise::Curve<[f64;3]> = Curve::new(&_tnoise)
            .add_control_point(-0.5, -0.9)
            .add_control_point(-0.1, -0.4)
            .add_control_point(0.1, 0.4)
            .add_control_point(0.5, 0.9);
        
        let mut blockIDs = Array3::<u8>::zeros(((CHUNK_WIDTH+2) as usize, (CHUNK_HEIGHT+2) as usize, (CHUNK_WIDTH+2) as usize));
        let start_x = (position[0] * CHUNK_WIDTH) - 1;
        let start_y = (position[1] * CHUNK_HEIGHT) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - 1;

        for (x, y, z) in iproduct!(0..CHUNK_WIDTH+2, 0..CHUNK_HEIGHT+2, 0..CHUNK_WIDTH+2) {
            let abs_y = y + start_y - 1;
            let mut cave_value = 0.5 - cavenoise.get([(start_x + x) as f64, (start_y + y) as f64, (start_z + z) as f64]).abs();
            cave_value *= if abs_y > 64 {
                1.0
            } else if abs_y < -51 {
                0.1
            } else {
                (abs_y as f64+64.0)/128.0
            };
            let low_value = (lownoise.get([(start_x + x) as f64, (start_y + y) as f64, (start_z + z) as f64]) * 16.0 + 16.0) as i32;
            let high_value = (highnoise.get([(start_x + x) as f64, (start_y + y) as f64, (start_z + z) as f64]) * 75.0 + 16.0) as i32;
            let height = cmp::max(low_value, high_value) as i32;
            if cave_value > 0.05 && abs_y < height {
                blockIDs[[x as usize, y as usize, z as usize]] = if abs_y == height - 1 {
                    3
                } else if abs_y > height - 4 {
                    2
                } else {
                    1
                };
            }
        }
        if blockIDs == Array3::<u8>::zeros(((CHUNK_WIDTH+2) as usize, (CHUNK_HEIGHT+2) as usize, (CHUNK_WIDTH+2) as usize)) {
            None
        } else {
            Some(blockIDs)
        }
    }
}