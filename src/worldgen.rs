use noise::{NoiseFn, BasicMulti, Seedable, MultiFractal, Worley};
use itertools::*;
use std::cmp;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

const CHUNK_WIDTH:i32 = 16;
const CHUNK_HEIGHT:i32 = 16;

fn get_distributed_points(chunk_x : i32, chunk_z : i32, outside_size : i32, cell_size : i32, seed : u32) -> Vec<[i32;2]> {
    //Generates distributed points within a chunk
    //outside_size would be the width of the structure
    //cell_size would be the size of the cell where the structure would be
    let mut point_list : Vec<[i32;2]> = vec![];
    let start_point = [(chunk_x * CHUNK_WIDTH) - 1 - outside_size,(chunk_z * CHUNK_WIDTH) - 1 - outside_size];
    let end_point = [start_point[0] + CHUNK_WIDTH + 1 + outside_size, start_point[1] + CHUNK_WIDTH + 1 + outside_size];
    // get cells
    for cell in iproduct!((start_point[0] / cell_size)-1..=(end_point[0] / cell_size)+1, 
        (start_point[1] / cell_size)-1..=(end_point[1] / cell_size)+1) {

        let mut rng = SmallRng::seed_from_u64(seed as u64 ^ ((seed as u64) << 32) ^ (cell.0.abs() as u64) << 32 ^ cell.1.abs() as u64);
        let cell_point_noise = rng.gen_range(0..(cell_size * cell_size / 2));
        point_list.push([(cell_point_noise % cell_size) + (cell.0 * cell_size), (cell_point_noise / cell_size) + (cell.1 * cell_size)]);
    }

    point_list
}

pub struct ChunkGenerator;

// stores structure data, including size
pub struct TreeGen {
}
pub trait Structure {
    fn get_struct_points(x : i32, y : i32, z : i32, seed: u32) -> Vec<(i32, i32, i32, u8)>;
}
impl Structure for TreeGen {
    fn get_struct_points(x_pos : i32, y_pos : i32, z_pos : i32, seed: u32) -> Vec<(i32, i32, i32, u8)> {
        // Gets the points that make up a tree and their values starting at (x, y, z) within a chunk
        let mut tree_points : Vec<(i32, i32, i32, u8)> = vec![];
        let mut rng = SmallRng::seed_from_u64(seed as u64 ^ ((x_pos as u64) << 16) ^ ((z_pos as u64) << 32));

        // trunk (TODO: randomize height based on position)
        let height = rng.gen_range(4..7);
        for y in y_pos..y_pos+height {
            tree_points.push((x_pos, y, z_pos, 4));
        }
        // leaves (TODO: make sphere-ish)
        for (x, y, z) in iproduct!(x_pos-2..=x_pos+2, y_pos+height-1..=y_pos+height+3, z_pos-2..=z_pos+2) {
            //tree_points.push((x, y, z, 5));
            let dx = x - x_pos;
            let dy = y - (y_pos+height+1);
            let dz = z - z_pos;
            if (dx * dx) + (dy * dy) + (dz * dz) < 9 {
                tree_points.push((x, y, z, 5));
            }
        }

        tree_points
    }
}

impl ChunkGenerator {
    pub fn generate(position : [i32;3], seed:u32) -> Option<[[[u8; 18]; 18]; 18]> {
        let cavenoise = BasicMulti::default().set_octaves(1).set_frequency(1.0/16.0).set_seed(seed);
        let lownoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/64.0).set_seed(seed+1);
        let highnoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/64.0).set_seed(seed+2);
        let biomenoise = Worley::default().set_frequency(1.0/64.0).set_displacement(0.5).set_seed(seed);
        let _tnoise = BasicMulti::default().set_octaves(5).set_frequency(0.5).set_seed(seed+3);
        
        let mut blockIDs = [[[0; 18]; 18]; 18];
        let start_x = (position[0] * CHUNK_WIDTH) - 1;
        let start_y = (position[1] * CHUNK_HEIGHT) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - 1;

        for (x, z) in iproduct!(0..CHUNK_WIDTH+2, 0..CHUNK_WIDTH+2) {
            let float_abs_pos = [(start_x + x - 1) as f64, (start_z + z - 1) as f64];
            let low_value = (lownoise.get(float_abs_pos) * 16.0 + 32.0) as i32;
            let high_value = (highnoise.get(float_abs_pos) * 75.0 + 32.0) as i32;
            let biome = ((biomenoise.get(float_abs_pos) + 1.0) * 3.0) as i32;
            if biome <= 0 {
                //Normal
                let height = cmp::max(low_value, high_value);
                for y in 0..CHUNK_HEIGHT+2 {
                    let abs_y = y + start_y - 1;
                    let mut cave_value = 0.5 - cavenoise.get([float_abs_pos[0], abs_y as f64, float_abs_pos[1]]).abs();
                    cave_value *= if abs_y > 128 {
                        1.0
                    } else if abs_y < 13 {
                        0.1
                    } else {
                        (abs_y as f64)/128.0
                    };
                    if cave_value > 0.05 && abs_y < height {
                        blockIDs[x as usize][y as usize][z as usize] = if abs_y == height - 1 {
                            3
                        } else if abs_y > height - 4 {
                            2
                        } else {
                            1
                        };
                    }
                }
            } else if biome == 1 {
                //Desert
                let height = cmp::max(low_value, high_value);
                for y in 0..CHUNK_HEIGHT+2 {
                    let abs_y = y + start_y - 1;
                    let mut cave_value = 0.5 - cavenoise.get([float_abs_pos[0], abs_y as f64, float_abs_pos[1]]).abs();
                    cave_value *= if abs_y > 64 {
                        1.0
                    } else if abs_y < -51 {
                        0.1
                    } else {
                        (abs_y as f64+64.0)/128.0
                    };
                    if cave_value > 0.05 && abs_y < height {
                        blockIDs[x as usize][y as usize][z as usize] = if abs_y > height - 4 {
                            6
                        } else {
                            1
                        };
                    }
                }
            } else if biome >= 2 {
                // Snow
                let height = cmp::max(low_value, high_value);
                for y in 0..CHUNK_HEIGHT+2 {
                    let abs_y = y + start_y - 1;
                    let mut cave_value = 0.5 - cavenoise.get([float_abs_pos[0], abs_y as f64, float_abs_pos[1]]).abs();
                    cave_value *= if abs_y > 64 {
                        1.0
                    } else if abs_y < -51 {
                        0.1
                    } else {
                        (abs_y as f64+64.0)/128.0
                    };
                    if cave_value > 0.05 && abs_y < height {
                        blockIDs[x as usize][y as usize][z as usize] = if abs_y == height - 1 {
                            7
                        } else if abs_y > height - 4 {
                            2
                        } else {
                            1
                        };
                    }
                }
            }
        }
        //if blockIDs == Array3::<u8>::zeros(((CHUNK_WIDTH+2) as usize, (CHUNK_HEIGHT+2) as usize, (CHUNK_WIDTH+2) as usize)) {
        if blockIDs == [[[0; 18]; 18]; 18] {
            None
        } else {
            Some(blockIDs)
        }
    }

    pub fn generateStructs(position : [i32;3], seed:u32, blocks:[[[u8; 18]; 18]; 18]) -> Option<[[[u8; 18]; 18]; 18]> {
        let mut new_blocks = blocks;
        let tree_points = get_distributed_points(position[0], position[2], 3, 10, seed);
        let lownoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/64.0).set_seed(seed+1);
        let highnoise = BasicMulti::default().set_octaves(5).set_frequency(1.0/64.0).set_seed(seed+2);
        let biomenoise = Worley::default().set_frequency(1.0/64.0).set_displacement(0.5).set_seed(seed);
        let start_x = (position[0] * CHUNK_WIDTH) - 1;
        let start_y = (position[1] * CHUNK_HEIGHT) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - 1;
        for (x, z) in iproduct!(-3..=CHUNK_WIDTH+5, -3..=CHUNK_WIDTH+5) {
            let float_abs_pos = [(start_x + x) as f64, (start_z + z) as f64];
            let biome = ((biomenoise.get(float_abs_pos) + 1.0) * 3.0) as i32;
            if biome <= 0 {
                let low_value = (lownoise.get(float_abs_pos) * 16.0 + 32.0) as i32;
                let high_value = (highnoise.get(float_abs_pos) * 75.0 + 32.0) as i32;
                let height = cmp::max(low_value, high_value);
                if start_y + 18 >= height && start_y <= height + 10 && low_value > high_value {
                    let abs_xz = [start_x + x, start_z + z];
                    if tree_points.iter().any(|&i| i == abs_xz) {
                        let tree_struct = TreeGen::get_struct_points(abs_xz[0], height+1, abs_xz[1], seed);
                        for block in tree_struct.iter() {
                            let chunk_point = [block.0 - start_x + 1, block.1 - start_y, block.2 - start_z + 1];
                            // make sure the block is in the chunk
                            if chunk_point[0] >= 0 && chunk_point[1] >= 0 && chunk_point[2] >= 0
                                && chunk_point[0] < CHUNK_WIDTH+2 && chunk_point[1] < CHUNK_WIDTH+2 && chunk_point[2] < CHUNK_WIDTH+2 {
                                new_blocks[chunk_point[0] as usize][chunk_point[1] as usize][chunk_point[2] as usize] = block.3;
                            }
                        }
                    }
                }
            }
        }
        if new_blocks == [[[0; 18]; 18]; 18] {
            None
        } else {
            Some(new_blocks)
        }
    }
}