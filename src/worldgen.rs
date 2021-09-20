//use noise::{NoiseFn, BasicMulti, Seedable, MultiFractal, Worley};
use simdnoise::*;
use itertools::*;
use std::cmp;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

const CHUNK_WIDTH:i32 = 32;
const CHUNK_WIDTH_U:usize = 32;

type BlockArray = ([[[u8; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]);

/// Generates distributed points within a chunk
/// outside_size would be the width of the structure
/// cell_size would be the size of the cell where the structure would be
fn get_distributed_points(chunk_x : i32, chunk_z : i32, outside_size : i32, cell_size : i32, seed : u32) -> Vec<[i32;2]> {
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

// stores structure data, including size
pub trait Structure {
    /// Gets the points that make up a structure and their positions
    fn get_struct_points(x : i32, y : i32, z : i32, seed: u32) -> Vec<(i32, i32, i32, u8)>;
}

pub struct TreeGen;

impl Structure for TreeGen {
    /// Gets the points that make up a tree and their positions
    fn get_struct_points(x_pos : i32, y_pos : i32, z_pos : i32, seed: u32) -> Vec<(i32, i32, i32, u8)> {
        let mut tree_points : Vec<(i32, i32, i32, u8)> = vec![];
        let mut rng = SmallRng::seed_from_u64(seed as u64 ^ ((x_pos as u64) << 16) ^ ((z_pos as u64) << 32));
        let height = rng.gen_range(4..7);
        // leaves (sphere shape)
        for (x, y, z) in iproduct!(x_pos-2..=x_pos+2, y_pos+height-1..=y_pos+height+3, z_pos-2..=z_pos+2) {
            //tree_points.push((x, y, z, 5));
            let dx = x - x_pos;
            let dy = y - (y_pos+height+1);
            let dz = z - z_pos;
            if (dx * dx) + (dy * dy) + (dz * dz) < 9 {
                tree_points.push((x, y, z, 5));
            }
        }
        // trunk 
        
        for y in y_pos..y_pos+height {
            tree_points.push((x_pos, y, z_pos, 4));
        }

        tree_points
    }
}

pub struct PineTreeGen;

impl Structure for PineTreeGen {
    /// Gets the points that make up a tree and their positions
    fn get_struct_points(x_pos : i32, y_pos : i32, z_pos : i32, seed: u32) -> Vec<(i32, i32, i32, u8)> {
        // Gets the points that make up a tree and their values starting at (x, y, z) within a chunk
        let mut tree_points : Vec<(i32, i32, i32, u8)> = vec![];
        let mut rng = SmallRng::seed_from_u64(seed as u64 ^ ((x_pos as u64) << 16) ^ ((z_pos as u64) << 32));

        let height = rng.gen_range(4..7);
        // leaves (cone shape)
        for (x, y, z) in iproduct!(x_pos-3..=x_pos+3, y_pos+height-3..=y_pos+height+3, z_pos-3..=z_pos+3) {
            let rel_y = (y - (y_pos+height-3)) as f32;
            let rad = (3.0 - (rel_y/3.0)) * (3.0 - (rel_y/3.0));
            let dx = x - x_pos;
            let dz = z - z_pos;
            if (dx * dx) + (dz * dz) < rad as i32 {
                tree_points.push((x, y, z, 5));
            }
        }
        // trunk
        
        for y in y_pos..y_pos+height {
            tree_points.push((x_pos, y, z_pos, 4));
        }

        tree_points
    }
}

pub struct ChunkGenerator {
    pos : [i32;3],
    seed : u32,
    heightmap : Vec<i32>,
    biomemap : Vec<i32>,
    cavemap : Vec<bool>,
    tree_points : Vec<(i32, i32)>,
    map_extra_width : i32,
}

impl ChunkGenerator {
    pub fn new(position : [i32;3], seed: u32, map_extra_width: i32) -> ChunkGenerator {
        let start_x = (position[0] * CHUNK_WIDTH) - map_extra_width;
        let start_y = (position[1] * CHUNK_WIDTH) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - map_extra_width;
        let ex_width_usize = map_extra_width as usize;
        let (cavenoise,_,_) = NoiseBuilder::fbm_3d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_y as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + 10)
            .with_freq(1.0/16.0).with_octaves(1).with_seed(seed as i32).generate();
        let (lownoise,_,_) = NoiseBuilder::fbm_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/16.0).with_octaves(5).with_seed((seed+1) as i32).generate();
        let (highnoise,_,_) = NoiseBuilder::fbm_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/16.0).with_octaves(5).with_seed((seed+2) as i32).generate();
        let (biomenoise,_,_) = NoiseBuilder::cellular_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/64.0).with_return_type(CellReturnType::CellValue).with_seed(seed as i32).generate();
        let mut heightmap : Vec<i32> = Vec::new();
        let mut biomemap : Vec<i32> = Vec::new();
        let mut cavemap : Vec<bool> = Vec::new();
        let mut tree_points : Vec<(i32, i32)> = Vec::new();
        let tree_xz_points = get_distributed_points(position[0], position[2], map_extra_width, 10, seed);
        let map_size = CHUNK_WIDTH + (map_extra_width * 2) + 2;
        for (x, z) in iproduct!(-map_extra_width-1..CHUNK_WIDTH+map_extra_width+1, -map_extra_width-1..CHUNK_WIDTH+map_extra_width+1) {
            let noise_idx = ((x+map_extra_width+1) * map_size + (z + map_extra_width+1)) as usize;
            let low_value = (lownoise[noise_idx] * 8.0 + 32.0) as i32;
            let high_value = (highnoise[noise_idx] * 100.0 + 16.0) as i32;
            let height = cmp::max(low_value, high_value);
            let biome = (biomenoise[noise_idx] * 3.0) as i32;

            // stuctures
            if start_y + CHUNK_WIDTH + 2 >= height && start_y <= height + 10 /*&& low_value > high_value*/ {
                let abs_xz = [start_x + x + (map_extra_width - 1), start_z + z + (map_extra_width - 1)];
                if tree_xz_points.iter().any(|&i| i == abs_xz) {
                    tree_points.push((abs_xz[0], abs_xz[1]));
                }
            }

            //cave noise
            for y in 0..CHUNK_WIDTH+2 {
                let mut cave_value = 0.5 - (cavenoise[((x + map_extra_width+1) * (CHUNK_WIDTH * CHUNK_WIDTH + 4) + y * (CHUNK_WIDTH + 2) + (z + map_extra_width+1)) as usize]).abs();
                let abs_y = y + start_y;
                cave_value *= if abs_y > 128 {
                    1.0
                } else if abs_y < 13 {
                    0.1
                } else {
                    (abs_y as f32)/128.0
                };
                cavemap.push(cave_value > 0.05);
            }
            heightmap.push(height);
            biomemap.push(biome);
        }

        ChunkGenerator {
            pos: position,
            seed,
            heightmap,
            biomemap,
            cavemap,
            tree_points,
            map_extra_width
        }
    }

    pub fn new_dont_build(position : [i32;3], seed: u32, map_extra_width: i32) -> ChunkGenerator {
        ChunkGenerator {
            pos: position,
            seed,
            heightmap: Vec::new(),
            biomemap: Vec::new(),
            cavemap: Vec::new(),
            tree_points: Vec::new(),
            map_extra_width
        }
    }

    pub fn build_maps(&mut self) {
        let start_x = (self.pos[0] * CHUNK_WIDTH) - self.map_extra_width;
        let start_y = (self.pos[1] * CHUNK_WIDTH) - 1;
        let start_z = (self.pos[2] * CHUNK_WIDTH) - self.map_extra_width;
        let ex_width_usize = self.map_extra_width as usize;
        let (cavenoise,_,_) = NoiseBuilder::fbm_3d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_y as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + 10)
            .with_freq(1.0/16.0).with_octaves(1).with_seed(self.seed as i32).generate();
        let (lownoise,_,_) = NoiseBuilder::fbm_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/16.0).with_octaves(5).with_seed((self.seed+1) as i32).generate();
        let (highnoise,_,_) = NoiseBuilder::fbm_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/16.0).with_octaves(5).with_seed((self.seed+2) as i32).generate();
        let (biomenoise,_,_) = NoiseBuilder::cellular_2d_offset(start_x as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2, start_z as f32, CHUNK_WIDTH_U + (ex_width_usize*2)+2)
            .with_freq(1.0/64.0).with_return_type(CellReturnType::CellValue).with_seed(self.seed as i32).generate();
        let tree_xz_points = get_distributed_points(self.pos[0], self.pos[2], self.map_extra_width, 10, self.seed);
        let map_size = CHUNK_WIDTH + (self.map_extra_width * 2) + 2;
        for (x, z) in iproduct!(-self.map_extra_width-1..CHUNK_WIDTH+self.map_extra_width+1, -self.map_extra_width-1..CHUNK_WIDTH+self.map_extra_width+1) {
            let noise_idx = ((x+self.map_extra_width+1) * map_size + (z + self.map_extra_width+1)) as usize;
            let low_value = (lownoise[noise_idx] * 8.0 + 32.0) as i32;
            let high_value = (highnoise[noise_idx] * 100.0 + 16.0) as i32;
            let height = cmp::max(low_value, high_value);
            let biome = (biomenoise[noise_idx] * 3.0) as i32;

            // stuctures
            if start_y + CHUNK_WIDTH + 2 >= height && start_y <= height + 10 && low_value > high_value {
                let abs_xz = [start_x + x, start_z + z];
                if tree_xz_points.iter().any(|&i| i == abs_xz) {
                    self.tree_points.push((abs_xz[0], abs_xz[1]));
                }
            }

            //cave noise
            for y in 0..CHUNK_WIDTH+2 {
                let mut cave_value = 0.5 - (cavenoise[((x + self.map_extra_width+1) * (CHUNK_WIDTH * CHUNK_WIDTH + 4) + y * (CHUNK_WIDTH + 2) + (z + self.map_extra_width+1)) as usize]).abs();
                let abs_y = y + start_y;
                cave_value *= if abs_y > 128 {
                    1.0
                } else if abs_y < 13 {
                    0.1
                } else {
                    (abs_y as f32)/128.0
                };
                self.cavemap.push(cave_value > 0.05);
            }
            self.heightmap.push(height);
            self.biomemap.push(biome);
        }
    }

    pub fn blankBlocks(&self) -> BlockArray {
        [[[0; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]
    }

    pub fn generate(&self, position : [i32;3], seed:u32) -> Option<BlockArray> {
        let mut blockIDs = [[[0; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2];
        let start_x = (position[0] * CHUNK_WIDTH) - 1;
        let start_y = (position[1] * CHUNK_WIDTH) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - 1;
        let map_size = CHUNK_WIDTH + 2 + (self.map_extra_width * 2);

        for (x, z) in iproduct!(0..CHUNK_WIDTH+2, 0..CHUNK_WIDTH+2) {
            let noise_idx = ((z+self.map_extra_width) * map_size + (x+self.map_extra_width)) as usize;
            let biome = self.biomemap[noise_idx];
            let height = self.heightmap[noise_idx];
            if biome <= 0 {
                //Normal
                for y in 0..CHUNK_WIDTH+2 {
                    let abs_y = y + start_y;
                    let is_not_cave = self.cavemap[((z+self.map_extra_width) * map_size * (CHUNK_WIDTH + 2) + (x + self.map_extra_width) * (CHUNK_WIDTH + 2) + y) as usize];
                    if is_not_cave && abs_y < height {
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
                for y in 0..CHUNK_WIDTH+2 {
                    let abs_y = y + start_y;
                    let is_not_cave = self.cavemap[((z+self.map_extra_width) * map_size * (CHUNK_WIDTH + 2) + (x + self.map_extra_width) * (CHUNK_WIDTH + 2) + y) as usize];
                    if is_not_cave && abs_y < height {
                        blockIDs[x as usize][y as usize][z as usize] = if abs_y > height - 4 {
                            6
                        } else {
                            1
                        };
                    }
                }
            } else if biome >= 2 {
                // Snow
                for y in 0..CHUNK_WIDTH+2 {
                    let abs_y = y + start_y;
                    let is_not_cave = self.cavemap[((z+self.map_extra_width) * map_size * (CHUNK_WIDTH + 2) + (x + self.map_extra_width) * (CHUNK_WIDTH + 2) + y) as usize];
                    if is_not_cave && abs_y < height {
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
        if blockIDs == [[[0; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2] {
            None
        } else {
            Some(blockIDs)
        }
    }

    pub fn generateStructs(&self, position : [i32;3], seed:u32, blocks:BlockArray)
         -> Option<BlockArray> {
        let start_x = (position[0] * CHUNK_WIDTH) - 1;
        let start_y = (position[1] * CHUNK_WIDTH) - 1;
        let start_z = (position[2] * CHUNK_WIDTH) - 1;
        let mut new_blocks = blocks;
        let map_size = CHUNK_WIDTH + (self.map_extra_width * 2) + 2;
        for (x, z) in iproduct!(-(self.map_extra_width-1)..=CHUNK_WIDTH+self.map_extra_width+1, -(self.map_extra_width-1)..=CHUNK_WIDTH+self.map_extra_width+1) {
            let noise_idx = ((z+self.map_extra_width) * map_size + (x+self.map_extra_width)) as usize;
            let biome = self.biomemap[noise_idx];
            let height = self.heightmap[noise_idx]-1;
            if biome <= 0 {
                if start_y + CHUNK_WIDTH + 2 >= height && start_y <= height + 10 {
                    //let abs_point = (start_x + x + self.map_extra_width, height, start_z + z + self.map_extra_width);
                    let abs_xz = (start_x + x, start_z + z);
                    if self.tree_points.iter().any(|&i| i == abs_xz) {
                        let tree_struct = TreeGen::get_struct_points(abs_xz.0, height, abs_xz.1, seed);
                        for block in tree_struct.iter() {
                            let chunk_point = [block.0 - start_x, block.1 - start_y + 1, block.2 - start_z];
                            // make sure the block is in the chunk
                            if chunk_point[0] >= 0 && chunk_point[1] >= 0 && chunk_point[2] >= 0
                                && chunk_point[0] < CHUNK_WIDTH+2 && chunk_point[1] < CHUNK_WIDTH+2 && chunk_point[2] < CHUNK_WIDTH+2 {
                                new_blocks[chunk_point[0] as usize][chunk_point[1] as usize][chunk_point[2] as usize] = block.3;
                            }
                        }
                    }
                }
            } else if biome >= 2 {
                if start_y + CHUNK_WIDTH + 2 >= height && start_y <= height + 10 {
                    let abs_xz = (start_x + x, start_z + z);
                    if self.tree_points.iter().any(|&i| i == abs_xz) {
                        let tree_struct = PineTreeGen::get_struct_points(abs_xz.0, height, abs_xz.1, seed);
                        for block in tree_struct.iter() {
                            let chunk_point = [block.0 - start_x, block.1 - start_y + 1, block.2 - start_z];
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
        if new_blocks == [[[0; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2] {
            None
        } else {
            Some(new_blocks)
        }
    }
}

impl Clone for ChunkGenerator {
    fn clone(&self) -> Self {
        ChunkGenerator {
            heightmap : self.heightmap.clone(),
            biomemap : self.biomemap.clone(),
            cavemap : self.cavemap.clone(),
            tree_points : self.tree_points.clone(),
            map_extra_width : self.map_extra_width,
            pos: self.pos,
            seed: self.seed
        }
    }
}