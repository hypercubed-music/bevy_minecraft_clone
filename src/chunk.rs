
use bevy_rapier3d::na::{Vector2, Vector3, Point};
use bevy::{render::pipeline::PrimitiveTopology,
            render::mesh::Indices, render::mesh::Mesh};
use bevy_rapier3d::prelude::*;
use std::time::{Duration, Instant};
use building_blocks::core::prelude::*;
use building_blocks::mesh::{
    greedy_quads, GreedyQuadsBuffer, IsOpaque, MergeVoxel, PosNormTexMesh, OrientedCubeFace, UnorientedQuad,
    RIGHT_HANDED_Y_UP_CONFIG,quad
};
use building_blocks::storage::prelude::*;
use bevy::prelude::*;
//use nalgebra::base::{Vector2, Vector3};

type Faces = (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>);

/*const CHUNK_WIDTH : i32 = 32;
const CHUNK_WIDTH_U : usize = 32;*/
use crate::{CHUNK_WIDTH, CHUNK_WIDTH_U};

pub const TEXIMG_WIDTH : f32 = 85.0;
/*const TEX_OFFSETS : [[[f32;2];6];8] = [[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],], //air?
                                            [[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0]], // stone
                                            [[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0]], //grass
                                            [[3.0,0.0],[3.0,0.0],[3.0,0.0],[3.0,0.0],[11.0,1.0],[2.0,0.0]], //grass
                                            [[28.0, 2.0], [28.0, 2.0], [28.0, 2.0], [28.0, 2.0], [29.0, 2.0], [29.0, 2.0]],  // wood
                                            [[22.0, 4.0],[22.0, 4.0],[22.0, 4.0],[22.0, 4.0],[22.0, 4.0],[22.0, 4.0]], //leaves
                                            [[3.0,1.0],[3.0,1.0],[3.0,1.0],[3.0,1.0],[3.0,1.0],[3.0,1.0]], //sand
                                            [[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[11.0,1.0],[5.0,5.0]], //snow
                                            ];

/*const uv_list : [Vector2<f32>; 6] = [Vector2::new(0.0,0.0), Vector2::new(0.0,1.0), Vector2::new(1.0,1.0), 
                                    Vector2::new(1.0,1.0), Vector2::new(1.0,0.0), Vector2::new(0.0,0.0)];*/
const uv_list : [Vector2<f32>; 6] = [Vector2::new(1.0,1.0),Vector2::new(0.0,0.0), Vector2::new(0.0,1.0), Vector2::new(0.0,0.0),Vector2::new(1.0,1.0), Vector2::new(1.0,0.0),  ];
const right_face : [Vector3<f32>; 6] = [ Vector3::new(1.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 0.0), Vector3::new(1.0, 0.0, 0.0),Vector3::new(1.0, 1.0, 0.0),Vector3::new(1.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 1.0), ];
const back_face : [Vector3<f32>; 6] = [Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0),Vector3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0), ];
const left_face : [Vector3<f32>; 6] = [ Vector3::new(0.0, 0.0, 0.0),Vector3::new(0.0, 1.0, 1.0), Vector3::new(0.0, 0.0, 1.0),Vector3::new(0.0, 1.0, 1.0), Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), ];
const front_face : [Vector3<f32>; 6] = [ Vector3::new(0.0, 0.0, 1.0),Vector3::new(1.0, 1.0, 1.0), Vector3::new(1.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 1.0),Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 1.0), ];
const bottom_face : [Vector3<f32>; 6] = [ Vector3::new(1.0, 0.0, 0.0),Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 1.0), Vector3::new(0.0, 0.0, 1.0),Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0), ];
const top_face : [Vector3<f32>; 6] = [Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 1.0), Vector3::new(0.0, 1.0, 1.0),  Vector3::new(1.0, 1.0, 1.0),Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0),];
*/

//top, middle, bottom
const layer_idx_list : [[u32;3];7] = [
    [26, 26, 26], //stone
    [9, 9, 9], //dirt
    [4, 10, 9], //grass
    [75, 74, 75], //wood
    [33, 33, 33], //leaves
    [49, 12, 9], //snow
    [48, 48, 48], //sand
    ];

#[derive(PartialEq)]
pub enum ChunkState {
    NoGen,
    GroundGen,
    StructGen,
    Rendered
}

struct MeshData {
    verts: Vec<Vector3<f32>>, 
    uvs: Vec<Vector2<f32>>, 
    norms: Vec<[f32;3]>
}

#[derive(Debug, Default, Clone)]
struct MeshBuf {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tex_coords: Vec<[f32; 2]>,
    pub layer: Vec<u32>,
    pub indices: Vec<u32>,
}

fn block_id_to_layer (id: u32, y_norm: f32) -> u32 {
    if y_norm > 0.0 {
        return layer_idx_list[id as usize][0];
    } else if y_norm == 0.0 {
        return layer_idx_list[id as usize][1];
    } else {
        return layer_idx_list[id as usize][2];
    }
}

impl MeshBuf {
    fn add_quad(
        &mut self,
        face: &OrientedCubeFace,
        quad: &UnorientedQuad,
        u_flip_face: Axis3,
        layer: u32,
        voxel_size: f32,
    ) {
        let start_index = self.positions.len() as u32;
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(quad, voxel_size));
        let normals = face.quad_mesh_normals();
        self.normals.extend_from_slice(&normals);

        let flip_v = true;
        let uvs = face.tex_coords(u_flip_face, flip_v, quad);
        self.tex_coords.extend_from_slice(&uvs);


        /*if normals[0][1] > 0.0 {
            self.layer.extend_from_slice(&[layer; 4]);
        } else if normals[0][1] == 0.0 {
            self.layer.extend_from_slice(&[layer+1; 4]);
        } else {
            self.layer.extend_from_slice(&[layer+2; 4]);
        }*/
        self.layer.extend_from_slice(&[block_id_to_layer(layer, normals[0][1]);4]);
        self.indices
            .extend_from_slice(&face.quad_mesh_indices(start_index));
    }
}

#[derive(Default, Clone, Copy)]
struct Voxel(u8);

impl MergeVoxel for Voxel {
    type VoxelValue = u8;

    fn voxel_merge_value(&self) -> Self::VoxelValue {
        self.0
    }
}

impl IsOpaque for Voxel {
    fn is_opaque(&self) -> bool {
        true
    }
}

impl IsEmpty for Voxel {
    fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

pub struct Chunk {
    //pub position : Vector3<i32>,
    pub position : [i32;3],
    //blockIDs : Array3::<u8>,
    pub blockIDs : [[[u8; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2],
}

impl Chunk {

    pub fn new(position : [i32;3]) -> Self {
        Chunk {
            //position : Vector3::new(position[0], position[1], position[2]),
            position,
            //blockIDs : Array3::<u8>::zeros(((CHUNK_WIDTH+2) as usize, (CHUNK_WIDTH+2) as usize, (CHUNK_WIDTH+2) as usize))
            blockIDs : [[[0; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2],
        }
    }

    pub fn set_blocks(&mut self, blocks : [[[u8; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]) -> &Self {
        self.blockIDs = blocks;
        self
    }

    pub fn render_with_lod(&self, lod: i32) -> Option<(Mesh, ColliderBundle)> {
        if lod == 0 {
            self.render_new()
        } else {
            let offset = [(self.position[0]* CHUNK_WIDTH), 
            (self.position[1] * CHUNK_WIDTH),
            (self.position[2] * CHUNK_WIDTH)];

            let extent = Extent3i::from_min_and_shape(PointN(offset), PointN([(CHUNK_WIDTH+2) / (lod + 1); 3]));
            let voxels = Array3x1::fill_with(extent, |point| {
                println!("{:?}", point);
                Voxel(self.blockIDs[((point.x() * (lod + 1)) - offset[0]) as usize][((point.y() * (lod + 1)) - offset[1]) as usize][((point.z() * (lod + 1)) - offset[2]) as usize])}
            );
            let mut greedy_buffer = GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
            greedy_quads(&voxels, &extent, &mut greedy_buffer);

            let mut mesh_buf = MeshBuf::default();
            for group in greedy_buffer.quad_groups.iter() {
                for quad in group.quads.iter() {
                    let mat = voxels.get(quad.minimum);
                    mesh_buf.add_quad(
                        &group.face,
                        quad,
                        RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                        mat.0 as u32 - 1,
                        (lod+1) as f32,
                    );
                }
            }

            let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

            let MeshBuf {
                positions,
                normals,
                tex_coords,
                layer,
                indices,
            } = mesh_buf;

            // copy before move
            let positions_copy = positions.clone();
            let indices_copy = indices.clone();

            render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
            render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
            render_mesh.set_attribute("Vertex_Layer", layer);
            render_mesh.set_indices(Some(Indices::U32(indices)));

            
            //conversion for collider.....
            let mut new_positions : Vec<Point<Real, 3>> = vec![];
            for v in positions_copy.iter() {
                new_positions.push(Point::<Real, 3>::new(v[0], v[1], v[2]));
            }
            let mut new_indices : Vec<[u32; 3]> = vec![];
            for i in (0..indices_copy.len()).step_by(3) {
                new_indices.push([indices_copy[i], indices_copy[i+1], indices_copy[i+2]]);
            }
            if new_positions.len() > 0 {
                let collider = ColliderBundle {
                    shape : ColliderShape::trimesh(new_positions, new_indices),
                    collider_type : ColliderType::Solid,
                    ..Default::default()
                };

                Some((render_mesh, collider))
            } else {
                None
            }
        }
    }

    pub fn render_new(&self) -> Option<(Mesh, ColliderBundle)> {
        let offset = [(self.position[0]* CHUNK_WIDTH), 
            (self.position[1] * CHUNK_WIDTH),
            (self.position[2] * CHUNK_WIDTH)];

        let extent = Extent3i::from_min_and_shape(PointN(offset), PointN([CHUNK_WIDTH+2; 3]));
        let voxels = Array3x1::fill_with(extent, |point|
            Voxel(self.blockIDs[(point.x() - offset[0]) as usize][(point.y() - offset[1]) as usize][(point.z() - offset[2]) as usize])
        );
        let mut greedy_buffer = GreedyQuadsBuffer::new(extent, RIGHT_HANDED_Y_UP_CONFIG.quad_groups());
        greedy_quads(&voxels, &extent, &mut greedy_buffer);

        let mut mesh_buf = MeshBuf::default();
        for group in greedy_buffer.quad_groups.iter() {
            for quad in group.quads.iter() {
                let mat = voxels.get(quad.minimum);
                mesh_buf.add_quad(
                    &group.face,
                    quad,
                    RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                    mat.0 as u32 - 1,
                    1.0
                );
            }
        }

        let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);

        let MeshBuf {
            positions,
            normals,
            tex_coords,
            layer,
            indices,
        } = mesh_buf;

        // copy before move
        let positions_copy = positions.clone();
        let indices_copy = indices.clone();

        render_mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        render_mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        render_mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
        render_mesh.set_attribute("Vertex_Layer", layer);
        render_mesh.set_indices(Some(Indices::U32(indices)));

        
        //conversion for collider.....
        let mut new_positions : Vec<Point<Real, 3>> = vec![];
        for v in positions_copy.iter() {
            new_positions.push(Point::<Real, 3>::new(v[0], v[1], v[2]));
        }
        let mut new_indices : Vec<[u32; 3]> = vec![];
        for i in (0..indices_copy.len()).step_by(3) {
            new_indices.push([indices_copy[i], indices_copy[i+1], indices_copy[i+2]]);
        }
        if new_positions.len() > 0 {
            let collider = ColliderBundle {
                shape : ColliderShape::trimesh(new_positions, new_indices),
                collider_type : ColliderType::Solid,
                ..Default::default()
            };

            Some((render_mesh, collider))
        } else {
            None
        }
    }

    pub fn setBlock(&mut self, pos:[i32;3], id:u8) -> [[[u8;CHUNK_WIDTH_U + 2];CHUNK_WIDTH_U + 2];CHUNK_WIDTH_U + 2] {
        self.blockIDs[pos[0] as usize][pos[1] as usize][pos[2] as usize] = id;
        self.blockIDs
    } 
}