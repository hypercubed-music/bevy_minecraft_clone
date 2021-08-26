
use nalgebra::{Vector2, Vector3, Point};
use ndarray::*;
use std::sync::{RwLock, Arc};
use bevy::{prelude::*,
            render::pipeline::PrimitiveTopology,
            render::mesh::Indices};
use bevy_rapier3d::prelude::*;
//use nalgebra::base::{Vector2, Vector3};

const CHUNK_WIDTH : i32 = 16;
const CHUNK_HEIGHT : i32 = 16;

const TEXIMG_WIDTH : f32 = 32.0;
const TEX_OFFSETS : [[[f32;2];6];4] = [[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],],
                                            [[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0],[19.0,0.0],],
                                            [[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0],[11.0,1.0],],
                                            [[3.0,0.0],[3.0,0.0],[3.0,0.0],[3.0,0.0],[11.0,1.0],[2.0,0.0]]
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

pub struct Chunk {
    pub position : Vector3<i32>,
    blockIDs : Array3::<u8>,
}

struct Vertex {
    position : Vector3<f32>,
    uv : Vector2<f32>
}

impl Vertex {
    fn from_pos_uv(position : Vector3<f32>, uv : Vector2<f32>) -> Self {
        Self {
            position, uv
        }
    }
}

impl Chunk {

    pub fn new(position : [i32;3]) -> Self {
        Chunk {
            position : Vector3::new(position[0], position[1], position[2]),
            blockIDs : Array3::<u8>::zeros(((CHUNK_WIDTH+2) as usize, (CHUNK_HEIGHT+2) as usize, (CHUNK_WIDTH+2) as usize))
        }
    }

    pub fn render(&mut self) -> Option<(Mesh, ColliderBundle)> {

        // Renders a mesh and collider

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        let faces = self.getRenderable();
        let (verts, uvs, norms) = self.buildMesh(faces);
        if verts.len() > 0 {
            let mut new_verts : Vec<[f32;3]> = vec![];
            for v in verts.iter() {
                new_verts.push([v.x, v.y, v.z]);
            }
            let mut new_uvs : Vec<[f32;2]> = vec![];
            for u in uvs.iter() {
                new_uvs.push([u.x, u.y]);
            }
        
            mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, new_verts);
            mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, new_uvs);
            mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, norms);
            let indices : Vec<u32> = (0 as u32..verts.len() as u32).collect();
            let mut new_indices : Vec<[u32; 3]> = vec![];
            for i in (0..indices.len()).step_by(3) {
                new_indices.push([indices[i], indices[i+1], indices[i+2]]);
            }
            mesh.set_indices(Some(Indices::U32(indices)));
            let mut new_verts_2 : Vec<Point<Real, 3>> = vec![];
            for v in verts.iter() {
                new_verts_2.push(Point::<Real, 3>::new(v.x, v.y, v.z));
            }
            let collider = ColliderBundle {
                shape : ColliderShape::trimesh(new_verts_2, new_indices),
                collider_type : ColliderType::Solid,
                ..Default::default()
            };
            Some((mesh, collider))
        } else {
            None
        }
    }

    pub fn set_blocks(&mut self, blocks : Array3::<u8>) -> &Self {
        self.blockIDs = blocks;
        self
    }

    fn buildMesh(&mut self, faces : (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>))
        -> (Vec<Vector3<f32>>, Vec<Vector2<f32>>, Vec<[f32;3]>) {
        let offset = Vector3::new((self.position.x * (CHUNK_WIDTH)) as f32, 
            (self.position.y * (CHUNK_HEIGHT)) as f32,
            (self.position.z * (CHUNK_WIDTH)) as f32);
        let mut verts = vec![];
        let mut uvs = vec![];
        let mut norms = vec![];
        for r in faces.0.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[r.x as usize, r.y as usize, r.z as usize]] as usize][0];
                verts.push(r + right_face[i] + offset - Vector3::new(1.0,0.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([-1.0,0.0,0.0]);
            }
        }
        for l in faces.1.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[l.x as usize, l.y as usize, l.z as usize]] as usize][1];
                verts.push(l + left_face[i] + offset + Vector3::new(1.0,0.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([1.0,0.0,0.0]);
            }
        }
        for t in faces.2.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[t.x as usize, t.y as usize, t.z as usize]] as usize][4];
                verts.push(t + top_face[i] + offset - Vector3::new(0.0,1.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,-1.0,0.0]);
            }
        }
        for b in faces.3.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[b.x as usize, b.y as usize, b.z as usize]] as usize][5];
                verts.push(b + bottom_face[i] + offset + Vector3::new(0.0,1.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,1.0,0.0]);
            }
        }
        for f in faces.4.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[f.x as usize, f.y as usize, f.z as usize]] as usize][2];
                verts.push(f + front_face[i] + offset - Vector3::new(0.0,0.0,1.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,0.0,-1.0]);
            }
        }
        for b in faces.5.iter() {
            for i in 0..6 {
                let uv_offset = TEX_OFFSETS[self.blockIDs[[b.x as usize, b.y as usize, b.z as usize]] as usize][3];
                verts.push(b + back_face[i] + offset + Vector3::new(0.0,0.0,1.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,0.0,1.0]);
            }
        }
        (verts, uvs, norms)
    }

    fn getRenderable(&self) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>, Vec<Vector3<f32>>) {
        // Goes through blockIDs and generates a list of renderable faces
        let mut right : Vec<Vector3<f32>> = vec![];
        let mut left : Vec<Vector3<f32>> = vec![];
        let mut top : Vec<Vector3<f32>> = vec![];
        let mut bottom : Vec<Vector3<f32>> = vec![];
        let mut front : Vec<Vector3<f32>> = vec![];
        let mut back : Vec<Vector3<f32>> = vec![];
        for ((ix, iy, iz), val) in self.blockIDs.indexed_iter() {
            let iix = ix as f32;
            let iiy = iy as f32;
            let iiz = iz as f32;
            if *val == 0 {
                continue;
            }
            if ix > 0 && ix < (CHUNK_WIDTH + 1) as usize
                && iy > 0 && iy < (CHUNK_HEIGHT + 1) as usize
                && iz > 0 && iz < (CHUNK_WIDTH + 1) as usize {
                if self.blockIDs[[ix - 1, iy, iz]] == 0 {
                    right.push(Vector3::new(iix, iiy, iiz));
                }
                if self.blockIDs[[ix + 1, iy, iz]] == 0 {
                    left.push(Vector3::new(iix, iiy, iiz));
                }
                if self.blockIDs[[ix, iy - 1, iz]] == 0 {
                    top.push(Vector3::new(iix, iiy, iiz));
                }
                if self.blockIDs[[ix, iy + 1, iz]] == 0 {
                    bottom.push(Vector3::new(iix, iiy, iiz));
                }
                if self.blockIDs[[ix, iy, iz - 1]] == 0 {
                    front.push(Vector3::new(iix, iiy, iiz));
                }
                if self.blockIDs[[ix, iy, iz + 1]] == 0 {
                    back.push(Vector3::new(iix, iiy, iiz));
                }
            }

            /*if ix > 0 {
                if self.blockIDs[[ix - 1, iy, iz]] == 0 {
                    right.push(Vector3::new(iix, iiy, iiz));
                }
            }
            if ix < (CHUNK_WIDTH + 1) as usize {
                if self.blockIDs[[ix + 1, iy, iz]] == 0 {
                    left.push(Vector3::new(iix, iiy, iiz));
                }
            }
            if iy > 0 {
                if self.blockIDs[[ix, iy - 1, iz]] == 0 {
                    top.push(Vector3::new(iix, iiy, iiz));
                }
            }
            if iy < (CHUNK_HEIGHT + 1) as usize {
                if self.blockIDs[[ix, iy + 1, iz]] == 0 {
                    bottom.push(Vector3::new(iix, iiy, iiz));
                }
            }
            if iz > 0 {
                if self.blockIDs[[ix, iy, iz - 1]] == 0 {
                    front.push(Vector3::new(iix, iiy, iiz));
                }
            }
            if iz < (CHUNK_WIDTH + 1) as usize {
                if self.blockIDs[[ix, iy, iz + 1]] == 0 {
                    back.push(Vector3::new(iix, iiy, iiz));
                }
            }*/
        }
        (right, left, top, bottom, front, back)
    }

    pub fn setBlock(&mut self, pos:[i32;3], id:u8) {
        self.blockIDs[[pos[0] as usize, pos[1] as usize, pos[2] as usize]] = id;
    } 
}