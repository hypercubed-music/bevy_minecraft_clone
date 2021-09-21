
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

const CHUNK_WIDTH : i32 = 32;
const CHUNK_WIDTH_U : usize = 32;

pub const TEXIMG_WIDTH : f32 = 85.0;
const TEX_OFFSETS : [[[f32;2];6];8] = [[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],], //air?
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

impl MeshBuf {
    fn add_quad(
        &mut self,
        face: &OrientedCubeFace,
        quad: &UnorientedQuad,
        u_flip_face: Axis3,
        layer: u32,
    ) {
        let voxel_size = 1.0;
        let start_index = self.positions.len() as u32;
        self.positions
            .extend_from_slice(&face.quad_mesh_positions(quad, voxel_size));
        self.normals.extend_from_slice(&face.quad_mesh_normals());

        let flip_v = true;
        let mut uvs = face.tex_coords(u_flip_face, flip_v, quad);
        /*for uv in uvs.iter_mut() {
            for c in uv.iter_mut() {
                *c *= 0.1;
            }
        }*/
        self.tex_coords.extend_from_slice(&uvs);

        self.layer.extend_from_slice(&[layer; 4]);
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

    pub fn render(&mut self) -> Option<(Mesh, ColliderBundle)> {

        // Renders a mesh and collider
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        let faces = self.getRenderable();
        let (verts, uvs, norms) = self.buildMesh(faces);
        if !verts.is_empty() {
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
            let indices : Vec<u32> = (0_u32..verts.len() as u32).collect();
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

    pub fn set_blocks(&mut self, blocks : [[[u8; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]) -> &Self {
        self.blockIDs = blocks;
        self
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

    fn buildMesh(&mut self, faces : Faces)
        -> (Vec<Vector3<f32>>, Vec<Vector2<f32>>, Vec<[f32;3]>) {
        /*let offset = Vector3::new((self.position.x * (CHUNK_WIDTH)) as f32, 
            (self.position.y * (CHUNK_WIDTH)) as f32,
            (self.position.z * (CHUNK_WIDTH)) as f32);*/
        let offset = Vector3::new((self.position[0]* (CHUNK_WIDTH)) as f32, 
            (self.position[1] * (CHUNK_WIDTH)) as f32,
            (self.position[2] * (CHUNK_WIDTH)) as f32);
        let mut verts = vec![];
        let mut uvs = vec![];
        let mut norms = vec![];
        for r in faces.0.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[r.x as usize, r.y as usize, r.z as usize]] as usize][0];
                let uv_offset = TEX_OFFSETS[self.blockIDs[r.x as usize][r.y as usize][r.z as usize] as usize][0];
                verts.push(r + right_face[i] + offset - Vector3::new(1.0,0.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([-1.0,0.0,0.0]);
            }
        }
        for l in faces.1.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[l.x as usize, l.y as usize, l.z as usize]] as usize][1];
                let uv_offset = TEX_OFFSETS[self.blockIDs[l.x as usize][l.y as usize][l.z as usize] as usize][1];
                verts.push(l + left_face[i] + offset + Vector3::new(1.0,0.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([1.0,0.0,0.0]);
            }
        }
        for t in faces.2.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[t.x as usize, t.y as usize, t.z as usize]] as usize][4];
                let uv_offset = TEX_OFFSETS[self.blockIDs[t.x as usize][t.y as usize][t.z as usize] as usize][4];
                verts.push(t + top_face[i] + offset - Vector3::new(0.0,1.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,-1.0,0.0]);
            }
        }
        for b in faces.3.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[b.x as usize, b.y as usize, b.z as usize]] as usize][5];
                let uv_offset = TEX_OFFSETS[self.blockIDs[b.x as usize][b.y as usize][b.z as usize] as usize][5];
                verts.push(b + bottom_face[i] + offset + Vector3::new(0.0,1.0,0.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,1.0,0.0]);
            }
        }
        for f in faces.4.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[f.x as usize, f.y as usize, f.z as usize]] as usize][2];
                let uv_offset = TEX_OFFSETS[self.blockIDs[f.x as usize][f.y as usize][f.z as usize] as usize][2];
                verts.push(f + front_face[i] + offset - Vector3::new(0.0,0.0,1.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,0.0,-1.0]);
            }
        }
        for b in faces.5.iter() {
            for i in 0..6 {
                //let uv_offset = TEX_OFFSETS[self.blockIDs[[b.x as usize, b.y as usize, b.z as usize]] as usize][3];
                let uv_offset = TEX_OFFSETS[self.blockIDs[b.x as usize][b.y as usize][b.z as usize] as usize][3];
                verts.push(b + back_face[i] + offset + Vector3::new(0.0,0.0,1.0));
                uvs.push((uv_list[i] + Vector2::new(uv_offset[0], uv_offset[1]))/ TEXIMG_WIDTH);
                norms.push([0.0,0.0,1.0]);
            }
        }
        (verts, uvs, norms)
    }

    fn getRenderable(&self) -> Faces {
        // Goes through blockIDs and generates a list of renderable faces
        let mut right : Vec<Vector3<f32>> = vec![];
        let mut left : Vec<Vector3<f32>> = vec![];
        let mut top : Vec<Vector3<f32>> = vec![];
        let mut bottom : Vec<Vector3<f32>> = vec![];
        let mut front : Vec<Vector3<f32>> = vec![];
        let mut back : Vec<Vector3<f32>> = vec![];
        for (ix, slice) in self.blockIDs.iter().enumerate() {
            for (iy, row) in slice.iter().enumerate() {
                for (iz, val) in row.iter().enumerate() {
                    let iix = ix as f32;
                    let iiy = iy as f32;
                    let iiz = iz as f32;
                    if *val == 0 {
                        continue;
                    }
                    if ix > 0 && ix < (CHUNK_WIDTH + 1) as usize
                        && iy > 0 && iy < (CHUNK_WIDTH + 1) as usize
                        && iz > 0 && iz < (CHUNK_WIDTH + 1) as usize {
                        if self.blockIDs[ix - 1][iy][iz] == 0 {
                            right.push(Vector3::new(iix, iiy, iiz));
                        }
                        if self.blockIDs[ix + 1][iy][iz] == 0 {
                            left.push(Vector3::new(iix, iiy, iiz));
                        }
                        if self.blockIDs[ix][iy - 1][iz] == 0 {
                            top.push(Vector3::new(iix, iiy, iiz));
                        }
                        if self.blockIDs[ix][iy + 1][iz] == 0 {
                            bottom.push(Vector3::new(iix, iiy, iiz));
                        }
                        if self.blockIDs[ix][iy][iz - 1] == 0 {
                            front.push(Vector3::new(iix, iiy, iiz));
                        }
                        if self.blockIDs[ix][iy][iz + 1] == 0 {
                            back.push(Vector3::new(iix, iiy, iiz));
                        }
                    }
                }
            }
        }
        (right, left, top, bottom, front, back)
    }

    pub fn setBlock(&mut self, pos:[i32;3], id:u8) -> [[[u8;CHUNK_WIDTH_U + 2];CHUNK_WIDTH_U + 2];CHUNK_WIDTH_U + 2] {
        self.blockIDs[pos[0] as usize][pos[1] as usize][pos[2] as usize] = id;
        self.blockIDs
    } 
}