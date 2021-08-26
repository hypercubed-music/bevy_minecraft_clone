use bevy::{
        prelude::*,
        pbr::AmbientLight};
use bevy_prototype_character_controller::{
    controller::{
        BodyTag,controller_to_pitch, controller_to_yaw
    },
};
use bevy_rapier3d::prelude::*;
mod player_utils;
use player_utils::{build_app, controller_to_kinematic, CharacterSettings, FakeKinematicRigidBody, MyRaycastSet, BlockHighlight};
use itertools::*;
use bevy_mod_raycast::{DefaultRaycastingPlugin, RayCastMesh, RayCastSource};
use nalgebra::{Vector2, Vector3, Point};
use rand::Rng;

use rand::distributions::{Distribution, Uniform};

mod chunk;
mod worldgen;

struct Game {
    chunkPos : Vec<[i32;3]>,
    chunks : Vec<Option<Entity>>,
    chunk_colliders : Vec<Option<Entity>>,
    seed : u32
}

const RENDER_DISTANCE : i32 = 5;
const RENDER_DISTANCE_VERTICAL : i32 = 3;

fn main() {
    let mut app = App::build();
    build_app(&mut app);
    app.add_plugin(RapierPhysicsPlugin::<NoUserData>::default())
        //.add_plugin(PickingPlugin)
        //.add_plugin(InteractablePickingPlugin)
        //.insert_resource(Msaa { samples: 4 })
        .insert_resource(CharacterSettings {
            focal_point: -Vec3::Z,     // Relative to head
            follow_offset: Vec3::ZERO, // Relative to head
            ..Default::default()
        })
        .insert_resource(ClearColor(Color::rgb(0.5294, 0.8078, 0.9216)))
        .insert_resource(Game {
            chunkPos : vec![],
            chunks : vec![],
            chunk_colliders : vec![],
            seed : Uniform::from(0..99999).sample(&mut rand::thread_rng())
        })
        .add_startup_system(setup.system())
        .add_system(controller_to_kinematic.system())
        .add_system(controller_to_yaw.system())
        .add_system(controller_to_pitch.system())
        .add_system(cursor_grab_system.system())
        .add_system(chunk_render_system.system())
        .add_system(handle_mouse.system())
        .run();
}

fn cursor_grab_system(
    mut windows: ResMut<Windows>,
    btn: Res<Input<MouseButton>>,
    key: Res<Input<KeyCode>>,
) {
    let window = windows.get_primary_mut().unwrap();

    if btn.just_pressed(MouseButton::Left) {
        window.set_cursor_lock_mode(true);
        window.set_cursor_visibility(false);
    }

    if key.just_pressed(KeyCode::Escape) {
        window.set_cursor_lock_mode(false);
        window.set_cursor_visibility(true);
    }
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut c_materials: ResMut<Assets<ColorMaterial>>,
    mut ambient_light: ResMut<AmbientLight>,
) {

    ambient_light.color = Color::WHITE;
    ambient_light.brightness = 0.5;

    commands.spawn_bundle(UiCameraBundle::default());

    // red dot
    commands
        .spawn_bundle(NodeBundle {
            style: Style {
                size: Size::new(Val::Px(5.0), Val::Px(5.0)),
                position_type: PositionType::Absolute,
                position: Rect {
                    left: Val::Percent(50.0),
                    bottom: Val::Percent(50.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            material: c_materials.add(Color::rgb(1.0,0.0,0.0).into()),
            ..Default::default()
        });

    // block highlight cube
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(bevy::prelude::shape::Cube{size:1.1})),
        material: materials.add(StandardMaterial{
            base_color:Color::rgba(0.8,0.8,0.8,0.2),
            unlit:true,
            ..Default::default()
        }),
        visible: Visible {
            is_transparent: true,
            ..Default::default()
        },
        ..Default::default()
    }).insert(BlockHighlight);
}

fn chunk_render_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    mut transform_query: Query<
        &mut Transform,
        (With<BodyTag>, With<FakeKinematicRigidBody>),
    >,
    mut game_state: ResMut<Game>,
) {
    let transform = transform_query.single_mut().expect("THERE CAN ONLY BE ONE");
    // generate possible chunk positions
    let p_pos : [i32;3] = [(transform.translation.x/16.0) as i32,
        (transform.translation.y/16.0) as i32,
        (transform.translation.z/16.0) as i32,];
    let mut chunk_pos : Vec<[i32;3]> = vec![];
    for (x, y, z) in iproduct!(p_pos[0]-RENDER_DISTANCE..p_pos[0]+RENDER_DISTANCE, 
        p_pos[1]-RENDER_DISTANCE_VERTICAL..p_pos[1]+RENDER_DISTANCE_VERTICAL, 
        p_pos[2]-RENDER_DISTANCE..p_pos[2]+RENDER_DISTANCE) {
            chunk_pos.push([x, y, z]);
    }
    chunk_pos.sort_by(|a, b| 
        ((a[0]-p_pos[0]).abs() + (a[1]-p_pos[0]).abs() + (a[2]-p_pos[2]).abs())
        .partial_cmp(&((b[0]-p_pos[0]).abs() + (b[1]-p_pos[0]).abs() + (b[2]-p_pos[2]).abs())).unwrap());
    // Check for chunks to add
    for pos in chunk_pos.iter() {
        if !game_state.chunkPos.iter().any(|i| i==pos) {
            // Chunk hasnt been rendered yet
            if let Some(new_gen) = worldgen::ChunkGenerator::generate(*pos, game_state.seed) {
                let mut new_chunk = chunk::Chunk::new(*pos);
                new_chunk.set_blocks(new_gen);
                // chunk can be non-empty but still have data
                if let Some((new_mesh, new_collider)) = new_chunk.render() {
                    let texture_handle = asset_server.load("atlas.png");
                    let material_handle = materials.add(StandardMaterial {
                        //base_color: Color::rgb(rand::thread_rng().gen(), rand::thread_rng().gen(), rand::thread_rng().gen()),
                        base_color_texture: Some(texture_handle.clone()),
                        unlit: true,
                        ..Default::default()
                    });
                    let chunk_id = commands.spawn_bundle(PbrBundle {
                        mesh: meshes.add(new_mesh),
                        material: material_handle,
                        ..Default::default()
                    })
                    .insert(RayCastMesh::<MyRaycastSet>::default())
                    .insert(new_chunk).id();
                    let chunk_collider_id = commands.spawn_bundle(new_collider).id();
                    game_state.chunks.push(Some(chunk_id));
                    game_state.chunk_colliders.push(Some(chunk_collider_id));
                } else {
                    game_state.chunks.push(None);
                    game_state.chunk_colliders.push(None);
                }
            } else {
                game_state.chunks.push(None);
                game_state.chunk_colliders.push(None);
            }
            game_state.chunkPos.push(*pos);
            break;
        }
    }
    // Check for chunks to remove
    for (idx,pos) in game_state.chunkPos.iter().enumerate() {
        if !chunk_pos.iter().any(|i| i==pos) {
            if let Some(chunk_id) = game_state.chunks[idx] {
                commands.entity(chunk_id).despawn();
            };
            if let Some(chunk_collider_id) = game_state.chunk_colliders[idx] {
                commands.entity(chunk_collider_id).despawn();
            };
            game_state.chunks.remove(idx);
            game_state.chunk_colliders.remove(idx);
            game_state.chunkPos.remove(idx);
            break;
        }
    }
}

fn updateChunks(
    mut game_state: ResMut<Game>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_query : Query<(&mut chunk::Chunk, &mut Handle<Mesh>)>,
    mut chunk_collider_query : Query<&mut ColliderShape>,
    chunk_pos : Vec<[i32;3]>,
    block_pos : Vec<[i32;3]>,
    id : u8
) {
    for (idx, pos) in chunk_pos.iter().enumerate() {
        let mut chunk_idx = 0;
        for (idx,ch) in game_state.chunkPos.iter_mut().enumerate() {
            if ch == pos {
                chunk_idx = idx;
                break;
            }
        }
        let mut new_chunk : Option<Entity> = None;
        let mut new_chunk_collider : Option<Entity> = None;
        if let Some(chunk_id) = game_state.chunks[chunk_idx] {
            let (mut chunk, mut chunk_mesh) = chunk_query.get_mut(chunk_id).unwrap();
            chunk.setBlock(block_pos[idx], id);
            new_chunk = Some(chunk_id);
            if let Some((new_mesh, new_collider)) = chunk.render() {
                *chunk_mesh = meshes.add(new_mesh);
                if let Some(chunk_collider_id) = game_state.chunk_colliders[chunk_idx] {
                    let mut chunk_collider = chunk_collider_query.get_mut(chunk_collider_id).unwrap();
                    *chunk_collider = new_collider.shape;
                    new_chunk_collider = Some(chunk_collider_id);
                }
            }
        }
        game_state.chunks[chunk_idx] = new_chunk;
        game_state.chunk_colliders[chunk_idx] = new_chunk_collider;
    }
}

fn handle_mouse(
    mut query: Query<&mut RayCastSource<MyRaycastSet>>,
    mut blockhighlight: Query<(&mut Transform, &mut Visible), With<BlockHighlight>>,
    chunk_query : Query<(&mut chunk::Chunk, &mut Handle<Mesh>)>,
    chunk_collider_query : Query<&mut ColliderShape>,
    buttons: Res<Input<MouseButton>>,
    game_state: ResMut<Game>,
    meshes: ResMut<Assets<Mesh>>,
) {
    // Handles block highlighting and mouse button presses
    let source = query.single_mut().unwrap(); 
    let (mut highlight_pos, mut highlight_vis) = blockhighlight.single_mut().unwrap();
    if let Some((_, intersection)) = source.intersect_top() {
        let normal = intersection.normal_ray().direction();
        let distance = intersection.distance();
        if distance < 200.0 {
            let position = intersection.position();
            let look_block : [i32;3] = [(position.x - 0.5 * normal.x).floor() as i32,
                                            (position.y - 0.5 * normal.y).floor() as i32,
                                            (position.z - 0.5 * normal.z).floor() as i32,];
            highlight_vis.is_visible = true;
            highlight_pos.translation = Vec3::new(look_block[0] as f32 + 0.55, look_block[1] as f32 + 0.55, look_block[2] as f32 + 0.55);
            if buttons.just_pressed(MouseButton::Right) && distance > 5.0 {
                println!("Mouse right button!");
                let new_block_pos = [(look_block[0] as f32 + normal.x.floor()) as i32, 
                    (look_block[1] as f32 + normal.y.floor()) as i32, 
                    (look_block[2] as f32 + normal.z.floor()) as i32];
                let mouse_chunk = [((new_block_pos[0] as f32 - 1.0) / 16.0).floor() as i32, 
                    ((new_block_pos[1] as f32 - 1.0) / 16.0).floor() as i32,
                    ((new_block_pos[2] as f32 - 1.0) / 16.0).floor() as i32];
                let rel_block_pos = [(new_block_pos[0] - (16 * mouse_chunk[0])),(new_block_pos[1] - (16 * mouse_chunk[1])),(new_block_pos[2] - (16 * mouse_chunk[2]))];
                let mut chunks_to_update : Vec<[i32;3]> = vec![mouse_chunk];
                let mut blocks_to_update : Vec<[i32;3]> = vec![rel_block_pos];
                // find chunk with correct position
                //println!("\tNew block position: {:?}", new_block_pos);
                //println!("\tIn chunk: {:?}", mouse_chunk);
                println!("\tPosition in chunk: {:?}", rel_block_pos);
                if rel_block_pos[0] == 1 {
                    chunks_to_update.push([mouse_chunk[0] - 1, mouse_chunk[1], mouse_chunk[2]]);
                    blocks_to_update.push([17, rel_block_pos[1], rel_block_pos[2]]);
                } else if rel_block_pos[0] == 16 {
                    chunks_to_update.push([mouse_chunk[0] + 1, mouse_chunk[1], mouse_chunk[2]]);
                    blocks_to_update.push([0, rel_block_pos[1], rel_block_pos[2]]);
                }
                if rel_block_pos[1] == 1 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1]-1, mouse_chunk[2]]);
                    blocks_to_update.push([rel_block_pos[0],17, rel_block_pos[2]]);
                } else if rel_block_pos[1] == 16 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1]+1, mouse_chunk[2]]);
                    blocks_to_update.push([rel_block_pos[0],0, rel_block_pos[2]]);
                }
                if rel_block_pos[2] == 1 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]-1]);
                    blocks_to_update.push([rel_block_pos[0], rel_block_pos[1], 17]);
                } else if rel_block_pos[2] == 16 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]+1]);
                    blocks_to_update.push([rel_block_pos[0], rel_block_pos[1], 0]);
                }
                updateChunks(game_state, meshes, chunk_query, chunk_collider_query, chunks_to_update, blocks_to_update, 1);
            } else if buttons.just_pressed(MouseButton::Left) {
                // add block
                println!("Mouse left button!");
                let mouse_chunk = [((look_block[0] as f32 - 1.0) / 16.0).floor() as i32, 
                    ((look_block[1] as f32 - 1.0) / 16.0).floor() as i32,
                    ((look_block[2] as f32 - 1.0) / 16.0).floor() as i32];
                let rel_block_pos = [(look_block[0] - (16 * mouse_chunk[0])),(look_block[1] - (16 * mouse_chunk[1])),(look_block[2] - (16 * mouse_chunk[2]))];
                let mut chunks_to_update : Vec<[i32;3]> = vec![mouse_chunk];
                let mut blocks_to_update : Vec<[i32;3]> = vec![rel_block_pos];
                //println!("\tDelete block position: {:?}", look_block);
                //println!("\tIn chunk: {:?}", mouse_chunk);
                println!("\tPosition in chunk: {:?}", rel_block_pos);
                if rel_block_pos[0] == 1 {
                    chunks_to_update.push([mouse_chunk[0] - 1, mouse_chunk[1], mouse_chunk[2]]);
                    blocks_to_update.push([17, rel_block_pos[1], rel_block_pos[2]]);
                } else if rel_block_pos[0] == 16 {
                    chunks_to_update.push([mouse_chunk[0] + 1, mouse_chunk[1], mouse_chunk[2]]);
                    blocks_to_update.push([0, rel_block_pos[1], rel_block_pos[2]]);
                }
                if rel_block_pos[1] == 1 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1]-1, mouse_chunk[2]]);
                    blocks_to_update.push([rel_block_pos[0],17, rel_block_pos[2]]);
                } else if rel_block_pos[1] == 16 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1]+1, mouse_chunk[2]]);
                    blocks_to_update.push([rel_block_pos[0],0, rel_block_pos[2]]);
                }
                if rel_block_pos[2] == 1 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]-1]);
                    blocks_to_update.push([rel_block_pos[0], rel_block_pos[1], 17]);
                } else if rel_block_pos[2] == 16 {
                    chunks_to_update.push([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]+1]);
                    blocks_to_update.push([rel_block_pos[0], rel_block_pos[1], 0]);
                }
                updateChunks(game_state, meshes, chunk_query, chunk_collider_query, chunks_to_update, blocks_to_update, 0);
            }
        }
    } else {
        //hide block highlight
        highlight_vis.is_visible = false;
    }
}