use bevy::{
        prelude::*,
        render::{
            pipeline::{FrontFace, PipelineDescriptor, RenderPipeline},
            shader::{shader_defs_system,ShaderStage, ShaderStages},
            render_graph::{base, RenderGraph, RenderResourcesNode},
            texture::{AddressMode, SamplerDescriptor},
        },
        pbr::AmbientLight,
        tasks::{AsyncComputeTaskPool, Task},
        asset::LoadState,};
use bevy_prototype_character_controller::{
    controller::{
        BodyTag,controller_to_pitch, controller_to_yaw
    },
};
use bevy_frustum_culling::*;
use std::time::{Instant};
use bevy_rapier3d::prelude::*;
mod player_utils;
use player_utils::{build_app, controller_to_kinematic, CharacterSettings, FakeKinematicRigidBody, MyRaycastSet, BlockHighlight};
use itertools::*;
use bevy_mod_raycast::{RayCastMesh, RayCastSource};
use rand::distributions::{Distribution, Uniform};
use bevy_physical_sky::{
    PhysicalSkyMaterial, PhysicalSkyPlugin, SolarPosition,
    PHYSICAL_SKY_FRAGMENT_SHADER, PHYSICAL_SKY_PASS_TIME_SYSTEM, PHYSICAL_SKY_VERTEX_SHADER,PHYSICAL_SKY_SETUP_SYSTEM, Utc, TimeZone
};
use futures_lite::future;
use rustc_hash::FxHashMap;

mod shaders;
mod chunk;
mod worldgen;

const CHUNK_WIDTH : i32 = 32;
const CHUNK_WIDTH_U : usize = 32;
const RENDER_DISTANCE : i32 = 5;
const RENDER_DISTANCE_VERTICAL : i32 = 3;

type BlockArray = ([[[u8; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]; CHUNK_WIDTH_U + 2]);

struct ChunkData {
    collider : Option<Entity>, 
    entity : Option<Entity>, 
    state : chunk::ChunkState, 
    blocks : chunk::Chunk,
    generator : worldgen::ChunkGenerator
}

struct Game {
    chunks : FxHashMap<[i32;3], ChunkData>,
    block_change_queue : Vec<([i32;3], [i32;3], u8)>,
    seed: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum AppState {
    Loading,
    Run,
}

struct Loading(Handle<Texture>);
struct ArrayTextureMaterial(pub Handle<StandardMaterial>);
struct ArrayTexturePipelines(pub RenderPipelines);

fn main() {
    let mut app = App::build();
    build_app(&mut app);
    app.add_plugin(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugin(BoundingVolumePlugin::<obb::Obb>::default())
        .add_plugin(FrustumCullingPlugin::<obb::Obb>::default())
        .insert_resource(CharacterSettings {
            focal_point: -Vec3::Z,     // Relative to head
            follow_offset: Vec3::ZERO, // Relative to head
            ..Default::default()
        })
        // Sky
        .insert_resource(SolarPosition {
            // Stockholm
            latitude: 0.0,
            longitude: 0.0,
            // one day per 8 minutes of real time
            simulation_seconds_per_second: 24.0 * 60.0 * 60.0 / (20.0 * 60.0),
            now: Utc.ymd(2021, 03, 01).and_hms(13, 0, 0),
            ..Default::default()
        })
        .add_plugin(PhysicalSkyPlugin)
        //.add_system_set(SystemSet::on_enter(AppState::Loading).with_system(load_assets.system()))
        //.add_system_set(SystemSet::on_update(AppState::Loading).with_system(check_loaded.system()))
        .add_system(
            update_sun_light_position
                .system()
                .label("update_sun_light_position")
                .after(PHYSICAL_SKY_PASS_TIME_SYSTEM),
        )
        //.add_startup_system(setup.system().after(PHYSICAL_SKY_SETUP_SYSTEM))
        .add_startup_system(setup.system().after(PHYSICAL_SKY_SETUP_SYSTEM))
        .insert_resource(State::new(AppState::Loading))
        .add_state(AppState::Loading)
        .add_system_set(SystemSet::on_enter(AppState::Loading).with_system(load_assets.system()))
        .add_system_set(SystemSet::on_update(AppState::Loading).with_system(check_loaded.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(controller_to_kinematic.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(controller_to_yaw.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(controller_to_pitch.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(cursor_grab_system.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(chunk_instancing_system.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(chunk_generate_system.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(handle_tasks.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(chunk_despawn_system.system()))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(handle_mouse.system().label("mouse")))
        .add_system_set(SystemSet::on_update(AppState::Run).with_system(updateChunks.system().after("mouse")))
        .run();
}

fn load_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    let handle = asset_server.load("array_texture.png");
    commands.insert_resource(Loading(handle));
}

/// Make sure that our texture is loaded so we can change some settings on it later
fn check_loaded(
    mut state: ResMut<State<AppState>>,
    handle: Res<Loading>,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut textures: ResMut<Assets<Texture>>,
    mut commands: Commands,
) {
    if let LoadState::Loaded = asset_server.get_load_state(&handle.0) {
        let texture = textures.get_mut(handle.0.clone()).unwrap();
        texture.sampler = SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            ..Default::default()
        };
        texture.reinterpret_stacked_2d_as_array(chunk::TEXIMG_WIDTH as u32);
        let material_handle = materials.add(StandardMaterial {
            base_color_texture: Some(handle.0.clone()),
            metallic: 0.0,
            roughness: 1.0,
            reflectance: 0.0,
            ..Default::default()
        });
        commands.insert_resource(ArrayTextureMaterial(material_handle));
        state.set(AppState::Run).unwrap();
    }
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut c_materials: ResMut<Assets<ColorMaterial>>,
    mut sky_materials: ResMut<Assets<PhysicalSkyMaterial>>,
    mut pipelines: ResMut<Assets<PipelineDescriptor>>,
    mut shaders: ResMut<Assets<Shader>>,
    mut ambient_light: ResMut<AmbientLight>,
) {
    //let texture_handle = asset_server.load("atlas.png");
    /*let texture = textures.get_mut(texture_handle.clone()).unwrap();
    texture.sampler = SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        ..Default::default()
    };
    texture.reinterpret_stacked_2d_as_array(chunk::TEXIMG_WIDTH as u32);*/

    // Create a new shader pipeline
    let pipeline = pipelines.add(PipelineDescriptor::default_config(ShaderStages {
        vertex: shaders.add(Shader::from_glsl(
            ShaderStage::Vertex,
            shaders::VERTEX_SHADER,
        )),
        fragment: Some(shaders.add(Shader::from_glsl(
            ShaderStage::Fragment,
            shaders::FRAGMENT_SHADER,
        ))),
    }));

    commands.insert_resource(ArrayTexturePipelines(RenderPipelines::from_pipelines(
        vec![RenderPipeline::new(pipeline)],
    )));

    commands.insert_resource(Game {
        chunks: FxHashMap::default(),
        block_change_queue : Vec::new(),
        seed : Uniform::from(0..99999).sample(&mut rand::thread_rng()),
        //block_texture : texture_handle
    });

    ambient_light.color = Color::WHITE;
    ambient_light.brightness = 0.15;
    // Create a new shader pipeline
    let mut pipeline_descriptor = PipelineDescriptor::default_config(ShaderStages {
        vertex: shaders.add(Shader::from_glsl(
            ShaderStage::Vertex,
            PHYSICAL_SKY_VERTEX_SHADER,
        )),
        fragment: Some(shaders.add(Shader::from_glsl(
            ShaderStage::Fragment,
            PHYSICAL_SKY_FRAGMENT_SHADER,
        ))),
    });
    // Reverse the winding so we can see the faces from the inside
    pipeline_descriptor.primitive.front_face = FrontFace::Cw;
    let pipeline = pipelines.add(pipeline_descriptor);

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
            material: c_materials.add(bevy::prelude::Color::rgb(1.0,0.0,0.0).into()),
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

    //sun
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(
            1000.5,
            592.0,
            3200.5,
        )),
        light: Light {
            color: Color::ANTIQUE_WHITE,
            intensity: 90000000.0,
            depth: 0.1..9000000.0,
            range: 9000000.0,
            ..Default::default()
        },
        ..Default::default()
    });

    // Create a new material
    let material = sky_materials.add(PhysicalSkyMaterial::red_sunset(true));

    // Sky box cube
    commands
        .spawn_bundle(MeshBundle {
            mesh: meshes.add(Mesh::from(bevy::prelude::shape::Icosphere {
                radius: 1000.0,
                subdivisions: 5,
            })),
            render_pipelines: RenderPipelines::from_pipelines(vec![RenderPipeline::new(pipeline)]),
            transform: Transform::from_xyz(0.5, 80.0, 0.5),
            ..Default::default()
        })
        .insert(material);
}

fn update_sun_light_position(
    solar_position: Res<SolarPosition>,
    mut query: Query<(&mut Transform, &mut Light), With<Light>>,
) {
    let (azimuth, inclination) = solar_position.get_azimuth_inclination();
    let (azimuth_radians, inclination_radians) = (
        (azimuth.to_radians() - std::f64::consts::PI) as f32,
        inclination.to_radians() as f32,
    );
    let translation = Vec3::new(
        azimuth_radians.cos(),
        azimuth_radians.sin() * inclination_radians.sin(),
        azimuth_radians.sin() * inclination_radians.cos(),
    )
    .normalize()
        * 4500.0;
    for (mut transform, mut light) in query.iter_mut() {
        *transform = Transform::from_translation(translation);
        (*light).intensity = if translation.y > 0.0 {90000000.0} else {0.0};
    }
}

fn chunk_instancing_system (mut transform_query: Query<
    &mut Transform,
    (With<BodyTag>, With<FakeKinematicRigidBody>),
>,
mut game_state: ResMut<Game>,) {
    let seed = game_state.seed.clone();
    // INstances a chunk
    let transform = transform_query.single_mut().expect("THERE CAN ONLY BE ONE");
    // generate possible chunk positions
    let float_chunk_width = CHUNK_WIDTH as f32;
    let p_pos : [i32;3] = [(transform.translation.x/float_chunk_width) as i32,
        (transform.translation.y/float_chunk_width) as i32,
        (transform.translation.z/float_chunk_width) as i32,];
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
        if !game_state.chunks.keys().any(|i| i==pos) {
            game_state.chunks.insert(*pos, ChunkData {
                entity : None, 
                collider : None, 
                state : chunk::ChunkState::NoGen,
                blocks : chunk::Chunk::new(*pos), 
                //generator : worldgen::ChunkGenerator::new(*pos, seed, 0)});
                generator : worldgen::ChunkGenerator::new_dont_build(*pos, seed, 3)});
        }
    }
}

fn chunk_generate_system( mut commands: Commands,
    mut game_state: ResMut<Game>,
    thread_pool: Res<AsyncComputeTaskPool>,
    mut transform_query: Query<
    &mut Transform,
    (With<BodyTag>, With<FakeKinematicRigidBody>),>,
) {
    // Builds chunks
    let float_chunk_width = CHUNK_WIDTH as f32;
    let transform = transform_query.single_mut().expect("THERE CAN ONLY BE ONE");
    let p_pos : [i32;3] = [(transform.translation.x/float_chunk_width) as i32,
        (transform.translation.y/float_chunk_width) as i32,
        (transform.translation.z/float_chunk_width) as i32,];
    let mut chunk_pos : Vec<[i32;3]> = vec![];
    for (x, y, z) in iproduct!(p_pos[0]-RENDER_DISTANCE..p_pos[0]+RENDER_DISTANCE, 
        p_pos[1]-RENDER_DISTANCE_VERTICAL..p_pos[1]+RENDER_DISTANCE_VERTICAL, 
        p_pos[2]-RENDER_DISTANCE..p_pos[2]+RENDER_DISTANCE) {
            chunk_pos.push([x, y, z]);
    }
    chunk_pos.sort_by(|a, b| 
        ((a[0]-p_pos[0]).abs() + (a[1]-p_pos[0]).abs() + (a[2]-p_pos[2]).abs())
        .partial_cmp(&((b[0]-p_pos[0]).abs() + (b[1]-p_pos[0]).abs() + (b[2]-p_pos[2]).abs())).unwrap());

    for pos in chunk_pos {
        let seed = game_state.seed.clone();
        if game_state.chunks.keys().any(|&x| x == pos) {
            let chunk = match game_state.chunks.get_mut(&pos) {
                Some(val) => val,
                None => {continue;}
            };
            if chunk.state == chunk::ChunkState::NoGen {
                let mut gen_clone = chunk.generator.clone();
                let gen_task = thread_pool.spawn(async move {
                    gen_clone.build_maps();
                    if let Some(new_gen) = gen_clone.generate(pos, seed) {
                        (gen_clone.generateStructs(pos, seed, new_gen), pos)
                    } else {
                        (gen_clone.generateStructs(pos, seed, gen_clone.blankBlocks()), pos)
                    }
                });
                commands.spawn().insert(gen_task);
            } 
        }
    }
}

fn handle_tasks(
    mut commands: Commands,
    mut transform_tasks: Query<(Entity, &mut Task<(Option<BlockArray>, [i32;3])>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut game_state: ResMut<Game>,
    array_texture_pipelines: Res<ArrayTexturePipelines>,
    array_texture_material: Res<ArrayTextureMaterial>,
) {
    //let texture = textures.get_mut(texture_handle.clone()).unwrap();
    //let pipeline = RenderPipelines::from_pipelines(vec![RenderPipeline::new(game_state.pipeline_handle.clone())]);
    // Set the texture to tile over the entire quad
    /*texture.sampler = SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        ..Default::default()
    };*/

    //texture.reinterpret_stacked_2d_as_array(chunk::TEXIMG_WIDTH as u32);
    let timer = Instant::now();
    for (entity, mut task) in transform_tasks.iter_mut() {
        if let Some((blocks, chunk_pos)) = future::block_on(future::poll_once(&mut *task)) {
            //Find chunk if it still exists
            if game_state.chunks.keys().any(|&i| i == chunk_pos) {
                let chunk = match game_state.chunks.get_mut(&chunk_pos) {
                    Some(val) => val,
                    None => {continue;}
                };
                if let Some(b) = blocks {
                    chunk.blocks.set_blocks(b);
                }
                // Add the mesh and collider to our tagged entity
                if let Some((new_mesh, new_collider)) = chunk.blocks.render_new() {
                    //let texture_handle = asset_server.load("atlas.png");
                    /*let chunk_id = commands.entity(entity).insert_bundle(PbrBundle {
                        mesh: meshes.add(new_mesh),
                        material: material_handle,
                        ..Default::default()
                    })*/
                    let chunk_id = commands.entity(entity).insert_bundle(PbrBundle {
                        mesh: meshes.add(new_mesh),
                        render_pipelines: array_texture_pipelines.0.clone(),
                        material: array_texture_material.0.clone(),
                        ..Default::default()
                    })
                    .insert(obb::Obb::default())
                    .insert(RayCastMesh::<MyRaycastSet>::default()).id();
                    let chunk_collider_id = commands.spawn_bundle(new_collider).id();
                    chunk.entity = Some(chunk_id);
                    chunk.collider = Some(chunk_collider_id);
                } else {
                    chunk.entity = None;
                    chunk.collider = None;
                }
                chunk.state = chunk::ChunkState::Rendered;

                // Task is complete, so remove task component from entity
                commands.entity(entity).remove::<Task<(Option<BlockArray>, [i32;3])>>();
            }
        }
    }
}

fn chunk_despawn_system(
    mut game_state: ResMut<Game>,
    mut commands: Commands,
    thread_pool: Res<AsyncComputeTaskPool>,
    mut transform_query: Query<
        &mut Transform,
        (With<BodyTag>, With<FakeKinematicRigidBody>),
    >,
) {
    let timer = Instant::now();
    // Check for chunks to remove
    let transform = transform_query.single_mut().expect("THERE CAN ONLY BE ONE");
    // generate possible chunk positions
    let float_chunk_width = CHUNK_WIDTH as f32;
    let p_pos : [i32;3] = [(transform.translation.x/float_chunk_width) as i32,
        (transform.translation.y/float_chunk_width) as i32,
        (transform.translation.z/float_chunk_width) as i32,];
    let mut chunk_pos : Vec<[i32;3]> = vec![];
    for (x, y, z) in iproduct!(p_pos[0]-RENDER_DISTANCE..p_pos[0]+RENDER_DISTANCE, 
        p_pos[1]-RENDER_DISTANCE_VERTICAL..p_pos[1]+RENDER_DISTANCE_VERTICAL, 
        p_pos[2]-RENDER_DISTANCE..p_pos[2]+RENDER_DISTANCE) {
            chunk_pos.push([x, y, z]);
    }
    chunk_pos.sort_by(|a, b| 
        ((a[0]-p_pos[0]).abs() + (a[1]-p_pos[0]).abs() + (a[2]-p_pos[2]).abs())
        .partial_cmp(&((b[0]-p_pos[0]).abs() + (b[1]-p_pos[0]).abs() + (b[2]-p_pos[2]).abs())).unwrap());
    let mut chunks_to_remove = vec![];
    let current_chunk_pos = game_state.chunks.keys().clone();
    for pos in current_chunk_pos {
        if !chunk_pos.iter().any(|i| i==pos) {
            if let Some(chunk_id) = game_state.chunks[pos].entity {
                commands.entity(chunk_id).despawn();
            };
            if let Some(chunk_collider_id) = game_state.chunks[pos].collider {
                commands.entity(chunk_collider_id).despawn();
            };
            chunks_to_remove.push(*pos);
        }
    }
    for pos in chunks_to_remove {
        game_state.chunks.remove(&pos);
    }
}

fn updateChunks(
    mut game_state: ResMut<Game>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_collider_query : Query<&mut ColliderShape>,
    mut chunk_mesh_query : Query<&mut Handle<Mesh>>,
    mut commands: Commands,
    //mut materials: ResMut<Assets<StandardMaterial>>,
    //asset_server: Res<AssetServer>,
    array_texture_pipelines: Res<ArrayTexturePipelines>,
    array_texture_material: Res<ArrayTextureMaterial>,
) {
    let block_change_queue = game_state.block_change_queue.clone();
    for (chunk_pos, block_pos, id) in block_change_queue.iter() {
        //let idx = game_state.current_chunks.iter().position(|x| x == chunk_pos).unwrap();
        //let chunk = &mut game_state.chunks[idx];
        let chunk = match game_state.chunks.get_mut(chunk_pos) {
            Some(val) => val,
            None => {continue;}
        };
        chunk.blocks.setBlock(*block_pos, *id);
        //let mut new_chunk_collider : Option<Entity> = None;
        if let Some(chunk_id) = chunk.entity {
            if let Some((new_mesh, new_collider)) = chunk.blocks.render_new() {
                let mut chunk_mesh = chunk_mesh_query.get_mut(chunk_id).unwrap();
                *chunk_mesh = meshes.add(new_mesh);
                if let Some(chunk_collider_id) = chunk.collider {
                    let mut chunk_collider = chunk_collider_query.get_mut(chunk_collider_id).unwrap();
                    *chunk_collider = new_collider.shape;
                } else {
                    let chunk_collider_id = commands.spawn_bundle(new_collider).id();
                    chunk.collider = Some(chunk_collider_id);
                }
            }
            
        } else  {
            if let Some((new_mesh, new_collider)) = chunk.blocks.render_new() {
                let chunk_id = commands.spawn_bundle(PbrBundle {
                    mesh: meshes.add(new_mesh),
                    render_pipelines: array_texture_pipelines.0.clone(),
                    material: array_texture_material.0.clone(),
                    ..Default::default()
                })
                .insert(obb::Obb::default())
                .insert(RayCastMesh::<MyRaycastSet>::default()).id();
                let chunk_collider_id = commands.spawn_bundle(new_collider).id();
                chunk.entity = Some(chunk_id);
                chunk.collider = Some(chunk_collider_id);
            }
        }
    }
    game_state.block_change_queue.clear();
}

fn handle_mouse(
    mut query: Query<&mut RayCastSource<MyRaycastSet>>,
    mut blockhighlight: Query<(&mut Transform, &mut Visible), With<BlockHighlight>>,
    buttons: Res<Input<MouseButton>>,
    mut game_state: ResMut<Game>,
) {
    // Handles block highlighting and mouse button presses
    let float_chunk_width = CHUNK_WIDTH as f32;
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
            highlight_pos.translation = Vec3::new(look_block[0] as f32 + 0.5, look_block[1] as f32 + 0.5, look_block[2] as f32 + 0.5);
            if buttons.just_pressed(MouseButton::Right) && distance > 5.0 {
                let new_block_pos = [(look_block[0] as f32 + normal.x.floor()) as i32, 
                    (look_block[1] as f32 + normal.y.floor()) as i32, 
                    (look_block[2] as f32 + normal.z.floor()) as i32];
                let mouse_chunk = [((new_block_pos[0] as f32 - 1.0) / float_chunk_width).floor() as i32, 
                    ((new_block_pos[1] as f32 - 1.0) / float_chunk_width).floor() as i32,
                    ((new_block_pos[2] as f32 - 1.0) / float_chunk_width).floor() as i32];
                let rel_block_pos = [(new_block_pos[0] - (CHUNK_WIDTH * mouse_chunk[0])),(new_block_pos[1] - (CHUNK_WIDTH * mouse_chunk[1])),(new_block_pos[2] - (CHUNK_WIDTH * mouse_chunk[2]))];
                game_state.block_change_queue.push((mouse_chunk, rel_block_pos, 1));
                // find chunk with correct position
                if rel_block_pos[0] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0] - 1, mouse_chunk[1], mouse_chunk[2]],[CHUNK_WIDTH+1, rel_block_pos[1], rel_block_pos[2]],1));
                } else if rel_block_pos[0] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0] + 1, mouse_chunk[1], mouse_chunk[2]],[0, rel_block_pos[1], rel_block_pos[2]],1));
                }
                if rel_block_pos[1] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1]-1, mouse_chunk[2]],[rel_block_pos[0],CHUNK_WIDTH+1, rel_block_pos[2]],1));
                } else if rel_block_pos[1] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1]+1, mouse_chunk[2]],[rel_block_pos[0],0, rel_block_pos[2]],1));
                }
                if rel_block_pos[2] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]-1],[rel_block_pos[0], rel_block_pos[1], CHUNK_WIDTH+1],1));
                } else if rel_block_pos[2] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0] - 1, mouse_chunk[1], mouse_chunk[2]],[rel_block_pos[0], rel_block_pos[1], 0],1));
                }
            } else if buttons.just_pressed(MouseButton::Left) {
                // add block
                let mouse_chunk = [((look_block[0] as f32 - 1.0) / float_chunk_width).floor() as i32, 
                    ((look_block[1] as f32 - 1.0) / float_chunk_width).floor() as i32,
                    ((look_block[2] as f32 - 1.0) / float_chunk_width).floor() as i32];
                let rel_block_pos = [(look_block[0] - (CHUNK_WIDTH * mouse_chunk[0])),(look_block[1] - (CHUNK_WIDTH * mouse_chunk[1])),(look_block[2] - (CHUNK_WIDTH * mouse_chunk[2]))];
                game_state.block_change_queue.push((mouse_chunk, rel_block_pos, 0));
                // update surrounding chunks if necessary
                if rel_block_pos[0] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0] - 1, mouse_chunk[1], mouse_chunk[2]],[CHUNK_WIDTH+1, rel_block_pos[1], rel_block_pos[2]],0));
                } else if rel_block_pos[0] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0] + 1, mouse_chunk[1], mouse_chunk[2]],[0, rel_block_pos[1], rel_block_pos[2]],0));
                }
                if rel_block_pos[1] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1]-1, mouse_chunk[2]],[rel_block_pos[0],CHUNK_WIDTH+1, rel_block_pos[2]],0));
                } else if rel_block_pos[1] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1]+1, mouse_chunk[2]],[rel_block_pos[0],0, rel_block_pos[2]],0));
                }
                if rel_block_pos[2] == 1 {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]-1],[rel_block_pos[0], rel_block_pos[1], CHUNK_WIDTH+1],0));
                } else if rel_block_pos[2] == CHUNK_WIDTH {
                    game_state.block_change_queue.push(([mouse_chunk[0], mouse_chunk[1], mouse_chunk[2]+1],[rel_block_pos[0], rel_block_pos[1], 0],0));
                }
            }
        }
    } else {
        //hide block highlight
        highlight_vis.is_visible = false;
    }
}

