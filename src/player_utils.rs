use bevy::{input::system::exit_on_esc_system, prelude::*, diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},};
use bevy_prototype_character_controller::{
    controller::{
        BodyTag, CameraTag, CharacterController, CharacterControllerPlugin, HeadTag, Mass, YawTag,
    },
    events::TranslationEvent,
    look::{LookDirection, LookEntity},
    
};
use bevy_rapier3d::{
    prelude::{ QueryPipeline, QueryPipelineColliderComponentsQuery},
};
use bevy_rapier3d::{prelude::*};
use bevy_mod_raycast::{RayCastSource, build_rays, update_raycast, RaycastSystem, PluginState};
pub struct BlockHighlight;

pub struct CharacterSettings {
    pub scale: Vec3,
    pub head_scale: f32,
    pub head_yaw: f32,
    pub follow_offset: Vec3,
    pub focal_point: Vec3,
}

pub struct MyRaycastSet;

impl Default for CharacterSettings {
    fn default() -> Self {
        Self {
            scale: Vec3::new(0.5, 1.7, 0.3),
            head_scale: 0.3,
            head_yaw: 0.0,
            follow_offset: Vec3::new(0.0, 4.0, 8.0), // Relative to head
            focal_point: Vec3::ZERO,                 // Relative to head
        }
    }
}

pub struct FakeKinematicRigidBody;

pub fn build_app(app: &mut AppBuilder) {
    app.add_plugins(DefaultPlugins)
        .add_plugin(CharacterControllerPlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .init_resource::<PluginState<MyRaycastSet>>()
        .add_system_to_stage(
            CoreStage::PostUpdate,
            build_rays::<MyRaycastSet>.system().label(RaycastSystem::BuildRays),
        )
        .add_system_to_stage(
            CoreStage::PostUpdate,
            update_raycast::<MyRaycastSet>
                .system()
                .label(RaycastSystem::UpdateRaycast)
                .after(RaycastSystem::BuildRays),
        )
        .add_system(exit_on_esc_system.system())
        
        //.add_startup_system(spawn_world.system())
        .add_startup_system(spawn_character.system());
}

pub fn spawn_character(
    mut commands: Commands,
    character_settings: Res<CharacterSettings>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let cube = meshes.add(Mesh::from(bevy::prelude::shape::Cube { size: 1.0 }));
    let red = materials.add(Color::hex("800000").unwrap().into());

    let body = commands
        .spawn_bundle((
            GlobalTransform::identity(),
            Transform::from_translation(Vec3::new(0.5,64.0,0.5)),
            CharacterController{
                walk_speed : 7.0,
                run_speed : 10.0,
                jumping : true,
                ..Default::default()
            },
            FakeKinematicRigidBody,
            Mass::new(80.0),
            BodyTag,
        ))
        .id();
    let yaw = commands
        .spawn_bundle((GlobalTransform::identity(), Transform::identity(), YawTag))
        .id();
    let body_model = commands
        .spawn_bundle(PbrBundle {
            material: red.clone(),
            mesh: cube.clone(),
            transform: Transform::from_matrix(Mat4::from_scale_rotation_translation(
                character_settings.scale - character_settings.head_scale * Vec3::Y,
                Quat::IDENTITY,
                Vec3::new(0.0, character_settings.head_scale, 0.0),
            )),
            ..Default::default()
        })
        .id();
    let head = commands
        .spawn_bundle((
            GlobalTransform::identity(),
            Transform::from_matrix(Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                Quat::from_rotation_y(character_settings.head_yaw),
                (0.5 * character_settings.scale.y + character_settings.head_scale) * Vec3::Y,
            )),
            HeadTag,
        ))
        .id();
    let head_model = commands
        .spawn_bundle(PbrBundle {
            material: red,
            mesh: cube,
            transform: Transform::from_scale(Vec3::splat(character_settings.head_scale)),
            ..Default::default()
        })
        .id();
    let mut camera_transform = Transform::from_matrix(Mat4::face_toward(
        character_settings.follow_offset,
        character_settings.focal_point,
        Vec3::Y,
    ));
    camera_transform.scale = Vec3::new(0.3,0.3,0.3);
    let camera = commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: camera_transform,
            ..Default::default()
        })
        .insert_bundle((LookDirection::default(), CameraTag))
        .insert(RayCastSource::<MyRaycastSet>::new_transform_empty())
        //.insert_bundle(PickingCameraBundle::default())
        .id();
    commands
        .entity(body)
        .insert(LookEntity(camera))
        .push_children(&[yaw]);
    commands.entity(yaw).push_children(&[body_model, head]);
    commands.entity(head).push_children(&[head_model, camera]);
}

pub fn controller_to_kinematic(
    mut translations: EventReader<TranslationEvent>,
    query_pipeline: Res<QueryPipeline>, 
    collider_query: QueryPipelineColliderComponentsQuery,
    mut query: Query<
        (&mut Transform, &mut CharacterController),
        (With<BodyTag>, With<FakeKinematicRigidBody>),
    >,
) {
    // Handle character physics
    let solid = true;
    let groups = InteractionGroups::all();
    let filter = None;
    for (mut transform, mut controller) in query.iter_mut() {
        let collider_set = QueryPipelineColliderComponentsSet(&collider_query);
        for translation in translations.iter() {
            let xz_transform = Vec3::new((**translation).x, 0.0, (**translation).z);
            let y_transform = Vec3::new(0.0, (**translation).y, 0.0);

            // Check X and Z direcitons first
            let feet_ray = Ray::new((transform.translation).into(), xz_transform.normalize().into());
            let head_ray = Ray::new((transform.translation + (Vec3::Y*1.1)).into(), xz_transform.normalize().into());
            
            // Only move if the rays dont hit
            if let None = query_pipeline.cast_ray(
                &collider_set, &head_ray, 0.5, solid, groups, filter) {
                if let None = query_pipeline.cast_ray(
                    &collider_set, &feet_ray, 0.5, true, InteractionGroups::all(), None) {
                    transform.translation += xz_transform;
                } else {
                    println!("foot ray hit");
                    controller.velocity.x = 0.0;
                    controller.velocity.z = 0.0;
                }
            } else {
                println!("head ray hit");
                controller.velocity.x = 0.0;
                controller.velocity.z = 0.0;
            }

            let gravity_ray = Ray::new((transform.translation).into(), (-Vec3::Y).into());
            
            if let Some((_,toi)) = query_pipeline.cast_ray(
                &collider_set, &gravity_ray, 50.0, solid, groups, filter) {
                    let floor_pos = gravity_ray.point_at(toi);
                    if floor_pos.y + 0.5 < transform.translation.y {
                        // fall as normal
                        controller.jumping = true;
                        transform.translation += y_transform;
                    } else {
                        if y_transform.y > 0.0 {
                            // jump
                            transform.translation += y_transform;
                            controller.jumping = true;
                        } else if y_transform.y < 0.0 {
                            // stop falling
                            controller.jumping = false;
                            transform.translation.y = floor_pos.y + 0.5;
                        }
                    }
            } else {
                controller.jumping = true;
                transform.translation += y_transform;
            }
        }
    }
}

