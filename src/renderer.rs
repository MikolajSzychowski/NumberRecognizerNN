use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::neural::{self, NeuralData};

const SCREEN_HEIGHT: usize = 770;
const SCREEN_WIDTH: usize = 1500;

pub fn init(grid: SharedGrid, neural_data: NeuralData, update: impl FnMut() + 'static) {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App {
        update: Some(Box::new(update)),
        grid: Some(grid),
        neural_data: Some(neural_data),
        ..Default::default()
    };
    event_loop.run_app(&mut app).unwrap();
}

pub const GRID_SIZE: usize = 28;
pub const CELL_PX: f32 = 40.0;
pub const GRID_ORIGIN: (f32, f32) = (20.0, 20.0);

pub type SharedGrid = Rc<RefCell<Vec<f32>>>;

pub fn new_shared_grid() -> SharedGrid {
    Rc::new(RefCell::new(vec![0.0; GRID_SIZE * GRID_SIZE]))
}

fn build_grid_vertices(grid: &[f32]) -> Vec<Vertex> {
    let mut verts = Vec::with_capacity(GRID_SIZE * GRID_SIZE * 6);
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let value = grid[row * GRID_SIZE + col];
            let color = [value, value, value]; // grayscale; swap for a colormap if you like

            let x0 = GRID_ORIGIN.0 + col as f32 * CELL_PX;
            let y0 = GRID_ORIGIN.1 + row as f32 * CELL_PX;
            let x1 = x0 + CELL_PX;
            let y1 = y0 + CELL_PX;

            let tl = Vertex {
                position: [x0, y0],
                color,
            };
            let tr = Vertex {
                position: [x1, y0],
                color,
            };
            let br = Vertex {
                position: [x1, y1],
                color,
            };
            let bl = Vertex {
                position: [x0, y1],
                color,
            };

            verts.push(tl);
            verts.push(tr);
            verts.push(br);
            verts.push(tl);
            verts.push(br);
            verts.push(bl);
        }
    }
    verts
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

const SHADER_SRC: &str = r#"
struct Uniforms {
    window_size: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
    let ndc_x = (position.x / uniforms.window_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (position.y / uniforms.window_size.y) * 2.0;
    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    vertex_buffer: Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    update: Option<Box<dyn FnMut()>>,
    vertex_count: u32,
    grid: Option<SharedGrid>,
    mouse_pos: (f32, f32),
    mouse_down: bool,
    neural_data: Option<NeuralData>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    window_size: [f32; 2],
    _padding: [f32; 2], // pad to 16 bytes
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes().with_title("Number Recognizer").with_inner_size(winit::dpi::LogicalSize::new(SCREEN_WIDTH as u32, SCREEN_HEIGHT as u32));
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.window = Some(window.clone());

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let caps = surface.get_capabilities(&adapter);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: SCREEN_WIDTH as u32,
            height: SCREEN_HEIGHT as u32,
            present_mode: wgpu::PresentMode::Fifo, // V-Sync
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &config);

        let initial_verts = build_grid_vertices(&self.grid.as_ref().unwrap().borrow());
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&initial_verts),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Square Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniforms = Uniforms {
            window_size: [config.width as f32, config.height as f32],
            _padding: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Square Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.render_pipeline = Some(render_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        self.vertex_count = (GRID_SIZE * GRID_SIZE * 6) as u32;

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(physical_size) => {
                if let (Some(surface), Some(device), Some(queue), Some(config), Some(uniform_buffer)) = (&self.surface, &self.device, &self.queue, &mut self.config, &self.uniform_buffer) {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(device, config);

                        let uniforms = Uniforms {
                            window_size: [config.width as f32, config.height as f32],
                            _padding: [0.0, 0.0],
                        };
                        queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
                    }
                }
            }
            WindowEvent::CursorMoved {
                position,
                ..
            } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
                if self.mouse_down {
                    self.paint_at_mouse();
                }
            }
            WindowEvent::MouseInput {
                state,
                button,
                ..
            } => {
                if button == winit::event::MouseButton::Left {
                    self.mouse_down = state == winit::event::ElementState::Pressed;
                    if self.mouse_down {
                        self.paint_at_mouse();
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(update) = &mut self.update {
                    update();
                }

                if let (Some(queue), Some(grid), Some(vbuffer)) = (&self.queue, &self.grid, &self.vertex_buffer) {
                    let verts = build_grid_vertices(&grid.borrow());
                    queue.write_buffer(vbuffer, 0, bytemuck::cast_slice(&verts));
                }

                if let (Some(surface), Some(device), Some(queue), Some(config), Some(pipeline), Some(vbuffer), Some(bind_group)) = (&self.surface, &self.device, &self.queue, &self.config, &self.render_pipeline, &self.vertex_buffer, &self.bind_group) {
                    let frame = match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(f) => f,
                        wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                            surface.configure(device, config);
                            f
                        }
                        wgpu::CurrentSurfaceTexture::Outdated => {
                            surface.configure(device, config);
                            return;
                        }
                        _ => return,
                    };
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: None,
                    });

                    {
                        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.1,
                                        g: 0.1,
                                        b: 0.1,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });

                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(0, bind_group, &[]);
                        render_pass.set_vertex_buffer(0, vbuffer.slice(..));
                        render_pass.draw(0..self.vertex_count, 0..1);
                    }

                    queue.submit(std::iter::once(encoder.finish()));
                    queue.present(frame);

                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => (),
        }
    }
}

impl App {
    fn paint_at_mouse(&mut self) {
        let (mx, my) = self.mouse_pos;
        let col = ((mx - GRID_ORIGIN.0) / CELL_PX).floor() as i32;
        let row = ((my - GRID_ORIGIN.1) / CELL_PX).floor() as i32;
        if row >= 0 && row < GRID_SIZE as i32 && col >= 0 && col < GRID_SIZE as i32 {
            if let (Some(grid), Some(neural_data)) = (&self.grid, &mut self.neural_data) {
                grid.borrow_mut()[row as usize * GRID_SIZE + col as usize] = 1.0;

                let scaled_pixels = Array1::from(grid.borrow().clone()) * 255.0;
                neural_data.calculate_network(&scaled_pixels);
            }
        }
    }
}
