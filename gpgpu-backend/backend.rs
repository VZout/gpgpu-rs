use lazy_static::lazy_static;
use std::sync::Mutex;

pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Context {
    fn create_device_queue() -> (wgpu::Device, wgpu::Queue) {
        async fn create_device_queue_async() -> (wgpu::Device, wgpu::Queue) {
            let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: None,
                })
                .await
                .expect("Failed to find an appropriate adapter");

            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
                        shader_validation: true,
                    },
                    None,
                )
                .await
                .expect("Failed to create device")
        }
        futures::executor::block_on(create_device_queue_async())
    }

    fn new() -> Self {
        let (device, queue) = Context::create_device_queue();

        Self { device, queue }
    }

    pub fn launch(&mut self, entry_point: &str) {
        let spriv_file = std::fs::read(std::env::var("GPGPU_EXAMPLE_GPU.SPV").unwrap()).unwrap();

        let module = self
            .device
            .create_shader_module(wgpu::util::make_spirv(&spriv_file[..]));

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &module,
                        entry_point,
                    },
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            cpass.dispatch(1, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        println!("GPU Finished '{}'", entry_point);
    }
}

#[derive(Debug)]
pub struct SpvFile {
    pub name: String,
    pub data: Vec<u32>,
}

pub fn init() {
    //#[cfg(debug_assertions)]
    env_logger::init();
}

lazy_static! {
    pub static ref CONTEXT: Mutex<Context> = Mutex::new(Context::new());
}
