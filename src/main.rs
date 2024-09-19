use pollster::FutureExt;

fn main() {
    env_logger::init();

    let instance = wgpu::Instance::new(Default::default());
    let adapter = instance
        .request_adapter(&Default::default())
        .block_on()
        .unwrap();

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .block_on()
        .unwrap();

    let cols = 2;
    let num_matrices = 4; // NOTE: coordinate with below shader

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "\
        alias Mat = mat{cols}x3<f32>;
        alias Type = Mat;
        @group(0) @binding(0) var<storage, read_write> buffer : array<Type, 4>;

        @compute @workgroup_size(1)
        fn main() {{
          var m : Mat;
          for (var c = 0u; c < {cols}; c++) {{
            m[c] = vec3(f32(c*3 + 1), f32(c*3 + 2), f32(c*3 + 3));
          }}
          buffer = array<Type, 4>(Type(m), Type(m * 2), Type(m * 3), Type(m * 4));
        }}
        "
            )
            .into(),
        ),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    let row_dimension = 3 /* rows */ + 1 /* padding */;
    let num_mat_elms = cols * row_dimension;
    let num_3d_mat_elms = num_mat_elms * num_matrices;
    let mut expected = vec![-1f32; num_3d_mat_elms];
    let mut output = vec![0f32; num_3d_mat_elms];
    let deadbeef = f32::from_bits(0xdeadbeef);
    for mat_idx in 0..num_matrices {
        for col_idx in 0..cols {
            let placeholder =
                |offset| u32::try_from((col_idx * 3 + offset) * (mat_idx + 1)).unwrap() as f32;
            let mat_offset = mat_idx * cols * row_dimension + col_idx * row_dimension;
            expected[mat_offset + 0] = placeholder(1);
            expected[mat_offset + 1] = placeholder(2);
            expected[mat_offset + 2] = placeholder(3);
            expected[mat_offset + 3] = deadbeef;
        }
    }
    let buf_size = expected.len() * size_of_val(&expected[0]);

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input"),
        size: u64::try_from(buf_size).unwrap(),
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let input = vec![deadbeef; expected.len()];
        let buffer_slice = input_buffer.slice(..);
        buffer_slice.get_mapped_range_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(input.as_ptr().cast(), buf_size)
        });
        input_buffer.unmap();
    }
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: u64::try_from(buf_size).unwrap(),
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: u64::try_from(buf_size).unwrap(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut command_encoder = device.create_command_encoder(&Default::default());

    command_encoder.copy_buffer_to_buffer(
        &input_buffer,
        0,
        &storage_buffer,
        0,
        buf_size.try_into().unwrap(),
    );
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }
    command_encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &output_buffer,
        0,
        buf_size.try_into().unwrap(),
    );

    queue.submit([command_encoder.finish()]);

    {
        let buffer_slice = output_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |res| res.unwrap());
        device.poll(wgpu::Maintain::Wait);
        {
            let buffer_slice = buffer_slice.get_mapped_range();
            let backing_buf =
                unsafe { std::slice::from_raw_parts(buffer_slice.as_ptr().cast(), output.len()) };
            output.copy_from_slice(backing_buf);
        }
        output_buffer.unmap();
    }

    assert_eq!(output, expected);
}
