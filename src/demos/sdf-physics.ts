import { makeCanvas, mustHave } from '../boilerplate'
import { WebGPUScan } from '../lib/scan'
import { mat4, vec4, quat } from 'gl-matrix'

const PALETTE = [
  [ 0.19215686274509805, 0.2235294117647059, 0.23529411764705882, 1 ],
  [ 0.12941176470588237, 0.4627450980392157, 1, 1 ],
  [ 0.2, 0.6313725490196078, 0.9921568627450981, 1 ],
  [ 0.9921568627450981, 0.792156862745098, 0.25098039215686274, 1 ],
  [ 0.9686274509803922, 0.596078431372549, 0.1411764705882353, 1 ]
]

const SWEEP_RADIUS = 0.01
const DONUT_RADIUS = 0.07
const PARTICLE_RADIUS =  SWEEP_RADIUS + DONUT_RADIUS

const SCAN_THREADS = 256
const SCAN_ITEMS = 4
const PARTICLE_WORKGROUP_SIZE = 256
const NUM_PARTICLES = 256

// rendering performance parameters
const RAY_STEPS = 32
const RAY_TOLER = 0.001
const BG_RAY_STEPS = 256
const BG_RAY_TOLER = 0.0001
const BG_TMIN = 0.00
const BG_TMAX = 1000.0
const BG_COLOR = PALETTE[0]
const RADIUS_PADDING = 1.5

// physics simulation
const DT = 0.1
const SUBSTEPS = 10
const SUB_DT = DT / SUBSTEPS
const JACOBI_POS = 0.25
const JACOBI_ROT = 0.25
const GRAVITY = -1

const COMMON_SHADER_FUNCS = `
fn rigidMotion (q:vec4<f32>, v:vec4<f32>) -> mat4x4<f32> {
  var q2 = q.xyz + q.xyz;

  var xx = q.x * q2.x;
  var xy = q.x * q2.y;
  var xz = q.x * q2.z;
  var yy = q.y * q2.y;
  var yz = q.y * q2.z;
  var zz = q.z * q2.z;
  var wx = q.w * q2.x;
  var wy = q.w * q2.y;
  var wz = q.w * q2.z;

  return mat4x4<f32>(
    1. - (yy + zz),
    xy + wz,
    xz - wy,
    0.,

    xy - wz,
    1. - (xx + zz),
    yz + wx,
    0.,

    xz + wy,
    yz - wx,
    1. - (xx + yy),
    0.,

    v.x,
    v.y,
    v.z,
    1.
  );
}

fn upper3x3 (m:mat4x4<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(
    m[0][0], m[1][0], m[2][0],
    m[0][1], m[1][1], m[2][1],
    m[0][2], m[1][2], m[2][2]);
}

fn particleSDF (p : vec3<f32>) -> f32 {
  var q = vec2<f32>(length(p.xz)-${DONUT_RADIUS}, p.y);
  return length(q) - ${SWEEP_RADIUS};
}

fn terrainSDF (p : vec3<f32>) -> f32 {
  return max(p.y + 1., 3. - distance(p, vec3(0., -1, 0.)));
}
`


async function main () {
  const adapter = mustHave(await navigator.gpu.requestAdapter())
  const device = await adapter.requestDevice()
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

  const canvas = makeCanvas()
  const context = mustHave(canvas.getContext('webgpu'))
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque',
  })

  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  })

  const backroundShader = device.createShaderModule({
    label: 'bgRenderShader',
    code: `
${COMMON_SHADER_FUNCS}

struct Uniforms {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  projInv: mat4x4<f32>,
  fog : vec4<f32>,
  lightDir : vec4<f32>,
  eye : vec4<f32>
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) rayDirection : vec3<f32>,
}

@vertex
fn vertMain(
  @builtin(vertex_index) vertexIndex : u32
) -> VertexOutput {
  var corners = array<vec2<f32>, 4>(
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, -1.0),
  );

  var screenPos = corners[vertexIndex];
  
  var result : VertexOutput;
  result.clipPosition = vec4(screenPos, 0., 1.);

  var rayCamera = uniforms.projInv * vec4(screenPos, -1., 1.);
  result.rayDirection = upper3x3(uniforms.view) * rayCamera.xyz;

  return result;
}


fn map(p : vec3<f32>) -> f32 {
  return terrainSDF(p);
}

fn traceRay (rayOrigin : vec3<f32>, rayDir : vec3<f32>, tmin : f32, tmax: f32) -> f32 {
  var t = tmin;
  for (var i = 0u; i < ${BG_RAY_STEPS}u; i = i + 1u) {
    var pos = rayOrigin + t * rayDir;
    var h = map(pos);
    if t > tmax {
      return -1.;
    }
    if h < ${BG_RAY_TOLER} {
      return t;
    }
    t += h;
  }
  return -1.;
}

fn surfNormal (pos : vec3<f32>) -> vec3<f32> {
  var e = vec2<f32>(1.0,-1.0)*0.5773;
  const eps = 0.0005;
  return normalize( e.xyy*map( pos + e.xyy*eps ) + 
            e.yyx*map( pos + e.yyx*eps ) + 
            e.yxy*map( pos + e.yxy*eps ) + 
					  e.xxx*map( pos + e.xxx*eps ) );
}

struct FragmentOutput {
  @builtin(frag_depth) depth : f32,
  @location(0) color : vec4<f32>,
}

@fragment
fn fragMain (@location(0) rayDirectionInterp : vec3<f32>) -> FragmentOutput {
  var result : FragmentOutput;
  
  var rayDirection = normalize(rayDirectionInterp);
  var rayOrigin = uniforms.eye.xyz;
  var rayDist = traceRay(rayOrigin, rayDirection, ${BG_TMIN}, ${BG_TMAX});

  if rayDist < 0. {
    result.depth = 1.;
    result.color = vec4(${BG_COLOR.join(', ')});
    return result;
  }
  
  var rayHit = rayDist * rayDirection + rayOrigin;

  var clipPos = uniforms.proj * uniforms.view * vec4(rayHit, 1.);
  result.depth = clipPos.z / clipPos.w;
  
  var N = surfNormal(rayHit);
  
  var diffuse = max(0., -dot(N, uniforms.lightDir.xyz));
  var ambient = 0.5 + 0.5 * N.y;
  var light = ambient * (diffuse * vec3(0.9, 0.7, 0.6) + vec3(0.1, 0.3, 0.2));
  var color = mix(uniforms.fog.xyz, light * vec3(0.1, 0.2, 1.), exp(-0.01 * result.depth));

  result.color = vec4(sqrt(color), 1.);

  return result;
}
    `
  })

  const renderShader = device.createShaderModule({
    label: 'particleRenderShader',
    code: `
${COMMON_SHADER_FUNCS}

struct Uniforms {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  projInv: mat4x4<f32>,
  fog : vec4<f32>,
  lightDir : vec4<f32>,
  eye : vec4<f32>
}
@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(0) @group(1) var<storage, read> position : array<vec4<f32>>;
@binding(1) @group(1) var<storage, read> rotation : array<vec4<f32>>;
@binding(2) @group(1) var<storage, read> color : array<vec4<f32>>;

struct VertexOutput {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) particleColor : vec4<f32>,
  @location(1) rayDirection : vec3<f32>,
  @location(2) rayOrigin : vec3<f32>,
  @location(3) model0 : vec4<f32>,
  @location(4) model1 : vec4<f32>,
  @location(5) model2 : vec4<f32>,
  @location(6) model3 : vec4<f32>,
}

@vertex
fn vertMain(
  @builtin(instance_index) instanceIdx : u32,
  @location(0) uv : vec2<f32>,
) -> VertexOutput {
  var result : VertexOutput;
  result.particleColor = color[instanceIdx];

  var sdfMat = rigidMotion(
    rotation[instanceIdx],
    position[instanceIdx]);
  result.model0 = sdfMat[0];
  result.model1 = sdfMat[1];
  result.model2 = sdfMat[2];
  result.model3 = sdfMat[3];
  
  var viewCenter = uniforms.view * sdfMat[3];
  var rayDirection = viewCenter + vec4(${RADIUS_PADDING * PARTICLE_RADIUS} * uv.x, ${RADIUS_PADDING * PARTICLE_RADIUS} * uv.y, -${PARTICLE_RADIUS}, 0.);
  result.clipPosition = uniforms.proj * rayDirection;

  var invRot = upper3x3(sdfMat);
  var invTran = -sdfMat[3].xyz;
  result.rayDirection = invRot * upper3x3(uniforms.view) * rayDirection.xyz;
  result.rayOrigin = invRot * (uniforms.eye.xyz + invTran);
  
  return result;
}

fn map(p : vec3<f32>) -> f32 {
  return particleSDF(p);
}

fn traceRay (rayOrigin : vec3<f32>, rayDir : vec3<f32>, tmin : f32, tmax: f32) -> f32 {
  var t = tmin;
  for (var i = 0u; i < ${RAY_STEPS}u; i = i + 1u) {
    var pos = rayOrigin + t * rayDir;
    var h = map(pos);
    if t > tmax {
      return -1.;
    }
    if h < ${RAY_TOLER} {
      return t;
    }
    t += h;
  }
  return -1.;
}

fn surfNormal (pos : vec3<f32>) -> vec3<f32> {
  var e = vec2<f32>(1.0,-1.0)*0.5773;
  const eps = 0.0005;
  return normalize( e.xyy*map( pos + e.xyy*eps ) + 
            e.yyx*map( pos + e.yyx*eps ) + 
            e.yxy*map( pos + e.yxy*eps ) + 
					  e.xxx*map( pos + e.xxx*eps ) );
}

struct FragmentOutput {
  @builtin(frag_depth) depth : f32,
  @location(0) color : vec4<f32>,
}
    
@fragment
fn fragMain(
  @location(0) particleColor : vec4<f32>,
  @location(1) rayDirectionInterp : vec3<f32>,
  @location(2) rayOrigin : vec3<f32>,
  @location(3) model0 : vec4<f32>,
  @location(4) model1 : vec4<f32>,
  @location(5) model2 : vec4<f32>,
  @location(6) model3 : vec4<f32>,
) -> FragmentOutput {
  var result : FragmentOutput;
  
  var tmin = length(rayDirectionInterp);
  var rayDirection = rayDirectionInterp / tmin;
  var rayDist = traceRay(rayOrigin, rayDirection, 0.5 * tmin, tmin + ${2. * RADIUS_PADDING * PARTICLE_RADIUS});
  if rayDist < 0. {
    discard;
  }
  var rayHit = rayDist * rayDirection + rayOrigin;

  var model = mat4x4<f32>(model0, model1, model2, model3);
  
  var clipPos = uniforms.proj * uniforms.view * model * vec4(rayHit, 1.);
  result.depth = clipPos.z / clipPos.w;
  
  var N = normalize(surfNormal(rayHit) * upper3x3(model));

  var diffuse = max(0., -dot(N, uniforms.lightDir.xyz));
  var ambient = 0.5 + 0.5 * N.y;
  var light = ambient * (diffuse * vec3(0.9, 0.7, 0.6) + vec3(0.1, 0.3, 0.2));
  var color = mix(uniforms.fog.xyz, light * particleColor.xyz, exp(-0.01 * result.depth));

  result.color = vec4(sqrt(color), 1.);
  return result;
}`
  })

  const particlePredictShader = device.createShaderModule({
    label: 'particlePhysics',
    code: `
${COMMON_SHADER_FUNCS}

@binding(0) @group(0) var<storage, read> position : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read> velocity : array<vec4<f32>>;
@binding(2) @group(0) var<storage, read_write> predictedPosition : array<vec4<f32>>;
@binding(3) @group(0) var<storage, read_write> positionUpdate : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> rotation : array<vec4<f32>>;
@binding(5) @group(0) var<storage, read> angVelocity : array<vec4<f32>>;
@binding(6) @group(0) var<storage, read_write> predictedRotation : array<vec4<f32>>;
@binding(7) @group(0) var<storage, read_write> rotationUpdate : array<vec4<f32>>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn predictPositions (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;

  var v = velocity[id];
  v.y = v.y - ${GRAVITY * SUB_DT};
  predictedPosition[id] = position[id] + v * ${SUB_DT};
  positionUpdate[id] = vec4(0.);

  var q = rotation[id];
  var omega = angVelocity[id];
  var R = rigidMotion(q, vec4<f32>(0.));
  predictedRotation[id] = normalize(q + ${0.5 * SUB_DT} * (R * omega));
  rotationUpdate[id] = vec4(0.);
}`
  })

  const particleUpdateShader = device.createShaderModule({
    label: 'particleUpdate',
    code: `
${COMMON_SHADER_FUNCS}

@binding(0) @group(0) var<storage, read_write> position : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read_write> velocity : array<vec4<f32>>;
@binding(2) @group(0) var<storage, read> predictedPosition : array<vec4<f32>>;
@binding(3) @group(0) var<storage, read> positionUpdate : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read_write> rotation : array<vec4<f32>>;
@binding(5) @group(0) var<storage, read_write> angVelocity : array<vec4<f32>>;
@binding(6) @group(0) var<storage, read> predictedRotation : array<vec4<f32>>;
@binding(7) @group(0) var<storage, read> rotationUpdate : array<vec4<f32>>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn updatePositions (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;

  var p = position[id];
  var prevPos = p.xyz;
  var nextPosition = predictedPosition[id].xyz + ${JACOBI_POS} * positionUpdate[id].xyz;
  velocity[id] = vec4((nextPosition - prevPosition) * ${1 / SUB_DT}, 0.);
  position[id] = vec4(nextPosition, p.w);

  var prevQ = rotation[id];
  var nextQ = normalize(predictedRotation[id] + ${JACOBI_ROT} * rotationUpdate[id]);

  rotation[id] = nextQ;
}
`
  })

  const particlePositionBuffer = device.createBuffer({
    label: 'particlePosition',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const particleRotationBuffer = device.createBuffer({
    label: 'particleRotation',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const particleColorBuffer = device.createBuffer({
    label: 'particleColor',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  })
  const particleVelocityBuffer = device.createBuffer({
    label: 'particleVelocity',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE,
  })
  const particleAngularVelocityBuffer = device.createBuffer({
    label: 'particleAngularVelocity',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE
  })
  const particlePositionPredictionBuffer = device.createBuffer({
    label: 'particlePositionPrediction',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE
  })
  const particlePositionCorrectionBuffer = device.createBuffer({
    label: 'particlePositionCorrection',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE
  })
  const particleRotationPredictionBuffer = device.createBuffer({
    label: 'particleRotationPrediction',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE
  })
  const particleRotationCorrectionBuffer = device.createBuffer({
    label: 'particleRotationCorrection',
    size: 4 * 4 * NUM_PARTICLES,
    usage: GPUBufferUsage.STORAGE
  })
  const particlePositionData = new Float32Array(particlePositionBuffer.getMappedRange())
  const particleRotationData = new Float32Array(particleRotationBuffer.getMappedRange())
  const particleColorData = new Float32Array(particleColorBuffer.getMappedRange())
  for (let i = 0; i < NUM_PARTICLES; ++i) {
    const color = PALETTE[1 + (i % (PALETTE.length - 1))]
    for (let j = 0; j < 4; ++j) {
      particlePositionData[4 * i + j] = 2 * Math.random() - 1
      particleColorData[4 * i + j] = color[j]
      particleRotationData[4 *i + j] = Math.random() - 0.5
    }
    const q = particleRotationData.subarray(4 * i, 4 * (i + 1))
    quat.normalize(q, q)
  }
  particlePositionBuffer.unmap()
  particleRotationBuffer.unmap()
  particleColorBuffer.unmap()

  const spriteQuadUV = device.createBuffer({
    size: 2 * 4 * 4,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true
  })
  new Float32Array(spriteQuadUV.getMappedRange()).set([
    -1, -1,
    -1, 1,
    1, -1,
    1, 1,
  ])
  spriteQuadUV.unmap()

  const renderUniformBindGroupLayout = device.createBindGroupLayout({
    label: 'renderUniformBindGroupLayout',
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
      buffer: {
        type: 'uniform',
        hasDynamicOffset: false,
      }
    }]
  } as const)

  const renderParticleBindGroupLayout = device.createBindGroupLayout({
    label: 'particleRenderBufferBindGroup',
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: 'read-only-storage',
        hasDynamicOffset: false,
      }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: 'read-only-storage',
        hasDynamicOffset: false,
      }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.VERTEX,
      buffer: {
        type: 'read-only-storage',
        hasDynamicOffset: false,
      }
    }
    ]
  } as const)

  const renderUniformData = new Float32Array(1024)
  let uniformPtr = 0
  function nextUniform (size:number) {
    const result = renderUniformData.subarray(uniformPtr, uniformPtr + size)
    uniformPtr += size
    return result
  }

  const view = nextUniform(16)
  const projection = nextUniform(16)
  const projectionInv = nextUniform(16)
  const fog = nextUniform(4)
  const lightDir = nextUniform(4)
  const eye = nextUniform(4)

  const renderUniformBuffer = device.createBuffer({
    size: renderUniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })

  const renderUniformBindGroup = device.createBindGroup({
    label: 'uniformBindGroup',
    layout: renderUniformBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: renderUniformBuffer,
        }
      }
    ]
  })

  const renderParticleBindGroup = device.createBindGroup({
    label: 'renderParticleBindGroup',
    layout: renderParticleBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: particlePositionBuffer
        }
      },
      {
        binding: 1,
        resource: {
          buffer: particleRotationBuffer
        }
      },
      {
        binding: 2,
        resource: {
          buffer: particleColorBuffer
        }
      }
    ]
  })

  const renderParticlePipeline = device.createRenderPipeline({
    label: 'renderParticlePipeline',
    layout: device.createPipelineLayout({
      label: 'renderLayout',
      bindGroupLayouts: [
        renderUniformBindGroupLayout,
        renderParticleBindGroupLayout
      ]
    }),
    vertex: {
      module: renderShader,
      entryPoint: 'vertMain',
      buffers: [{
        arrayStride: 2 * 4,
        attributes: [{
          shaderLocation: 0,
          offset: 0,
          format: 'float32x2',
        }]
      }]
    },
    fragment: {
      module: renderShader,
      entryPoint: 'fragMain',
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus'
    }
  } as const)

  const renderBackgroundPipeline = device.createRenderPipeline({
    label: 'renderBackgroundPipeline',
    layout: device.createPipelineLayout({
      label: 'bgLayout',
      bindGroupLayouts: [
        renderUniformBindGroupLayout,
      ]
    }),
    vertex: {
      module: backroundShader,
      entryPoint: 'vertMain',
      buffers: [{
        arrayStride: 2 * 4,
        attributes: [{
          shaderLocation: 0,
          offset: 0,
          format: 'float32x2',
        }]
      }]
    },
    fragment: {
      module: backroundShader,
      entryPoint: 'fragMain',
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: 'triangle-strip',
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'always',
      format: 'depth24plus'
    }
  } as const)

  const predictPipeline = device.createComputePipeline({
    label: 'particlePredictPipeline',
    layout: 'auto',
    compute: {
      module: particlePredictShader,
      entryPoint: 'predictPositions'
    }
  })

  const predictBindGroup = device.createBindGroup({
    layout: predictPipeline.getBindGroupLayout(0),
    entries: [
      particlePositionBuffer,
      particleVelocityBuffer,
      particlePositionPredictionBuffer,
      particlePositionCorrectionBuffer,
      particleRotationBuffer,
      particleAngularVelocityBuffer,
      particleRotationPredictionBuffer,
      particleRotationCorrectionBuffer
    ].map((buffer, binding) => {
      return {
        binding,
        resource: { buffer }
      }
    })
  } as const)

  const updatePipeline = device.createComputePipeline({
    label: 'particleUpdatePipeline',
    layout: 'auto',
    compute: {
      module: particleUpdateShader,
      entryPoint: 'updatePositions'
    }
  })

  const updateBindGroup = device.createBindGroup({
    layout: updatePipeline.getBindGroupLayout(0),
    entries: [
      particlePositionBuffer,
      particleVelocityBuffer,
      particlePositionPredictionBuffer,
      particlePositionCorrectionBuffer,
      particleRotationBuffer,
      particleAngularVelocityBuffer,
      particleRotationPredictionBuffer,
      particleRotationCorrectionBuffer
    ].map((buffer, binding) => {
      return {
        binding,
        resource: { buffer }
      }
    })
  } as const)
    
  function frame (tick:number) {
    mat4.perspective(projection, Math.PI / 4, canvas.width / canvas.height, 0.01, 50)
    mat4.invert(projectionInv, projection)
    const theta = 0.0001 * tick
    vec4.set(eye, 8  * Math.cos(theta), 3, 8 * Math.sin(theta), 0)
    mat4.lookAt(view, eye, [0, -0.5, 0], [0, 1, 0])
    vec4.copy(fog, PALETTE[0] as vec4)
    vec4.set(lightDir, -1, -1, -0.2, 0)
    vec4.normalize(lightDir, lightDir)
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformData.buffer, 0, renderUniformData.byteLength)

    const commandEncoder = device.createCommandEncoder()

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'load',
          storeOp: 'store',
        }
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: 'load',
        depthStoreOp: 'store'
      }
    } as const,)

    passEncoder.setBindGroup(0, renderUniformBindGroup)
    passEncoder.setBindGroup(1, renderParticleBindGroup)
    passEncoder.setVertexBuffer(0, spriteQuadUV)
    
    passEncoder.setPipeline(renderBackgroundPipeline)
    passEncoder.draw(4)
    
    passEncoder.setPipeline(renderParticlePipeline);
    passEncoder.draw(4, NUM_PARTICLES)

    passEncoder.end()

    // const computePass = commandEncoder.beginComputePass()
    // for (let i = 0; i < SUBSTEPS; ++i) {
    //   computePass.setBindGroup(0, predictBindGroup)
    //   computePass.setPipeline(predictPipeline)
    //   computePass.dispatchWorkgroups(NUM_PARTICLES / PARTICLE_WORKGROUP_SIZE)

    //   // solve constraints

    //   computePass.setBindGroup(0, updateBindGroup)
    //   computePass.setPipeline(updatePipeline)
    //   computePass.dispatchWorkgroups(NUM_PARTICLES / PARTICLE_WORKGROUP_SIZE)
    // }
    // computePass.end()

    device.queue.submit([commandEncoder.finish()])
    requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

main().catch(err => console.error(err))