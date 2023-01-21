import { mustHave } from '../boilerplate'

async function main () {
  const adapter = mustHave(await navigator.gpu.requestAdapter())
  const device = await adapter.requestDevice()

  const multiplyPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `
struct MatrixParams {
  M : u32,
  N : u32,
  K : u32,
}
@binding(0) @group(0) var<uniform> params : MatrixParams;
        
@binding(1) @group(0) var<storage, read> aCoeffs : array<vec4<f32>>;
@binding(2) @group(0) var<storage, read> bCoeffs : array<vec4<f32>>;
@binding(3) @group(0) var<storage, read_write> cCoeffs : array<vec4<f32>>;

var<workgroup> aBlock : array<array<mat4x4<f32>, 8>, 8>;
var<workgroup> bBlock : array<array<mat4x4<f32>, 8>, 8>;

@compute @workgroup_size(8, 8, 1)
fn matrixMult(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) localId : vec3<u32>) {
  var M = params.M;
  var N = params.N;
  var K = params.K;
  var Q = 32u * groupId.xy + 4u * localId.xy;
  var ii = localId.x;
  var jj = localId.y;
  var out = mat4x4<f32>();
  for (var k = 0u; k < K; k = k + 32u) {
    for (var r = 0u; r < 4u; r = r + 1u) {
      aBlock[ii][jj][r] = aCoeffs[(K * (Q.x + r) + 4u * jj + k) >> 2];
      bBlock[ii][jj][r] = bCoeffs[(N * (4u * ii + k + r) + Q.y) >> 2];
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 8u; kk = kk + 1u) {
      out = out + transpose(aBlock[ii][kk]) * transpose(bBlock[kk][jj]);
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r = r + 1u) {
    cCoeffs[(M * (Q.x + r) + Q.y) >> 2] = out[r];
  }
}`
      }),
      entryPoint: 'matrixMult'
    }
  })

  const kernels = {
    async gpu(m:number, n:number, k:number) {
      const paramBuffer = device.createBuffer({
        size: 3 * 4,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true
      })
      new Uint32Array(paramBuffer.getMappedRange()).set([ m, n, k ])
      paramBuffer.unmap()
      const Abuffer = device.createBuffer({
        size: m * k * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      const Bbuffer = device.createBuffer({
        size: k * n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      const Cbuffer = device.createBuffer({
        size: m * n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      })
      const CReadCopy = device.createBuffer({
        size: m * n * 4, 
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      })

      const bindGroup = device.createBindGroup({
        layout: multiplyPipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: paramBuffer
            }
          },
          {
            binding: 1,
            resource: {
              buffer: Abuffer
            }
          },
          {
            binding: 2,
            resource: {
              buffer: Bbuffer,
            }
          },
          {
            binding: 3,
            resource: {
              buffer: Cbuffer
            }
          }
        ]
      })

      return {
        async mult (a:Float32Array, b:Float32Array, c:Float32Array) {
          device.queue.writeBuffer(Abuffer, 0, a.buffer, a.byteOffset, a.byteLength)
          device.queue.writeBuffer(Bbuffer, 0, b.buffer, b.byteOffset, b.byteLength)

          const commandEncoder = device.createCommandEncoder()
          const passEncoder = commandEncoder.beginComputePass()
          passEncoder.setPipeline(multiplyPipeline)
          passEncoder.setBindGroup(0, bindGroup)
          passEncoder.dispatchWorkgroups(n / 32, m / 32, 1)
          passEncoder.end()
          commandEncoder.copyBufferToBuffer(Cbuffer, 0, CReadCopy, 0, 4 * n * m)
          device.queue.submit([commandEncoder.finish()])

          await CReadCopy.mapAsync(GPUMapMode.READ);
          c.set(new Float32Array(CReadCopy.getMappedRange()))
          CReadCopy.unmap()
        },
        async free () {
          paramBuffer.destroy()
          Abuffer.destroy()
          Bbuffer.destroy()
          Cbuffer.destroy()
        }
      }
    },
    async cpu(m:number, n:number, k:number) {
      return {
        async mult (a:Float32Array, b:Float32Array, c:Float32Array) {
          for (let ii = 0; ii < m; ++ii) {
            for (let jj = 0; jj < n; ++jj) {
              let s = 0
              for (let kk = 0; kk < k; ++kk) {
                s += a[k * ii + kk] * b[n * kk + jj]
              }
              c[n * ii + jj] = s
            }
          }
        },
        async free () { 
          // placeholder
        }
      }
    },
  } as const

  const ui = createUIBoilerPlate(kernels)

  function randMatrix (m:number, n:number) {
    const A = new Float32Array(m * n)
    for (let i = 0; i < A.length; ++i) {
      A[i] = 2 * Math.random()  - 1
    }
    return A
  }

  while (true) {
    const {startN, endN, stepN, iter, kernel} = await ui.go()

    ui.clear()
    ui.log(`Benchmarking ${kernel} from n = ${startN} to ${endN} (step = ${stepN}) ${iter}/step....`)

    for (let n = startN; n <= endN; n += stepN) {
      const alg = await kernels[kernel](n, n, n)
      const A = randMatrix(n, n)
      const B = randMatrix(n, n)
      const C = new Float32Array(n * n)
      
      const tStart = performance.now()
      for (let i = 0; i < iter; ++i) {
        await alg.mult(A, B, C)
      }
      const tElapsed = performance.now() - tStart

      const work = iter * Math.pow(n, 3)
      ui.log(`${n}x${n}: ${work / tElapsed} FLOPs (${tElapsed} ms)`)
      await alg.free()

      await ui.sleep(16)
    }
  }
}

main().catch(err => console.error(err))

// boring ui code for completeness, do not pay attention to it....
function createUIBoilerPlate<KernelT extends {}> (kernels:KernelT) {
  const formContainer = document.createElement('div')
  Object.assign(formContainer.style, {
    padding: '32px',
    margin: '32px'
  })

  function uiInput (spec:{
    label:string,
    min:number,
    max:number,
    step:number,
    value:number
  }) {
    const container = document.createElement('div')
    Object.assign(container.style, {
      padding: '2px'
    })
    const input = document.createElement('input')
    const id = 'input-' + Math.random()
    input.id = id
    input.type = 'number'
    input.min = spec.min + ''
    input.max = spec.max + ''
    input.step = spec.step + ''
    input.value = spec.value + ''
    const label = document.createElement('label')
    label.htmlFor = id
    label.innerText = spec.label
    container.appendChild(label)
    container.appendChild(input)
    formContainer.appendChild(container)
    return {
      value: () => (Math.floor(+input.value / spec.step) * spec.step) >>> 0
    }
  }

  const textNode = document.createElement('div')
  textNode.innerText = `Single precision matrix multiply demo.
This measures the number of flops required to multiply two NxN matrices together across various values of N.
The matrix size (N) must be a multiple of 32.`
  formContainer.appendChild(textNode)

  const startN = uiInput({ label: 'Min N', min: 32, max: 2048, step: 32, value: 32 })
  const endN = uiInput({ label: 'Max N', min: 32, max: 2048, step: 32, value: 1024 })
  const stepN = uiInput({ label: 'Step N', min: 32, max: 2048, step: 32, value: 128 })
  const iterCount = uiInput({ label: 'Iterations per step', min: 1, max: 100, step: 1, value: 10 })

  const selectContainer = document.createElement('div')
  Object.assign(selectContainer.style, {
    padding: '1px'
  })
  const kernelSelect = document.createElement('select')
  for (const k of Object.keys(kernels)) {
    const opt = document.createElement('option')
    opt.value = opt.text = k
    kernelSelect.appendChild(opt)
  }
  const selectLabel = document.createElement('label')
  selectLabel.innerText = 'Kernel: '
  selectLabel.htmlFor = kernelSelect.id = 'kernel-select'
  selectContainer.appendChild(selectLabel)
  selectContainer.appendChild(kernelSelect)
  formContainer.appendChild(selectContainer)

  const goButton = document.createElement('input')
  goButton.type = 'button'
  goButton.value = 'Go!'
  formContainer.appendChild(goButton)

  const logPre = document.createElement('pre')
  formContainer.appendChild(logPre)

  document.body.appendChild(formContainer)
  
  return {
    sleep (ms:number) {
      return new Promise<void>((resolve) => {
        setTimeout(resolve, ms)
      })
    },

    clear() {
      logPre.innerText = ''
    },

    log (line:string) {
      logPre.innerText += line + '\n'
    },

    go () {
      return new Promise<{
        startN: number
        endN: number
        stepN: number
        iter: number,
        kernel: keyof KernelT,
      }>((resolve) => {
        function handler () {
          resolve({
            startN: startN.value(),
            endN: endN.value(),
            stepN: stepN.value(),
            iter: iterCount.value(),
            kernel: kernelSelect.value as any,
          })
        }
        goButton.addEventListener('click', handler)
      })
    }
  }
}