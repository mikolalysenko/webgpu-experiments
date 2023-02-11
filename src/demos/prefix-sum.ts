import { makeBench, mustHave } from '../boilerplate'

const THREADS_PER_GROUP_X = 256
const THREADS_PER_GROUP_Y = 1
const THREADS_PER_GROUP_Z = 1
const THREADS_PER_GROUP = THREADS_PER_GROUP_X * THREADS_PER_GROUP_Y * THREADS_PER_GROUP_Z
const ITEMS_PER_THREAD = 256
const ITEMS_PER_GROUP = ITEMS_PER_THREAD * THREADS_PER_GROUP

// used bank conflict resolution, does not seem to make any difference on rtx4090
const LOG_NUM_BANKS = 5

async function main () {
  const adapter = mustHave(await navigator.gpu.requestAdapter())
  const device = await adapter.requestDevice()

  const prefixSumShader = device.createShaderModule({
    code: `
struct SumParams {
  N : u32,
}
@binding(0) @group(0) var<uniform> params : SumParams;
@binding(1) @group(0) var<storage, read_write> post : array<f32>;
@binding(2) @group(0) var<storage, read_write> data : array<f32>;
@binding(3) @group(0) var<storage, read_write> work : array<f32>;

fn conflictFreeOffset (offset:u32) -> u32 {
  return offset + (offset >> ${LOG_NUM_BANKS});
}

var<workgroup> workerSums : array<f32, ${2 * THREADS_PER_GROUP}>;
fn partialSum (localId : u32) -> f32 {
  var offset = 1u;
  for (var d = ${THREADS_PER_GROUP >> 1}u; d > 0u; d = d >> 1u) {
    if (localId < d) {
      var ai = conflictFreeOffset(offset * (2u * localId + 1u) - 1u);
      var bi = conflictFreeOffset(offset * (2u * localId + 2u) - 1u);
      workerSums[bi] = workerSums[bi] + workerSums[ai];
    }
    offset *= 2u;
    workgroupBarrier();
  }

  if (localId == 0u) {
    workerSums[conflictFreeOffset(${THREADS_PER_GROUP - 1}u)] = 0.;
  }

  for (var d = 1u; d < ${THREADS_PER_GROUP}u; d = d * 2u) {
    offset = offset >> 1u;
    if (localId < d) {
      var ai = conflictFreeOffset(offset * (2u * localId + 1u) - 1u);
      var bi = conflictFreeOffset(offset * (2u * localId + 2u) - 1u);
      var a = workerSums[ai];
      var b = workerSums[bi];
      workerSums[ai] = b;
      workerSums[bi] = a + b;
    }
    workgroupBarrier();
  }

  return workerSums[conflictFreeOffset(localId)];
}

@compute @workgroup_size(${THREADS_PER_GROUP_X}, ${THREADS_PER_GROUP_Y}, ${THREADS_PER_GROUP_Z})
fn prefixSumIn(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) localVec : vec3<u32>,
  @builtin(global_invocation_id) globalVec : vec3<u32>) {
  var localId = ${THREADS_PER_GROUP_Y * THREADS_PER_GROUP_Z}u * localVec.x + ${THREADS_PER_GROUP_Z}u * localVec.y + localVec.z;
  var globalId = ${THREADS_PER_GROUP_Y * THREADS_PER_GROUP_Z}u * globalVec.x + ${THREADS_PER_GROUP_Z}u * globalVec.y + globalVec.z;

  var N = params.N;

  var s = 0.;
  var localVals = array<f32, ${ITEMS_PER_THREAD}>();
  for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
    s = s + data[${ITEMS_PER_THREAD} * globalId + i];
    localVals[i] = s;
  }
  workerSums[conflictFreeOffset(localId)] = s;
  workgroupBarrier();

  s = partialSum(localId);

  for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
    work[${ITEMS_PER_THREAD} * globalId + i] = s + localVals[i];
  }
  if (localId == ${THREADS_PER_GROUP - 1}u) {
    post[groupId.x] = s + localVals[${ITEMS_PER_THREAD} - 1];
  }
}

@compute @workgroup_size(${THREADS_PER_GROUP_X}, ${THREADS_PER_GROUP_Y}, ${THREADS_PER_GROUP_Z})
fn prefixSumPost(@builtin(local_invocation_id) localVec : vec3<u32>) {
  var localId = ${THREADS_PER_GROUP_Y * THREADS_PER_GROUP_Z}u * localVec.x + ${THREADS_PER_GROUP_Z}u * localVec.y + localVec.z;

  var s = 0.;
  var localVals = array<f32, ${ITEMS_PER_THREAD}>();
  for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
    s = s + post[${ITEMS_PER_THREAD} * localId + i];
    localVals[i] = s;
  }
  workerSums[conflictFreeOffset(localId)] = s;
  workgroupBarrier();

  s = partialSum(localId);
  for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
    post[${ITEMS_PER_THREAD} * localId + i] = s + localVals[i];
  }
}

@compute @workgroup_size(${THREADS_PER_GROUP_X}, ${THREADS_PER_GROUP_Y}, ${THREADS_PER_GROUP_Z})
fn prefixSumOut(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(global_invocation_id) globalVec : vec3<u32>) {
  var globalId = globalVec.x * ${THREADS_PER_GROUP_Y * THREADS_PER_GROUP_Z}u + globalVec.y * ${THREADS_PER_GROUP_Z}u + globalVec.z;
  if (groupId.x > 0u) {
    var s = post[groupId.x - 1u];
    for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
      data[${ITEMS_PER_THREAD} * globalId + i] = s + work[${ITEMS_PER_THREAD} * globalId + i];
    }
  } else {
    for (var i = 0u; i < ${ITEMS_PER_THREAD}; i = i + 1u) {
      data[${ITEMS_PER_THREAD} * globalId + i] = work[${ITEMS_PER_THREAD} * globalId + i];
    }
  }
}
`
  })

  const layout = device.createPipelineLayout({
    bindGroupLayouts: [
      device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform'
          }
        }, {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage'
          }
        }, {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage'
          }
        }, {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage'
          }
        }]
      } as const),
    ]
  })

  const prefixSumIn = device.createComputePipeline({
    layout,
    compute: {
      module: prefixSumShader,
      entryPoint: 'prefixSumIn'
    }
  })

  const prefixSumPost = device.createComputePipeline({
    layout,
    compute: {
      module: prefixSumShader,
      entryPoint: 'prefixSumPost'
    }
  })

  const prefixSumOut = device.createComputePipeline({
    layout,
    compute: {
      module: prefixSumShader,
      entryPoint: 'prefixSumOut'
    }
  })


  const kernels = {
    async gpu(n:number) {
      const paramBuffer = device.createBuffer({
        size: 1 * 4,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true
      })
      new Uint32Array(paramBuffer.getMappedRange()).set([ n ])
      paramBuffer.unmap()

      const postBuffer = device.createBuffer({
        label: 'postBuffer',
        size: ITEMS_PER_GROUP * 4,
        usage: GPUBufferUsage.STORAGE
      })
      const dataBuffer = device.createBuffer({
        label: 'dataBuffer',
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
      const workBuffer = device.createBuffer({
        label: 'workBuffer',
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC 
      })
      const readBuffer = device.createBuffer({
        label: 'readBuffer',
        size: n * 4, 
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      })

      const prefixSumBindGroup = device.createBindGroup({
        layout: prefixSumIn.getBindGroupLayout(0),
        entries: [{
          binding: 0,
          resource: {
            buffer: paramBuffer
          }
        }, {
          binding: 1,
          resource: {
            buffer: postBuffer
          }
        }, {
          binding: 2,
          resource: {
            buffer: dataBuffer
          }
        }, {
          binding: 3,
          resource: {
            buffer: workBuffer
          }
        }]
      })

      return {
        async prefixsum (out:Float32Array, src:Float32Array, skipTransfer:boolean) {
          if (!skipTransfer) {
            device.queue.writeBuffer(dataBuffer, 0, src.buffer, src.byteOffset, src.byteLength)
          }

          const commandEncoder = device.createCommandEncoder()
          const passEncoder = commandEncoder.beginComputePass()

          // set up bind groups
          passEncoder.setBindGroup(0, prefixSumBindGroup)
          
          passEncoder.setPipeline(prefixSumIn)
          passEncoder.dispatchWorkgroups(n / ITEMS_PER_GROUP, 1, 1)
          if (n > ITEMS_PER_GROUP) {
            passEncoder.setPipeline(prefixSumPost)
            passEncoder.dispatchWorkgroups(1, 1, 1)
            passEncoder.setPipeline(prefixSumOut)
            passEncoder.dispatchWorkgroups(n / ITEMS_PER_GROUP, 1, 1)
          }

          passEncoder.end()

          if (!skipTransfer) {
            if (n > ITEMS_PER_GROUP) {
              commandEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, 4 * n)
            } else {
              commandEncoder.copyBufferToBuffer(workBuffer, 0, readBuffer, 0, 4 * n)
            }
          }

          device.queue.submit([commandEncoder.finish()])

          if (!skipTransfer) {
            await readBuffer.mapAsync(GPUMapMode.READ);
            out.set(new Float32Array(readBuffer.getMappedRange()))
            readBuffer.unmap()
          } else {
            await device.queue.onSubmittedWorkDone()
          }
        },
        async free () {
          paramBuffer.destroy()
          dataBuffer.destroy()
          readBuffer.destroy()
          postBuffer.destroy()
        }
      }
    },
    async cpu(n:number) {
      return {
        async prefixsum (out:Float32Array, src:Float32Array) {
          let s = 0
          for (let i = 0; i < src.length; ++i) {
            s += src[i]
            out[i] = s
          }
        },
        async free () { }
      }
    },
  } as const

  const minN = Math.log2(ITEMS_PER_GROUP)
  const maxN = Math.min(25, 2 * minN)
  const rangeN = {
    type: 'number',
    min: '' + minN,
    max: '' + maxN,
    step: '1',
  }

  const ui = makeBench({
    header: `A pretty slow and clumsy GPU prefix sum.  Array length N must be a multiple of ${ITEMS_PER_GROUP}.
GPU performance is bottlenecked by data transfer costs.  Disabling CPU transfer will improve performance.`,
    inputs: {
      startN:{
        label: 'Min logN',
        props: { value: '' + minN, ...rangeN },
        value: (x) => +x
      },
      endN:{
        label: 'Max logN',
        props: { value: '' + maxN, ...rangeN },
        value: (x) => +x
      },
      iter:{
        label: 'Iterations per step',
        props: { type: 'number', min: '1', max: '300', step: '1', value: '100' },
        value: (x) => +x,
      },
      transfer:{
        label: 'Enable GPU transfer',
        props: { type: 'checkbox', checked: true },
        value: (_, e) => !!e.checked,
      },
      test: {
        label: 'Test mode',
        props: { type: 'checkbox', checked: false },
        value: (_, e) => !!e.checked
      }
    },
    kernels,
  })

  function randArray (n:number) {
    const A = new Float32Array(n)
    for (let i = 0; i < A.length; ++i) {
      A[i] = 2 * Math.random()  - 1
    }
    return A
  }

  while (true) {
    const {inputs:{startN, endN, iter, transfer, test}, kernel} = await ui.go()

    ui.clear()
    if (test) {
      const n = 1 << startN

      ui.log(`Testing ${kernel} with n=${n}`)

      const alg = await kernels[kernel](n)

      const A = randArray(n)
      const B = new Float32Array(n)
      const C = new Float32Array(n)

      const doTest = async () =>  {
        ui.log('run cpu...')
        C[0] = A[0]
        for (let i = 1; i < n; ++i) {
          C[i] = C[i - 1] + A[i]
          B[i] = 0
        }
        ui.log('run kernel...')
        await alg.prefixsum(B, A, false)
        ui.log('testing...')
        await ui.sleep(100)

        let foundError = false
        for (let i = 0; i < n; ++i) {
          if (Math.abs(C[i] - B[i]) > 0.001) {
            ui.log(`!! ${i}: ${C[i].toFixed(4)} != ${B[i].toFixed(4)}`)
            await ui.sleep(5)
          }
        }
        if (!foundError) {
          ui.log('Pass')
        }
      }

      for (let i = 0; i < n; ++i) {
        A[i] = 1
      }
      ui.log('test 1...')
      await doTest()

      ui.log('test random...')
      await doTest()


      await alg.free()
    } else {
      ui.log(`Benchmarking ${kernel} from n = 2^${startN} to 2^${endN} ${iter}/step....`)

      for (let logn = startN; logn <= endN; ++logn) {
        const n = 1 << logn
        const alg = await kernels[kernel](n)
        const A = randArray(n)
        const B = new Float32Array(n)
        
        const tStart = performance.now()
        for (let i = 0; i < iter; ++i) {
          await alg.prefixsum(B, A, !transfer)
        }
        const tElapsed = performance.now() - tStart

        const work = iter * n
        ui.log(`n=${n}: ~${(work / tElapsed).toPrecision(3)} FLOPs (${tElapsed.toPrecision(4)} ms, avg ${(tElapsed / iter).toPrecision(4)} ms per pass)`)
        await alg.free()

        await ui.sleep(16)
      }

      ui.log('done')
    }
  }
}

main().catch(err => console.error(err))
