import { makeBench, mustHave } from '../boilerplate'

const THREADS_PER_GROUP = 64
const ITEMS_PER_THREAD = 64
const ITEMS_PER_GROUP = ITEMS_PER_THREAD * THREADS_PER_GROUP

async function main () {
  const adapter = mustHave(await navigator.gpu.requestAdapter())
  const device = await adapter.requestDevice()

  const prefixSumBasePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `
const THREADS_PER_GROUP = ${THREADS_PER_GROUP};
const ITEMS_PER_THREAD = ${ITEMS_PER_THREAD};
const ITEMS_PER_GROUP = ${ITEMS_PER_GROUP};
        
struct SumParams {
  N : u32,
}
@binding(0) @group(0) var<uniform> params : SumParams;
        
@binding(1) @group(0) var<storage, read> dataIn : array<f32>;
@binding(2) @group(0) var<storage, read_write> dataOut : array<f32>;
@binding(3) @group(0) var<storage, read_write> post : array<f32>;

var<workgroup> workerSums : array<f32, THREADS_PER_GROUP>;

@compute @workgroup_size(THREADS_PER_GROUP, 1, 1)
fn prefixSum(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) localId : vec3<u32>,
  @builtin(global_invocation_id) globalId : vec3<u32>) {
  var N = params.N;

  var s = 0.;
  var localVals = array<f32, ITEMS_PER_THREAD>();
  for (var i = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
    s = s + dataIn[ITEMS_PER_THREAD * globalId.x + i];
    localVals[i] = s;
  }
  workerSums[localId.x] = s;

  workgroupBarrier();

  // FIXME: try using blelloch here
  s = 0.;
  for (var i = 0u; i < localId.x; i = i + 1u) {
    s = s + workerSums[i];
  }

  for (var i = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
    dataOut[ITEMS_PER_THREAD * globalId.x + i] = s + localVals[i];
  }
  if (localId.x == THREADS_PER_GROUP - 1u) {
    post[groupId.x] = s + localVals[ITEMS_PER_THREAD - 1];
  }
}`
      }),
      entryPoint: 'prefixSum'
    }
  })

  const prefixSumGatherPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: `
const THREADS_PER_GROUP = ${THREADS_PER_GROUP};
const ITEMS_PER_THREAD = ${ITEMS_PER_THREAD};
const ITEMS_PER_GROUP = ${ITEMS_PER_GROUP};
        
struct SumParams {
  N : u32,
}
@binding(0) @group(0) var<uniform> params : SumParams;
        
@binding(1) @group(0) var<storage, read_write> dataOut : array<f32>;
@binding(2) @group(0) var<storage, read> dataIn : array<f32>;
@binding(3) @group(0) var<storage, read> post : array<f32>;

var<workgroup> workerSums : array<f32, THREADS_PER_GROUP>;

@compute @workgroup_size(THREADS_PER_GROUP, 1, 1)
fn prefixSum(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) localId : vec3<u32>,
  @builtin(global_invocation_id) globalId : vec3<u32>) {
  var N = params.N;

  var s = 0.;
  var localVals = array<f32, ITEMS_PER_THREAD>();
  for (var i = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
    var index = ITEMS_PER_THREAD * localId.x + i;
    if (index < groupId.x) {
      s = s + post[index];
    }
    localVals[i] = s;
  }
  workerSums[localId.x] = s;

  workgroupBarrier();

  // FIXME: use blelloch here
  s = 0.;
  for (var i = 0u; i < groupId.x; i = i + 1u) {
    s = s + workerSums[i];
  }

  for (var i = 0u; i < ITEMS_PER_THREAD; i = i + 1u) {
    dataOut[ITEMS_PER_THREAD * globalId.x + i] = s + dataIn[ITEMS_PER_THREAD * globalId.x + i];
  }
}`
      }),
      entryPoint: 'prefixSum'
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

      const dataBuffer = device.createBuffer({
        label: 'dataBuffer',
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      })
      const tempBuffer = device.createBuffer({
        label: 'tempBuffer',
        size: n * 4,
        usage: GPUBufferUsage.STORAGE 
      })
      const readBuffer = device.createBuffer({
        label: 'readBuffer',
        size: n * 4, 
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      })
      const postBuffer = device.createBuffer({
        label: 'swapBuffer',
        size: (n / ITEMS_PER_GROUP) * 4,
        usage: GPUBufferUsage.STORAGE
      })

      const entries = [
        {
          binding: 0,
          resource: {
            buffer: paramBuffer
          }
        },
        {
          binding: 1,
          resource: {
            buffer: dataBuffer
          }
        },
        {
          binding: 2,
          resource: {
            buffer: tempBuffer
          }
        },
        {
          binding: 3,
          resource: {
            buffer: postBuffer
          }
        }
      ]
      
      const baseBindGroup = device.createBindGroup({
        layout: prefixSumBasePipeline.getBindGroupLayout(0),
        entries,
      })

      const gatheBindGroup = device.createBindGroup({
        layout: prefixSumGatherPipeline.getBindGroupLayout(0),
        entries,
      })

      return {
        async prefixsum (out:Float32Array, src:Float32Array, skipTransfer:boolean) {
          if (!skipTransfer) {
            device.queue.writeBuffer(dataBuffer, 0, src.buffer, src.byteOffset, src.byteLength)
          }

          const commandEncoder = device.createCommandEncoder()
          const passEncoder = commandEncoder.beginComputePass()
          passEncoder.setPipeline(prefixSumBasePipeline)
          passEncoder.setBindGroup(0, baseBindGroup)
          passEncoder.dispatchWorkgroups(n / ITEMS_PER_GROUP, 1, 1)
          passEncoder.setPipeline(prefixSumGatherPipeline)
          passEncoder.setBindGroup(0, gatheBindGroup)
          passEncoder.dispatchWorkgroups(n / ITEMS_PER_GROUP, 1, 1)
          passEncoder.end()
          if (!skipTransfer) {
            commandEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, 4 * n)
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
          tempBuffer.destroy()
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

  const ui = makeBench({
    header: `A pretty slow and clumsy GPU prefix sum.  Array length N must be a multiple of ${ITEMS_PER_GROUP}.
Performance may be low due to IO bottlenecks.
Disabling CPU transfer can improve performance.`,
    inputs: {
      startN:{
        label: 'Min logN',
        props: { type: 'number', min: '12', max: '24', step: '1', value: '12' },
        value: (x) => +x
      },
      endN:{
        label: 'Max logN',
        props: { type: 'number', min: '12', max: '24', step: '1', value: '24' },
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
    const {inputs:{startN, endN, iter, transfer}, kernel} = await ui.go()

    ui.clear()
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
  }
}

main().catch(err => console.error(err))
