"use strict";
(() => {
  // src/boilerplate.ts
  function mustHave(x) {
    if (!x) {
      document.body.innerHTML = `Your browser does not support WebGPU`;
      throw new Error("WebGPU not supported");
    }
    return x;
  }
  mustHave(navigator.gpu);
  function makeBench({ header, inputs, kernels }) {
    const formContainer = document.createElement("div");
    Object.assign(formContainer.style, {
      padding: "32px",
      margin: "32px"
    });
    const headerNode = document.createElement("div");
    headerNode.innerText = header;
    formContainer.appendChild(headerNode);
    const inputElements = Object.entries(inputs).map(([key, { label, props, value }]) => {
      const container = document.createElement("div");
      Object.assign(container.style, {
        padding: "2px"
      });
      const input = document.createElement("input");
      Object.assign(input, props);
      const labelElement = document.createElement("label");
      labelElement.htmlFor = input.id = "input-" + Math.random();
      labelElement.innerText = label;
      container.appendChild(labelElement);
      container.appendChild(input);
      formContainer.appendChild(container);
      return [key, input, value || ((x) => x)];
    });
    const kernelSelectContainer = document.createElement("div");
    Object.assign(kernelSelectContainer.style, {
      padding: "1px"
    });
    const kernelSelect = document.createElement("select");
    for (const k of Object.keys(kernels)) {
      const opt = document.createElement("option");
      opt.value = opt.text = k;
      kernelSelect.appendChild(opt);
    }
    const selectLabel = document.createElement("label");
    selectLabel.innerText = "Kernel: ";
    selectLabel.htmlFor = kernelSelect.id = "kernel-select";
    kernelSelectContainer.appendChild(selectLabel);
    kernelSelectContainer.appendChild(kernelSelect);
    formContainer.appendChild(kernelSelectContainer);
    const goButton = document.createElement("input");
    goButton.type = "button";
    goButton.value = "Go!";
    formContainer.appendChild(goButton);
    const logPre = document.createElement("pre");
    formContainer.appendChild(logPre);
    document.body.appendChild(formContainer);
    return {
      sleep(ms) {
        return new Promise((resolve) => {
          setTimeout(resolve, ms);
        });
      },
      clear() {
        logPre.innerText = "";
      },
      log(line) {
        logPre.innerText += line + "\n";
      },
      go() {
        return new Promise((resolve) => {
          function handler() {
            const inputs2 = /* @__PURE__ */ Object.create(null);
            for (const [key, element, read] of inputElements) {
              inputs2[key] = read(element.value, element);
            }
            resolve({
              inputs: inputs2,
              kernel: kernelSelect.value
            });
          }
          goButton.addEventListener("click", handler);
        });
      }
    };
  }

  // src/lib/scan.ts
  var MAX_BUFFER_SIZE = 134217728;
  var DEFAULT_DATA_TYPE = "f32";
  var DEFAULT_DATA_SIZE = 4;
  var DEFAULT_DATA_FUNC = "A + B";
  var DEFAULT_DATA_UNIT = "0.";
  var WebGPUScan = class {
    constructor(config) {
      this.logNumBanks = 5;
      this.threadsPerGroup = 256;
      this.itemsPerThread = 256;
      this.itemsPerGroup = 65536;
      this.itemSize = 4;
      this.device = config.device;
      if (config["threadsPerGroup"]) {
        this.threadsPerGroup = config["threadsPerGroup"] >>> 0;
        if (this.threadsPerGroup < 1 || this.threadsPerGroup > 256) {
          throw new Error("Threads per group must be between 1 and 256");
        }
      }
      if (config["itemsPerThread"]) {
        this.itemsPerThread = config["itemsPerThread"] >>> 0;
        if (this.itemsPerThread < 1) {
          throw new Error("Items per thread must be > 1");
        }
      }
      this.itemsPerGroup = this.threadsPerGroup * this.itemsPerThread;
      const dataType = config.dataType || DEFAULT_DATA_TYPE;
      const dataSize = config.dataSize || DEFAULT_DATA_SIZE;
      const dataFunc = config.dataFunc || DEFAULT_DATA_FUNC;
      const dataUnit = config.dataUnit || DEFAULT_DATA_UNIT;
      this.itemSize = dataSize;
      this.prefixSumShader = this.device.createShaderModule({
        code: `
${config.header || ""}

@binding(0) @group(0) var<storage, read_write> post : array<${dataType}>;
@binding(0) @group(1) var<storage, read_write> data : array<${dataType}>;
@binding(1) @group(1) var<storage, read_write> work : array<${dataType}>;

fn conflictFreeOffset (offset:u32) -> u32 {
  return offset + (offset >> ${this.logNumBanks});
}
  
var<workgroup> workerSums : array<${dataType}, ${2 * this.threadsPerGroup}>;
fn partialSum (localId : u32) -> ${dataType} {
  var offset = 1u;
  for (var d = ${this.threadsPerGroup >> 1}u; d > 0u; d = d >> 1u) {
    if (localId < d) {
      var ai = conflictFreeOffset(offset * (2u * localId + 1u) - 1u);
      var bi = conflictFreeOffset(offset * (2u * localId + 2u) - 1u);
      var A = workerSums[ai];
      var B = workerSums[bi];
      workerSums[bi] = ${dataFunc};
    }
    offset *= 2u;
    workgroupBarrier();
  }
  if (localId == 0u) {
    workerSums[conflictFreeOffset(${this.threadsPerGroup - 1}u)] = ${dataUnit};
  }
  for (var d = 1u; d < ${this.threadsPerGroup}u; d = d * 2u) {
    offset = offset >> 1u;
    if (localId < d) {
      var ai = conflictFreeOffset(offset * (2u * localId + 1u) - 1u);
      var bi = conflictFreeOffset(offset * (2u * localId + 2u) - 1u);
      var A = workerSums[ai];
      var B = workerSums[bi];
      workerSums[ai] = B;
      workerSums[bi] = ${dataFunc};
    }
    workgroupBarrier();
  }

  return workerSums[conflictFreeOffset(localId)];
}
  
@compute @workgroup_size(${this.threadsPerGroup}, 1, 1)
fn prefixSumIn(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(local_invocation_id) localVec : vec3<u32>,
  @builtin(global_invocation_id) globalVec : vec3<u32>) {
  var localId = localVec.x;
  var globalId = globalVec.x;
  var offset = ${this.itemsPerThread}u * globalId;

  var A = ${dataUnit};
  var localVals = array<${dataType}, ${this.itemsPerThread}>();
  for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
    var B = data[offset + i];
    A = ${dataFunc};
    localVals[i] = A;
  }
  workerSums[conflictFreeOffset(localId)] = A;
  workgroupBarrier();

  A = partialSum(localId);

  for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
    var B = localVals[i];
    var C = ${dataFunc};
    work[offset + i] = C;
    if (i == ${this.itemsPerThread - 1}u && localId == ${this.threadsPerGroup - 1}u) {
      post[groupId.x] = C;
    }
  }
}

@compute @workgroup_size(${this.threadsPerGroup}, 1, 1)
fn prefixSumPost(@builtin(local_invocation_id) localVec : vec3<u32>) {
  var localId = localVec.x;
  var offset = localId * ${this.itemsPerThread}u;

  var A = ${dataUnit};
  var localVals = array<${dataType}, ${this.itemsPerThread}>();
  for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
    var B = post[offset + i];
    A = ${dataFunc};
    localVals[i] = A;
  }
  workerSums[conflictFreeOffset(localId)] = A;
  workgroupBarrier();

  A = partialSum(localId);
  for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
    var B = localVals[i];
    post[offset + i] = ${dataFunc};
  }
}

@compute @workgroup_size(${this.threadsPerGroup}, 1, 1)
fn prefixSumOut(
  @builtin(workgroup_id) groupId : vec3<u32>,
  @builtin(global_invocation_id) globalVec : vec3<u32>) {
  var globalId = globalVec.x;
  var offset = ${this.itemsPerThread}u * globalId;
  if (groupId.x > 0u) {
    var s = post[groupId.x - 1u];
    for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
      data[offset + i] = s + work[offset + i];
    }
  } else {
    for (var i = 0u; i < ${this.itemsPerThread}u; i = i + 1u) {
      data[offset + i] = work[offset + i];
    }
  }
}
`
      });
      this.postBuffer = this.device.createBuffer({
        label: "postBuffer",
        size: this.itemsPerGroup * this.itemSize,
        usage: GPUBufferUsage.STORAGE
      });
      this.postBindGroupLayout = this.device.createBindGroupLayout({
        label: "postBindGroupLayout",
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
            hasDynamicOffset: false,
            minBindingSize: this.itemSize * this.itemsPerGroup
          }
        }]
      });
      this.postBindGroup = this.device.createBindGroup({
        label: "postBindGroup",
        layout: this.postBindGroupLayout,
        entries: [{
          binding: 0,
          resource: {
            buffer: this.postBuffer
          }
        }]
      });
      this.dataBindGroupLayout = this.device.createBindGroupLayout({
        label: "dataBindGroupLayout",
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
            hasDynamicOffset: false,
            minBindingSize: this.itemSize * this.itemsPerGroup
          }
        }, {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage",
            hasDynamicOffset: false,
            minBindingSize: this.itemSize * this.itemsPerGroup
          }
        }]
      });
      const layout = this.device.createPipelineLayout({
        label: "commonScanLayout",
        bindGroupLayouts: [
          this.postBindGroupLayout,
          this.dataBindGroupLayout
        ]
      });
      this.prefixSumIn = this.device.createComputePipelineAsync({
        label: "prefixSumIn",
        layout,
        compute: {
          module: this.prefixSumShader,
          entryPoint: "prefixSumIn"
        }
      });
      this.prefixSumPost = this.device.createComputePipelineAsync({
        label: "prefixSumPost",
        layout: this.device.createPipelineLayout({
          label: "postScanLayout",
          bindGroupLayouts: [this.postBindGroupLayout]
        }),
        compute: {
          module: this.prefixSumShader,
          entryPoint: "prefixSumPost"
        }
      });
      this.prefixSumOut = this.device.createComputePipelineAsync({
        label: "prefixSumOut",
        layout,
        compute: {
          module: this.prefixSumShader,
          entryPoint: "prefixSumOut"
        }
      });
    }
    minItems() {
      return this.itemsPerGroup;
    }
    minSize() {
      return this.minItems() * this.itemSize;
    }
    maxItems() {
      return Math.min(this.itemsPerGroup * this.itemsPerGroup, Math.floor(MAX_BUFFER_SIZE / (this.itemSize * this.itemsPerGroup)) * this.itemsPerGroup);
    }
    maxSize() {
      return this.itemSize;
    }
    async createPass(n, data, work) {
      if (n < this.minItems() || n > this.maxItems() || n % this.itemsPerGroup !== 0) {
        throw new Error("Invalid item count");
      }
      let ownsWorkBuffer = false;
      let workBuffer = null;
      if (n > this.minItems()) {
        if (work) {
          workBuffer = work;
        } else {
          workBuffer = this.device.createBuffer(
            {
              label: "workBuffer",
              size: n * this.itemSize,
              usage: GPUBufferUsage.STORAGE
            }
          );
          ownsWorkBuffer = true;
        }
      }
      let dataBindGroup;
      if (workBuffer) {
        dataBindGroup = this.device.createBindGroup({
          label: "dataBindGroup",
          layout: this.dataBindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: data
            }
          }, {
            binding: 1,
            resource: {
              buffer: workBuffer
            }
          }]
        });
      } else {
        dataBindGroup = this.device.createBindGroup({
          label: "dataBindGroupSmall",
          layout: this.postBindGroupLayout,
          entries: [{
            binding: 0,
            resource: {
              buffer: data
            }
          }]
        });
      }
      return new WebGPUScanPass(
        n / this.itemsPerGroup >>> 0,
        dataBindGroup,
        this.postBindGroup,
        workBuffer,
        ownsWorkBuffer,
        await this.prefixSumIn,
        await this.prefixSumPost,
        await this.prefixSumOut
      );
    }
    destroy() {
      this.postBuffer.destroy();
    }
  };
  var WebGPUScanPass = class {
    constructor(numGroups, dataBindGroup, postBindGroup, work, ownsWorkBuffer, prefixSumIn, prefixSumPost, prefixSumOut) {
      this.numGroups = numGroups;
      this.dataBindGroup = dataBindGroup;
      this.postBindGroup = postBindGroup;
      this.work = work;
      this.ownsWorkBuffer = ownsWorkBuffer;
      this.prefixSumIn = prefixSumIn;
      this.prefixSumPost = prefixSumPost;
      this.prefixSumOut = prefixSumOut;
    }
    run(passEncoder) {
      if (this.work) {
        passEncoder.setBindGroup(0, this.postBindGroup);
        passEncoder.setBindGroup(1, this.dataBindGroup);
        passEncoder.setPipeline(this.prefixSumIn);
        passEncoder.dispatchWorkgroups(this.numGroups);
        passEncoder.setPipeline(this.prefixSumPost);
        passEncoder.dispatchWorkgroups(1);
        passEncoder.setPipeline(this.prefixSumOut);
        passEncoder.dispatchWorkgroups(this.numGroups);
      } else {
        passEncoder.setBindGroup(0, this.dataBindGroup);
        passEncoder.setPipeline(this.prefixSumPost);
        passEncoder.dispatchWorkgroups(1);
      }
    }
    destroy() {
      if (this.ownsWorkBuffer && this.work) {
        this.work.destroy();
      }
    }
  };

  // src/demos/prefix-sum.ts
  async function main() {
    const adapter = mustHave(await navigator.gpu.requestAdapter());
    const device = await adapter.requestDevice();
    const prefixSum = new WebGPUScan({
      device
    });
    const kernels = {
      async gpu(n) {
        const dataBuffer = device.createBuffer({
          label: "dataBuffer",
          size: n * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        const readBuffer = device.createBuffer({
          label: "readBuffer",
          size: n * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        const pass = await prefixSum.createPass(n, dataBuffer);
        return {
          async prefixsum(out, src, skipTransfer) {
            if (!skipTransfer) {
              device.queue.writeBuffer(dataBuffer, 0, src.buffer, src.byteOffset, src.byteLength);
            }
            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            pass.run(passEncoder);
            passEncoder.end();
            if (!skipTransfer) {
              commandEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, 4 * n);
            }
            device.queue.submit([commandEncoder.finish()]);
            if (!skipTransfer) {
              await readBuffer.mapAsync(GPUMapMode.READ);
              out.set(new Float32Array(readBuffer.getMappedRange()));
              readBuffer.unmap();
            } else {
              await device.queue.onSubmittedWorkDone();
            }
          },
          async free() {
            pass.destroy();
            dataBuffer.destroy();
            readBuffer.destroy();
          }
        };
      },
      async cpu(n) {
        return {
          async prefixsum(out, src) {
            let s = 0;
            for (let i = 0; i < src.length; ++i) {
              s += src[i];
              out[i] = s;
            }
          },
          async free() {
          }
        };
      }
    };
    const minN = Math.log2(prefixSum.minItems()) | 0;
    const maxN = Math.log2(prefixSum.maxItems()) | 0;
    const rangeN = {
      type: "number",
      min: "" + minN,
      max: "" + maxN,
      step: "1"
    };
    const ui = makeBench({
      header: `GPU prefix sum/scan demo.  Array length N must be a multiple of ${prefixSum.itemsPerGroup}.
GPU performance is bottlenecked by data transfer costs.  Disabling CPU transfer will improve performance.`,
      inputs: {
        startN: {
          label: "Min logN",
          props: { value: "" + minN, ...rangeN },
          value: (x) => +x
        },
        endN: {
          label: "Max logN",
          props: { value: "" + maxN, ...rangeN },
          value: (x) => +x
        },
        iter: {
          label: "Iterations per step",
          props: { type: "number", min: "1", max: "300", step: "1", value: "100" },
          value: (x) => +x
        },
        transfer: {
          label: "Enable GPU transfer",
          props: { type: "checkbox", checked: true },
          value: (_, e) => !!e.checked
        },
        test: {
          label: "Test mode",
          props: { type: "checkbox", checked: false },
          value: (_, e) => !!e.checked
        }
      },
      kernels
    });
    function randArray(n) {
      const A = new Float32Array(n);
      for (let i = 0; i < A.length; ++i) {
        A[i] = 2 * Math.random() - 1;
      }
      return A;
    }
    while (true) {
      const { inputs: { startN, endN, iter, transfer, test }, kernel } = await ui.go();
      ui.clear();
      if (test) {
        const n = 1 << startN;
        ui.log(`Testing ${kernel} with n=${n}`);
        const alg = await kernels[kernel](n);
        const A = randArray(n);
        const B = new Float32Array(n);
        const C = new Float32Array(n);
        const doTest = async () => {
          ui.log("run cpu...");
          C[0] = A[0];
          for (let i = 1; i < n; ++i) {
            C[i] = C[i - 1] + A[i];
            B[i] = 0;
          }
          ui.log("run kernel...");
          await alg.prefixsum(B, A, false);
          ui.log("testing...");
          await ui.sleep(100);
          let foundError = false;
          for (let i = 0; i < n; ++i) {
            if (Math.abs(C[i] - B[i]) > 1e-3) {
              ui.log(`!! ${i}: ${C[i].toFixed(4)} != ${B[i].toFixed(4)}`);
              await ui.sleep(5);
            }
          }
          if (!foundError) {
            ui.log("Pass");
          }
        };
        for (let i = 0; i < n; ++i) {
          A[i] = 1;
        }
        ui.log("test 1...");
        await doTest();
        ui.log("test random...");
        await doTest();
        await alg.free();
      } else {
        ui.log(`Benchmarking ${kernel} from n = 2^${startN} to 2^${endN} ${iter}/step....`);
        for (let logn = startN; logn <= endN; ++logn) {
          const n = 1 << logn;
          const alg = await kernels[kernel](n);
          const A = randArray(n);
          const B = new Float32Array(n);
          const tStart = performance.now();
          for (let i = 0; i < iter; ++i) {
            await alg.prefixsum(B, A, !transfer);
          }
          const tElapsed = performance.now() - tStart;
          const work = iter * n;
          ui.log(`n=${n}: ~${(work / tElapsed).toPrecision(3)} FLOPs (${tElapsed.toPrecision(4)} ms, avg ${(tElapsed / iter).toPrecision(4)} ms per pass)`);
          await alg.free();
          await ui.sleep(16);
        }
        ui.log("done");
      }
    }
  }
  main().catch((err) => console.error(err));
})();
