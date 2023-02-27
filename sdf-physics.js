"use strict";
(() => {
  var __defProp = Object.defineProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };

  // src/boilerplate.ts
  function mustHave(x) {
    if (!x) {
      document.body.innerHTML = `Your browser does not support WebGPU`;
      throw new Error("WebGPU not supported");
    }
    return x;
  }
  mustHave(navigator.gpu);
  function makeCanvas() {
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
      width: "100%",
      height: "100%",
      position: "absolute",
      left: "0",
      top: "0",
      margin: "0",
      padding: "0"
    });
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    document.body.appendChild(canvas);
    return canvas;
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

  // node_modules/gl-matrix/esm/common.js
  var EPSILON = 1e-6;
  var ARRAY_TYPE = typeof Float32Array !== "undefined" ? Float32Array : Array;
  var RANDOM = Math.random;
  var degree = Math.PI / 180;
  if (!Math.hypot)
    Math.hypot = function() {
      var y = 0, i = arguments.length;
      while (i--) {
        y += arguments[i] * arguments[i];
      }
      return Math.sqrt(y);
    };

  // node_modules/gl-matrix/esm/mat3.js
  var mat3_exports = {};
  __export(mat3_exports, {
    add: () => add,
    adjoint: () => adjoint,
    clone: () => clone,
    copy: () => copy,
    create: () => create,
    determinant: () => determinant,
    equals: () => equals,
    exactEquals: () => exactEquals,
    frob: () => frob,
    fromMat2d: () => fromMat2d,
    fromMat4: () => fromMat4,
    fromQuat: () => fromQuat,
    fromRotation: () => fromRotation,
    fromScaling: () => fromScaling,
    fromTranslation: () => fromTranslation,
    fromValues: () => fromValues,
    identity: () => identity,
    invert: () => invert,
    mul: () => mul,
    multiply: () => multiply,
    multiplyScalar: () => multiplyScalar,
    multiplyScalarAndAdd: () => multiplyScalarAndAdd,
    normalFromMat4: () => normalFromMat4,
    projection: () => projection,
    rotate: () => rotate,
    scale: () => scale,
    set: () => set,
    str: () => str,
    sub: () => sub,
    subtract: () => subtract,
    translate: () => translate,
    transpose: () => transpose
  });
  function create() {
    var out = new ARRAY_TYPE(9);
    if (ARRAY_TYPE != Float32Array) {
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[5] = 0;
      out[6] = 0;
      out[7] = 0;
    }
    out[0] = 1;
    out[4] = 1;
    out[8] = 1;
    return out;
  }
  function fromMat4(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[4];
    out[4] = a[5];
    out[5] = a[6];
    out[6] = a[8];
    out[7] = a[9];
    out[8] = a[10];
    return out;
  }
  function clone(a) {
    var out = new ARRAY_TYPE(9);
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    return out;
  }
  function copy(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    return out;
  }
  function fromValues(m00, m01, m02, m10, m11, m12, m20, m21, m22) {
    var out = new ARRAY_TYPE(9);
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m10;
    out[4] = m11;
    out[5] = m12;
    out[6] = m20;
    out[7] = m21;
    out[8] = m22;
    return out;
  }
  function set(out, m00, m01, m02, m10, m11, m12, m20, m21, m22) {
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m10;
    out[4] = m11;
    out[5] = m12;
    out[6] = m20;
    out[7] = m21;
    out[8] = m22;
    return out;
  }
  function identity(out) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 1;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 1;
    return out;
  }
  function transpose(out, a) {
    if (out === a) {
      var a01 = a[1], a02 = a[2], a12 = a[5];
      out[1] = a[3];
      out[2] = a[6];
      out[3] = a01;
      out[5] = a[7];
      out[6] = a02;
      out[7] = a12;
    } else {
      out[0] = a[0];
      out[1] = a[3];
      out[2] = a[6];
      out[3] = a[1];
      out[4] = a[4];
      out[5] = a[7];
      out[6] = a[2];
      out[7] = a[5];
      out[8] = a[8];
    }
    return out;
  }
  function invert(out, a) {
    var a00 = a[0], a01 = a[1], a02 = a[2];
    var a10 = a[3], a11 = a[4], a12 = a[5];
    var a20 = a[6], a21 = a[7], a22 = a[8];
    var b01 = a22 * a11 - a12 * a21;
    var b11 = -a22 * a10 + a12 * a20;
    var b21 = a21 * a10 - a11 * a20;
    var det = a00 * b01 + a01 * b11 + a02 * b21;
    if (!det) {
      return null;
    }
    det = 1 / det;
    out[0] = b01 * det;
    out[1] = (-a22 * a01 + a02 * a21) * det;
    out[2] = (a12 * a01 - a02 * a11) * det;
    out[3] = b11 * det;
    out[4] = (a22 * a00 - a02 * a20) * det;
    out[5] = (-a12 * a00 + a02 * a10) * det;
    out[6] = b21 * det;
    out[7] = (-a21 * a00 + a01 * a20) * det;
    out[8] = (a11 * a00 - a01 * a10) * det;
    return out;
  }
  function adjoint(out, a) {
    var a00 = a[0], a01 = a[1], a02 = a[2];
    var a10 = a[3], a11 = a[4], a12 = a[5];
    var a20 = a[6], a21 = a[7], a22 = a[8];
    out[0] = a11 * a22 - a12 * a21;
    out[1] = a02 * a21 - a01 * a22;
    out[2] = a01 * a12 - a02 * a11;
    out[3] = a12 * a20 - a10 * a22;
    out[4] = a00 * a22 - a02 * a20;
    out[5] = a02 * a10 - a00 * a12;
    out[6] = a10 * a21 - a11 * a20;
    out[7] = a01 * a20 - a00 * a21;
    out[8] = a00 * a11 - a01 * a10;
    return out;
  }
  function determinant(a) {
    var a00 = a[0], a01 = a[1], a02 = a[2];
    var a10 = a[3], a11 = a[4], a12 = a[5];
    var a20 = a[6], a21 = a[7], a22 = a[8];
    return a00 * (a22 * a11 - a12 * a21) + a01 * (-a22 * a10 + a12 * a20) + a02 * (a21 * a10 - a11 * a20);
  }
  function multiply(out, a, b) {
    var a00 = a[0], a01 = a[1], a02 = a[2];
    var a10 = a[3], a11 = a[4], a12 = a[5];
    var a20 = a[6], a21 = a[7], a22 = a[8];
    var b00 = b[0], b01 = b[1], b02 = b[2];
    var b10 = b[3], b11 = b[4], b12 = b[5];
    var b20 = b[6], b21 = b[7], b22 = b[8];
    out[0] = b00 * a00 + b01 * a10 + b02 * a20;
    out[1] = b00 * a01 + b01 * a11 + b02 * a21;
    out[2] = b00 * a02 + b01 * a12 + b02 * a22;
    out[3] = b10 * a00 + b11 * a10 + b12 * a20;
    out[4] = b10 * a01 + b11 * a11 + b12 * a21;
    out[5] = b10 * a02 + b11 * a12 + b12 * a22;
    out[6] = b20 * a00 + b21 * a10 + b22 * a20;
    out[7] = b20 * a01 + b21 * a11 + b22 * a21;
    out[8] = b20 * a02 + b21 * a12 + b22 * a22;
    return out;
  }
  function translate(out, a, v) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a10 = a[3], a11 = a[4], a12 = a[5], a20 = a[6], a21 = a[7], a22 = a[8], x = v[0], y = v[1];
    out[0] = a00;
    out[1] = a01;
    out[2] = a02;
    out[3] = a10;
    out[4] = a11;
    out[5] = a12;
    out[6] = x * a00 + y * a10 + a20;
    out[7] = x * a01 + y * a11 + a21;
    out[8] = x * a02 + y * a12 + a22;
    return out;
  }
  function rotate(out, a, rad) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a10 = a[3], a11 = a[4], a12 = a[5], a20 = a[6], a21 = a[7], a22 = a[8], s = Math.sin(rad), c = Math.cos(rad);
    out[0] = c * a00 + s * a10;
    out[1] = c * a01 + s * a11;
    out[2] = c * a02 + s * a12;
    out[3] = c * a10 - s * a00;
    out[4] = c * a11 - s * a01;
    out[5] = c * a12 - s * a02;
    out[6] = a20;
    out[7] = a21;
    out[8] = a22;
    return out;
  }
  function scale(out, a, v) {
    var x = v[0], y = v[1];
    out[0] = x * a[0];
    out[1] = x * a[1];
    out[2] = x * a[2];
    out[3] = y * a[3];
    out[4] = y * a[4];
    out[5] = y * a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    return out;
  }
  function fromTranslation(out, v) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 1;
    out[5] = 0;
    out[6] = v[0];
    out[7] = v[1];
    out[8] = 1;
    return out;
  }
  function fromRotation(out, rad) {
    var s = Math.sin(rad), c = Math.cos(rad);
    out[0] = c;
    out[1] = s;
    out[2] = 0;
    out[3] = -s;
    out[4] = c;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 1;
    return out;
  }
  function fromScaling(out, v) {
    out[0] = v[0];
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = v[1];
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = 1;
    return out;
  }
  function fromMat2d(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = 0;
    out[3] = a[2];
    out[4] = a[3];
    out[5] = 0;
    out[6] = a[4];
    out[7] = a[5];
    out[8] = 1;
    return out;
  }
  function fromQuat(out, q) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var x2 = x + x;
    var y2 = y + y;
    var z2 = z + z;
    var xx = x * x2;
    var yx = y * x2;
    var yy = y * y2;
    var zx = z * x2;
    var zy = z * y2;
    var zz = z * z2;
    var wx = w * x2;
    var wy = w * y2;
    var wz = w * z2;
    out[0] = 1 - yy - zz;
    out[3] = yx - wz;
    out[6] = zx + wy;
    out[1] = yx + wz;
    out[4] = 1 - xx - zz;
    out[7] = zy - wx;
    out[2] = zx - wy;
    out[5] = zy + wx;
    out[8] = 1 - xx - yy;
    return out;
  }
  function normalFromMat4(out, a) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    var b00 = a00 * a11 - a01 * a10;
    var b01 = a00 * a12 - a02 * a10;
    var b02 = a00 * a13 - a03 * a10;
    var b03 = a01 * a12 - a02 * a11;
    var b04 = a01 * a13 - a03 * a11;
    var b05 = a02 * a13 - a03 * a12;
    var b06 = a20 * a31 - a21 * a30;
    var b07 = a20 * a32 - a22 * a30;
    var b08 = a20 * a33 - a23 * a30;
    var b09 = a21 * a32 - a22 * a31;
    var b10 = a21 * a33 - a23 * a31;
    var b11 = a22 * a33 - a23 * a32;
    var det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) {
      return null;
    }
    det = 1 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[2] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[3] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[4] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[5] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[6] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[7] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[8] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    return out;
  }
  function projection(out, width, height) {
    out[0] = 2 / width;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = -2 / height;
    out[5] = 0;
    out[6] = -1;
    out[7] = 1;
    out[8] = 1;
    return out;
  }
  function str(a) {
    return "mat3(" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + ", " + a[4] + ", " + a[5] + ", " + a[6] + ", " + a[7] + ", " + a[8] + ")";
  }
  function frob(a) {
    return Math.hypot(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
  }
  function add(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];
    out[6] = a[6] + b[6];
    out[7] = a[7] + b[7];
    out[8] = a[8] + b[8];
    return out;
  }
  function subtract(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    out[4] = a[4] - b[4];
    out[5] = a[5] - b[5];
    out[6] = a[6] - b[6];
    out[7] = a[7] - b[7];
    out[8] = a[8] - b[8];
    return out;
  }
  function multiplyScalar(out, a, b) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b;
    out[6] = a[6] * b;
    out[7] = a[7] * b;
    out[8] = a[8] * b;
    return out;
  }
  function multiplyScalarAndAdd(out, a, b, scale5) {
    out[0] = a[0] + b[0] * scale5;
    out[1] = a[1] + b[1] * scale5;
    out[2] = a[2] + b[2] * scale5;
    out[3] = a[3] + b[3] * scale5;
    out[4] = a[4] + b[4] * scale5;
    out[5] = a[5] + b[5] * scale5;
    out[6] = a[6] + b[6] * scale5;
    out[7] = a[7] + b[7] * scale5;
    out[8] = a[8] + b[8] * scale5;
    return out;
  }
  function exactEquals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3] && a[4] === b[4] && a[5] === b[5] && a[6] === b[6] && a[7] === b[7] && a[8] === b[8];
  }
  function equals(a, b) {
    var a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
    var b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
    return Math.abs(a0 - b0) <= EPSILON * Math.max(1, Math.abs(a0), Math.abs(b0)) && Math.abs(a1 - b1) <= EPSILON * Math.max(1, Math.abs(a1), Math.abs(b1)) && Math.abs(a2 - b2) <= EPSILON * Math.max(1, Math.abs(a2), Math.abs(b2)) && Math.abs(a3 - b3) <= EPSILON * Math.max(1, Math.abs(a3), Math.abs(b3)) && Math.abs(a4 - b4) <= EPSILON * Math.max(1, Math.abs(a4), Math.abs(b4)) && Math.abs(a5 - b5) <= EPSILON * Math.max(1, Math.abs(a5), Math.abs(b5)) && Math.abs(a6 - b6) <= EPSILON * Math.max(1, Math.abs(a6), Math.abs(b6)) && Math.abs(a7 - b7) <= EPSILON * Math.max(1, Math.abs(a7), Math.abs(b7)) && Math.abs(a8 - b8) <= EPSILON * Math.max(1, Math.abs(a8), Math.abs(b8));
  }
  var mul = multiply;
  var sub = subtract;

  // node_modules/gl-matrix/esm/mat4.js
  var mat4_exports = {};
  __export(mat4_exports, {
    add: () => add2,
    adjoint: () => adjoint2,
    clone: () => clone2,
    copy: () => copy2,
    create: () => create2,
    determinant: () => determinant2,
    equals: () => equals2,
    exactEquals: () => exactEquals2,
    frob: () => frob2,
    fromQuat: () => fromQuat3,
    fromQuat2: () => fromQuat2,
    fromRotation: () => fromRotation2,
    fromRotationTranslation: () => fromRotationTranslation,
    fromRotationTranslationScale: () => fromRotationTranslationScale,
    fromRotationTranslationScaleOrigin: () => fromRotationTranslationScaleOrigin,
    fromScaling: () => fromScaling2,
    fromTranslation: () => fromTranslation2,
    fromValues: () => fromValues2,
    fromXRotation: () => fromXRotation,
    fromYRotation: () => fromYRotation,
    fromZRotation: () => fromZRotation,
    frustum: () => frustum,
    getRotation: () => getRotation,
    getScaling: () => getScaling,
    getTranslation: () => getTranslation,
    identity: () => identity2,
    invert: () => invert2,
    lookAt: () => lookAt,
    mul: () => mul2,
    multiply: () => multiply2,
    multiplyScalar: () => multiplyScalar2,
    multiplyScalarAndAdd: () => multiplyScalarAndAdd2,
    ortho: () => ortho,
    orthoNO: () => orthoNO,
    orthoZO: () => orthoZO,
    perspective: () => perspective,
    perspectiveFromFieldOfView: () => perspectiveFromFieldOfView,
    perspectiveNO: () => perspectiveNO,
    perspectiveZO: () => perspectiveZO,
    rotate: () => rotate2,
    rotateX: () => rotateX,
    rotateY: () => rotateY,
    rotateZ: () => rotateZ,
    scale: () => scale2,
    set: () => set2,
    str: () => str2,
    sub: () => sub2,
    subtract: () => subtract2,
    targetTo: () => targetTo,
    translate: () => translate2,
    transpose: () => transpose2
  });
  function create2() {
    var out = new ARRAY_TYPE(16);
    if (ARRAY_TYPE != Float32Array) {
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[4] = 0;
      out[6] = 0;
      out[7] = 0;
      out[8] = 0;
      out[9] = 0;
      out[11] = 0;
      out[12] = 0;
      out[13] = 0;
      out[14] = 0;
    }
    out[0] = 1;
    out[5] = 1;
    out[10] = 1;
    out[15] = 1;
    return out;
  }
  function clone2(a) {
    var out = new ARRAY_TYPE(16);
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    out[9] = a[9];
    out[10] = a[10];
    out[11] = a[11];
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];
    return out;
  }
  function copy2(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    out[9] = a[9];
    out[10] = a[10];
    out[11] = a[11];
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];
    return out;
  }
  function fromValues2(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33) {
    var out = new ARRAY_TYPE(16);
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m03;
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = m13;
    out[8] = m20;
    out[9] = m21;
    out[10] = m22;
    out[11] = m23;
    out[12] = m30;
    out[13] = m31;
    out[14] = m32;
    out[15] = m33;
    return out;
  }
  function set2(out, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33) {
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m03;
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = m13;
    out[8] = m20;
    out[9] = m21;
    out[10] = m22;
    out[11] = m23;
    out[12] = m30;
    out[13] = m31;
    out[14] = m32;
    out[15] = m33;
    return out;
  }
  function identity2(out) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 1;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 1;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function transpose2(out, a) {
    if (out === a) {
      var a01 = a[1], a02 = a[2], a03 = a[3];
      var a12 = a[6], a13 = a[7];
      var a23 = a[11];
      out[1] = a[4];
      out[2] = a[8];
      out[3] = a[12];
      out[4] = a01;
      out[6] = a[9];
      out[7] = a[13];
      out[8] = a02;
      out[9] = a12;
      out[11] = a[14];
      out[12] = a03;
      out[13] = a13;
      out[14] = a23;
    } else {
      out[0] = a[0];
      out[1] = a[4];
      out[2] = a[8];
      out[3] = a[12];
      out[4] = a[1];
      out[5] = a[5];
      out[6] = a[9];
      out[7] = a[13];
      out[8] = a[2];
      out[9] = a[6];
      out[10] = a[10];
      out[11] = a[14];
      out[12] = a[3];
      out[13] = a[7];
      out[14] = a[11];
      out[15] = a[15];
    }
    return out;
  }
  function invert2(out, a) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    var b00 = a00 * a11 - a01 * a10;
    var b01 = a00 * a12 - a02 * a10;
    var b02 = a00 * a13 - a03 * a10;
    var b03 = a01 * a12 - a02 * a11;
    var b04 = a01 * a13 - a03 * a11;
    var b05 = a02 * a13 - a03 * a12;
    var b06 = a20 * a31 - a21 * a30;
    var b07 = a20 * a32 - a22 * a30;
    var b08 = a20 * a33 - a23 * a30;
    var b09 = a21 * a32 - a22 * a31;
    var b10 = a21 * a33 - a23 * a31;
    var b11 = a22 * a33 - a23 * a32;
    var det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) {
      return null;
    }
    det = 1 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return out;
  }
  function adjoint2(out, a) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    out[0] = a11 * (a22 * a33 - a23 * a32) - a21 * (a12 * a33 - a13 * a32) + a31 * (a12 * a23 - a13 * a22);
    out[1] = -(a01 * (a22 * a33 - a23 * a32) - a21 * (a02 * a33 - a03 * a32) + a31 * (a02 * a23 - a03 * a22));
    out[2] = a01 * (a12 * a33 - a13 * a32) - a11 * (a02 * a33 - a03 * a32) + a31 * (a02 * a13 - a03 * a12);
    out[3] = -(a01 * (a12 * a23 - a13 * a22) - a11 * (a02 * a23 - a03 * a22) + a21 * (a02 * a13 - a03 * a12));
    out[4] = -(a10 * (a22 * a33 - a23 * a32) - a20 * (a12 * a33 - a13 * a32) + a30 * (a12 * a23 - a13 * a22));
    out[5] = a00 * (a22 * a33 - a23 * a32) - a20 * (a02 * a33 - a03 * a32) + a30 * (a02 * a23 - a03 * a22);
    out[6] = -(a00 * (a12 * a33 - a13 * a32) - a10 * (a02 * a33 - a03 * a32) + a30 * (a02 * a13 - a03 * a12));
    out[7] = a00 * (a12 * a23 - a13 * a22) - a10 * (a02 * a23 - a03 * a22) + a20 * (a02 * a13 - a03 * a12);
    out[8] = a10 * (a21 * a33 - a23 * a31) - a20 * (a11 * a33 - a13 * a31) + a30 * (a11 * a23 - a13 * a21);
    out[9] = -(a00 * (a21 * a33 - a23 * a31) - a20 * (a01 * a33 - a03 * a31) + a30 * (a01 * a23 - a03 * a21));
    out[10] = a00 * (a11 * a33 - a13 * a31) - a10 * (a01 * a33 - a03 * a31) + a30 * (a01 * a13 - a03 * a11);
    out[11] = -(a00 * (a11 * a23 - a13 * a21) - a10 * (a01 * a23 - a03 * a21) + a20 * (a01 * a13 - a03 * a11));
    out[12] = -(a10 * (a21 * a32 - a22 * a31) - a20 * (a11 * a32 - a12 * a31) + a30 * (a11 * a22 - a12 * a21));
    out[13] = a00 * (a21 * a32 - a22 * a31) - a20 * (a01 * a32 - a02 * a31) + a30 * (a01 * a22 - a02 * a21);
    out[14] = -(a00 * (a11 * a32 - a12 * a31) - a10 * (a01 * a32 - a02 * a31) + a30 * (a01 * a12 - a02 * a11));
    out[15] = a00 * (a11 * a22 - a12 * a21) - a10 * (a01 * a22 - a02 * a21) + a20 * (a01 * a12 - a02 * a11);
    return out;
  }
  function determinant2(a) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    var b00 = a00 * a11 - a01 * a10;
    var b01 = a00 * a12 - a02 * a10;
    var b02 = a00 * a13 - a03 * a10;
    var b03 = a01 * a12 - a02 * a11;
    var b04 = a01 * a13 - a03 * a11;
    var b05 = a02 * a13 - a03 * a12;
    var b06 = a20 * a31 - a21 * a30;
    var b07 = a20 * a32 - a22 * a30;
    var b08 = a20 * a33 - a23 * a30;
    var b09 = a21 * a32 - a22 * a31;
    var b10 = a21 * a33 - a23 * a31;
    var b11 = a22 * a33 - a23 * a32;
    return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  }
  function multiply2(out, a, b) {
    var a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    var a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    var a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    var a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    var b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[4];
    b1 = b[5];
    b2 = b[6];
    b3 = b[7];
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[8];
    b1 = b[9];
    b2 = b[10];
    b3 = b[11];
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    b0 = b[12];
    b1 = b[13];
    b2 = b[14];
    b3 = b[15];
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
  }
  function translate2(out, a, v) {
    var x = v[0], y = v[1], z = v[2];
    var a00, a01, a02, a03;
    var a10, a11, a12, a13;
    var a20, a21, a22, a23;
    if (a === out) {
      out[12] = a[0] * x + a[4] * y + a[8] * z + a[12];
      out[13] = a[1] * x + a[5] * y + a[9] * z + a[13];
      out[14] = a[2] * x + a[6] * y + a[10] * z + a[14];
      out[15] = a[3] * x + a[7] * y + a[11] * z + a[15];
    } else {
      a00 = a[0];
      a01 = a[1];
      a02 = a[2];
      a03 = a[3];
      a10 = a[4];
      a11 = a[5];
      a12 = a[6];
      a13 = a[7];
      a20 = a[8];
      a21 = a[9];
      a22 = a[10];
      a23 = a[11];
      out[0] = a00;
      out[1] = a01;
      out[2] = a02;
      out[3] = a03;
      out[4] = a10;
      out[5] = a11;
      out[6] = a12;
      out[7] = a13;
      out[8] = a20;
      out[9] = a21;
      out[10] = a22;
      out[11] = a23;
      out[12] = a00 * x + a10 * y + a20 * z + a[12];
      out[13] = a01 * x + a11 * y + a21 * z + a[13];
      out[14] = a02 * x + a12 * y + a22 * z + a[14];
      out[15] = a03 * x + a13 * y + a23 * z + a[15];
    }
    return out;
  }
  function scale2(out, a, v) {
    var x = v[0], y = v[1], z = v[2];
    out[0] = a[0] * x;
    out[1] = a[1] * x;
    out[2] = a[2] * x;
    out[3] = a[3] * x;
    out[4] = a[4] * y;
    out[5] = a[5] * y;
    out[6] = a[6] * y;
    out[7] = a[7] * y;
    out[8] = a[8] * z;
    out[9] = a[9] * z;
    out[10] = a[10] * z;
    out[11] = a[11] * z;
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];
    return out;
  }
  function rotate2(out, a, rad, axis) {
    var x = axis[0], y = axis[1], z = axis[2];
    var len4 = Math.hypot(x, y, z);
    var s, c, t;
    var a00, a01, a02, a03;
    var a10, a11, a12, a13;
    var a20, a21, a22, a23;
    var b00, b01, b02;
    var b10, b11, b12;
    var b20, b21, b22;
    if (len4 < EPSILON) {
      return null;
    }
    len4 = 1 / len4;
    x *= len4;
    y *= len4;
    z *= len4;
    s = Math.sin(rad);
    c = Math.cos(rad);
    t = 1 - c;
    a00 = a[0];
    a01 = a[1];
    a02 = a[2];
    a03 = a[3];
    a10 = a[4];
    a11 = a[5];
    a12 = a[6];
    a13 = a[7];
    a20 = a[8];
    a21 = a[9];
    a22 = a[10];
    a23 = a[11];
    b00 = x * x * t + c;
    b01 = y * x * t + z * s;
    b02 = z * x * t - y * s;
    b10 = x * y * t - z * s;
    b11 = y * y * t + c;
    b12 = z * y * t + x * s;
    b20 = x * z * t + y * s;
    b21 = y * z * t - x * s;
    b22 = z * z * t + c;
    out[0] = a00 * b00 + a10 * b01 + a20 * b02;
    out[1] = a01 * b00 + a11 * b01 + a21 * b02;
    out[2] = a02 * b00 + a12 * b01 + a22 * b02;
    out[3] = a03 * b00 + a13 * b01 + a23 * b02;
    out[4] = a00 * b10 + a10 * b11 + a20 * b12;
    out[5] = a01 * b10 + a11 * b11 + a21 * b12;
    out[6] = a02 * b10 + a12 * b11 + a22 * b12;
    out[7] = a03 * b10 + a13 * b11 + a23 * b12;
    out[8] = a00 * b20 + a10 * b21 + a20 * b22;
    out[9] = a01 * b20 + a11 * b21 + a21 * b22;
    out[10] = a02 * b20 + a12 * b21 + a22 * b22;
    out[11] = a03 * b20 + a13 * b21 + a23 * b22;
    if (a !== out) {
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }
    return out;
  }
  function rotateX(out, a, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    var a10 = a[4];
    var a11 = a[5];
    var a12 = a[6];
    var a13 = a[7];
    var a20 = a[8];
    var a21 = a[9];
    var a22 = a[10];
    var a23 = a[11];
    if (a !== out) {
      out[0] = a[0];
      out[1] = a[1];
      out[2] = a[2];
      out[3] = a[3];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }
    out[4] = a10 * c + a20 * s;
    out[5] = a11 * c + a21 * s;
    out[6] = a12 * c + a22 * s;
    out[7] = a13 * c + a23 * s;
    out[8] = a20 * c - a10 * s;
    out[9] = a21 * c - a11 * s;
    out[10] = a22 * c - a12 * s;
    out[11] = a23 * c - a13 * s;
    return out;
  }
  function rotateY(out, a, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    var a00 = a[0];
    var a01 = a[1];
    var a02 = a[2];
    var a03 = a[3];
    var a20 = a[8];
    var a21 = a[9];
    var a22 = a[10];
    var a23 = a[11];
    if (a !== out) {
      out[4] = a[4];
      out[5] = a[5];
      out[6] = a[6];
      out[7] = a[7];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }
    out[0] = a00 * c - a20 * s;
    out[1] = a01 * c - a21 * s;
    out[2] = a02 * c - a22 * s;
    out[3] = a03 * c - a23 * s;
    out[8] = a00 * s + a20 * c;
    out[9] = a01 * s + a21 * c;
    out[10] = a02 * s + a22 * c;
    out[11] = a03 * s + a23 * c;
    return out;
  }
  function rotateZ(out, a, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    var a00 = a[0];
    var a01 = a[1];
    var a02 = a[2];
    var a03 = a[3];
    var a10 = a[4];
    var a11 = a[5];
    var a12 = a[6];
    var a13 = a[7];
    if (a !== out) {
      out[8] = a[8];
      out[9] = a[9];
      out[10] = a[10];
      out[11] = a[11];
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
    }
    out[0] = a00 * c + a10 * s;
    out[1] = a01 * c + a11 * s;
    out[2] = a02 * c + a12 * s;
    out[3] = a03 * c + a13 * s;
    out[4] = a10 * c - a00 * s;
    out[5] = a11 * c - a01 * s;
    out[6] = a12 * c - a02 * s;
    out[7] = a13 * c - a03 * s;
    return out;
  }
  function fromTranslation2(out, v) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 1;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 1;
    out[11] = 0;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1;
    return out;
  }
  function fromScaling2(out, v) {
    out[0] = v[0];
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = v[1];
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = v[2];
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function fromRotation2(out, rad, axis) {
    var x = axis[0], y = axis[1], z = axis[2];
    var len4 = Math.hypot(x, y, z);
    var s, c, t;
    if (len4 < EPSILON) {
      return null;
    }
    len4 = 1 / len4;
    x *= len4;
    y *= len4;
    z *= len4;
    s = Math.sin(rad);
    c = Math.cos(rad);
    t = 1 - c;
    out[0] = x * x * t + c;
    out[1] = y * x * t + z * s;
    out[2] = z * x * t - y * s;
    out[3] = 0;
    out[4] = x * y * t - z * s;
    out[5] = y * y * t + c;
    out[6] = z * y * t + x * s;
    out[7] = 0;
    out[8] = x * z * t + y * s;
    out[9] = y * z * t - x * s;
    out[10] = z * z * t + c;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function fromXRotation(out, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = c;
    out[6] = s;
    out[7] = 0;
    out[8] = 0;
    out[9] = -s;
    out[10] = c;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function fromYRotation(out, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    out[0] = c;
    out[1] = 0;
    out[2] = -s;
    out[3] = 0;
    out[4] = 0;
    out[5] = 1;
    out[6] = 0;
    out[7] = 0;
    out[8] = s;
    out[9] = 0;
    out[10] = c;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function fromZRotation(out, rad) {
    var s = Math.sin(rad);
    var c = Math.cos(rad);
    out[0] = c;
    out[1] = s;
    out[2] = 0;
    out[3] = 0;
    out[4] = -s;
    out[5] = c;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 1;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function fromRotationTranslation(out, q, v) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var x2 = x + x;
    var y2 = y + y;
    var z2 = z + z;
    var xx = x * x2;
    var xy = x * y2;
    var xz = x * z2;
    var yy = y * y2;
    var yz = y * z2;
    var zz = z * z2;
    var wx = w * x2;
    var wy = w * y2;
    var wz = w * z2;
    out[0] = 1 - (yy + zz);
    out[1] = xy + wz;
    out[2] = xz - wy;
    out[3] = 0;
    out[4] = xy - wz;
    out[5] = 1 - (xx + zz);
    out[6] = yz + wx;
    out[7] = 0;
    out[8] = xz + wy;
    out[9] = yz - wx;
    out[10] = 1 - (xx + yy);
    out[11] = 0;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1;
    return out;
  }
  function fromQuat2(out, a) {
    var translation = new ARRAY_TYPE(3);
    var bx = -a[0], by = -a[1], bz = -a[2], bw = a[3], ax = a[4], ay = a[5], az = a[6], aw = a[7];
    var magnitude = bx * bx + by * by + bz * bz + bw * bw;
    if (magnitude > 0) {
      translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2 / magnitude;
      translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2 / magnitude;
      translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2 / magnitude;
    } else {
      translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2;
      translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2;
      translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2;
    }
    fromRotationTranslation(out, a, translation);
    return out;
  }
  function getTranslation(out, mat) {
    out[0] = mat[12];
    out[1] = mat[13];
    out[2] = mat[14];
    return out;
  }
  function getScaling(out, mat) {
    var m11 = mat[0];
    var m12 = mat[1];
    var m13 = mat[2];
    var m21 = mat[4];
    var m22 = mat[5];
    var m23 = mat[6];
    var m31 = mat[8];
    var m32 = mat[9];
    var m33 = mat[10];
    out[0] = Math.hypot(m11, m12, m13);
    out[1] = Math.hypot(m21, m22, m23);
    out[2] = Math.hypot(m31, m32, m33);
    return out;
  }
  function getRotation(out, mat) {
    var scaling = new ARRAY_TYPE(3);
    getScaling(scaling, mat);
    var is1 = 1 / scaling[0];
    var is2 = 1 / scaling[1];
    var is3 = 1 / scaling[2];
    var sm11 = mat[0] * is1;
    var sm12 = mat[1] * is2;
    var sm13 = mat[2] * is3;
    var sm21 = mat[4] * is1;
    var sm22 = mat[5] * is2;
    var sm23 = mat[6] * is3;
    var sm31 = mat[8] * is1;
    var sm32 = mat[9] * is2;
    var sm33 = mat[10] * is3;
    var trace = sm11 + sm22 + sm33;
    var S = 0;
    if (trace > 0) {
      S = Math.sqrt(trace + 1) * 2;
      out[3] = 0.25 * S;
      out[0] = (sm23 - sm32) / S;
      out[1] = (sm31 - sm13) / S;
      out[2] = (sm12 - sm21) / S;
    } else if (sm11 > sm22 && sm11 > sm33) {
      S = Math.sqrt(1 + sm11 - sm22 - sm33) * 2;
      out[3] = (sm23 - sm32) / S;
      out[0] = 0.25 * S;
      out[1] = (sm12 + sm21) / S;
      out[2] = (sm31 + sm13) / S;
    } else if (sm22 > sm33) {
      S = Math.sqrt(1 + sm22 - sm11 - sm33) * 2;
      out[3] = (sm31 - sm13) / S;
      out[0] = (sm12 + sm21) / S;
      out[1] = 0.25 * S;
      out[2] = (sm23 + sm32) / S;
    } else {
      S = Math.sqrt(1 + sm33 - sm11 - sm22) * 2;
      out[3] = (sm12 - sm21) / S;
      out[0] = (sm31 + sm13) / S;
      out[1] = (sm23 + sm32) / S;
      out[2] = 0.25 * S;
    }
    return out;
  }
  function fromRotationTranslationScale(out, q, v, s) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var x2 = x + x;
    var y2 = y + y;
    var z2 = z + z;
    var xx = x * x2;
    var xy = x * y2;
    var xz = x * z2;
    var yy = y * y2;
    var yz = y * z2;
    var zz = z * z2;
    var wx = w * x2;
    var wy = w * y2;
    var wz = w * z2;
    var sx = s[0];
    var sy = s[1];
    var sz = s[2];
    out[0] = (1 - (yy + zz)) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - (xx + zz)) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - (xx + yy)) * sz;
    out[11] = 0;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1;
    return out;
  }
  function fromRotationTranslationScaleOrigin(out, q, v, s, o) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var x2 = x + x;
    var y2 = y + y;
    var z2 = z + z;
    var xx = x * x2;
    var xy = x * y2;
    var xz = x * z2;
    var yy = y * y2;
    var yz = y * z2;
    var zz = z * z2;
    var wx = w * x2;
    var wy = w * y2;
    var wz = w * z2;
    var sx = s[0];
    var sy = s[1];
    var sz = s[2];
    var ox = o[0];
    var oy = o[1];
    var oz = o[2];
    var out0 = (1 - (yy + zz)) * sx;
    var out1 = (xy + wz) * sx;
    var out2 = (xz - wy) * sx;
    var out4 = (xy - wz) * sy;
    var out5 = (1 - (xx + zz)) * sy;
    var out6 = (yz + wx) * sy;
    var out8 = (xz + wy) * sz;
    var out9 = (yz - wx) * sz;
    var out10 = (1 - (xx + yy)) * sz;
    out[0] = out0;
    out[1] = out1;
    out[2] = out2;
    out[3] = 0;
    out[4] = out4;
    out[5] = out5;
    out[6] = out6;
    out[7] = 0;
    out[8] = out8;
    out[9] = out9;
    out[10] = out10;
    out[11] = 0;
    out[12] = v[0] + ox - (out0 * ox + out4 * oy + out8 * oz);
    out[13] = v[1] + oy - (out1 * ox + out5 * oy + out9 * oz);
    out[14] = v[2] + oz - (out2 * ox + out6 * oy + out10 * oz);
    out[15] = 1;
    return out;
  }
  function fromQuat3(out, q) {
    var x = q[0], y = q[1], z = q[2], w = q[3];
    var x2 = x + x;
    var y2 = y + y;
    var z2 = z + z;
    var xx = x * x2;
    var yx = y * x2;
    var yy = y * y2;
    var zx = z * x2;
    var zy = z * y2;
    var zz = z * z2;
    var wx = w * x2;
    var wy = w * y2;
    var wz = w * z2;
    out[0] = 1 - yy - zz;
    out[1] = yx + wz;
    out[2] = zx - wy;
    out[3] = 0;
    out[4] = yx - wz;
    out[5] = 1 - xx - zz;
    out[6] = zy + wx;
    out[7] = 0;
    out[8] = zx + wy;
    out[9] = zy - wx;
    out[10] = 1 - xx - yy;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
    return out;
  }
  function frustum(out, left, right, bottom, top, near, far) {
    var rl = 1 / (right - left);
    var tb = 1 / (top - bottom);
    var nf = 1 / (near - far);
    out[0] = near * 2 * rl;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = near * 2 * tb;
    out[6] = 0;
    out[7] = 0;
    out[8] = (right + left) * rl;
    out[9] = (top + bottom) * tb;
    out[10] = (far + near) * nf;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = far * near * 2 * nf;
    out[15] = 0;
    return out;
  }
  function perspectiveNO(out, fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf;
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[15] = 0;
    if (far != null && far !== Infinity) {
      nf = 1 / (near - far);
      out[10] = (far + near) * nf;
      out[14] = 2 * far * near * nf;
    } else {
      out[10] = -1;
      out[14] = -2 * near;
    }
    return out;
  }
  var perspective = perspectiveNO;
  function perspectiveZO(out, fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf;
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[15] = 0;
    if (far != null && far !== Infinity) {
      nf = 1 / (near - far);
      out[10] = far * nf;
      out[14] = far * near * nf;
    } else {
      out[10] = -1;
      out[14] = -near;
    }
    return out;
  }
  function perspectiveFromFieldOfView(out, fov, near, far) {
    var upTan = Math.tan(fov.upDegrees * Math.PI / 180);
    var downTan = Math.tan(fov.downDegrees * Math.PI / 180);
    var leftTan = Math.tan(fov.leftDegrees * Math.PI / 180);
    var rightTan = Math.tan(fov.rightDegrees * Math.PI / 180);
    var xScale = 2 / (leftTan + rightTan);
    var yScale = 2 / (upTan + downTan);
    out[0] = xScale;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = yScale;
    out[6] = 0;
    out[7] = 0;
    out[8] = -((leftTan - rightTan) * xScale * 0.5);
    out[9] = (upTan - downTan) * yScale * 0.5;
    out[10] = far / (near - far);
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = far * near / (near - far);
    out[15] = 0;
    return out;
  }
  function orthoNO(out, left, right, bottom, top, near, far) {
    var lr = 1 / (left - right);
    var bt = 1 / (bottom - top);
    var nf = 1 / (near - far);
    out[0] = -2 * lr;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = -2 * bt;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = 2 * nf;
    out[11] = 0;
    out[12] = (left + right) * lr;
    out[13] = (top + bottom) * bt;
    out[14] = (far + near) * nf;
    out[15] = 1;
    return out;
  }
  var ortho = orthoNO;
  function orthoZO(out, left, right, bottom, top, near, far) {
    var lr = 1 / (left - right);
    var bt = 1 / (bottom - top);
    var nf = 1 / (near - far);
    out[0] = -2 * lr;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = -2 * bt;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = nf;
    out[11] = 0;
    out[12] = (left + right) * lr;
    out[13] = (top + bottom) * bt;
    out[14] = near * nf;
    out[15] = 1;
    return out;
  }
  function lookAt(out, eye, center, up) {
    var x0, x1, x2, y0, y1, y2, z0, z1, z2, len4;
    var eyex = eye[0];
    var eyey = eye[1];
    var eyez = eye[2];
    var upx = up[0];
    var upy = up[1];
    var upz = up[2];
    var centerx = center[0];
    var centery = center[1];
    var centerz = center[2];
    if (Math.abs(eyex - centerx) < EPSILON && Math.abs(eyey - centery) < EPSILON && Math.abs(eyez - centerz) < EPSILON) {
      return identity2(out);
    }
    z0 = eyex - centerx;
    z1 = eyey - centery;
    z2 = eyez - centerz;
    len4 = 1 / Math.hypot(z0, z1, z2);
    z0 *= len4;
    z1 *= len4;
    z2 *= len4;
    x0 = upy * z2 - upz * z1;
    x1 = upz * z0 - upx * z2;
    x2 = upx * z1 - upy * z0;
    len4 = Math.hypot(x0, x1, x2);
    if (!len4) {
      x0 = 0;
      x1 = 0;
      x2 = 0;
    } else {
      len4 = 1 / len4;
      x0 *= len4;
      x1 *= len4;
      x2 *= len4;
    }
    y0 = z1 * x2 - z2 * x1;
    y1 = z2 * x0 - z0 * x2;
    y2 = z0 * x1 - z1 * x0;
    len4 = Math.hypot(y0, y1, y2);
    if (!len4) {
      y0 = 0;
      y1 = 0;
      y2 = 0;
    } else {
      len4 = 1 / len4;
      y0 *= len4;
      y1 *= len4;
      y2 *= len4;
    }
    out[0] = x0;
    out[1] = y0;
    out[2] = z0;
    out[3] = 0;
    out[4] = x1;
    out[5] = y1;
    out[6] = z1;
    out[7] = 0;
    out[8] = x2;
    out[9] = y2;
    out[10] = z2;
    out[11] = 0;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1;
    return out;
  }
  function targetTo(out, eye, target, up) {
    var eyex = eye[0], eyey = eye[1], eyez = eye[2], upx = up[0], upy = up[1], upz = up[2];
    var z0 = eyex - target[0], z1 = eyey - target[1], z2 = eyez - target[2];
    var len4 = z0 * z0 + z1 * z1 + z2 * z2;
    if (len4 > 0) {
      len4 = 1 / Math.sqrt(len4);
      z0 *= len4;
      z1 *= len4;
      z2 *= len4;
    }
    var x0 = upy * z2 - upz * z1, x1 = upz * z0 - upx * z2, x2 = upx * z1 - upy * z0;
    len4 = x0 * x0 + x1 * x1 + x2 * x2;
    if (len4 > 0) {
      len4 = 1 / Math.sqrt(len4);
      x0 *= len4;
      x1 *= len4;
      x2 *= len4;
    }
    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = 0;
    out[4] = z1 * x2 - z2 * x1;
    out[5] = z2 * x0 - z0 * x2;
    out[6] = z0 * x1 - z1 * x0;
    out[7] = 0;
    out[8] = z0;
    out[9] = z1;
    out[10] = z2;
    out[11] = 0;
    out[12] = eyex;
    out[13] = eyey;
    out[14] = eyez;
    out[15] = 1;
    return out;
  }
  function str2(a) {
    return "mat4(" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + ", " + a[4] + ", " + a[5] + ", " + a[6] + ", " + a[7] + ", " + a[8] + ", " + a[9] + ", " + a[10] + ", " + a[11] + ", " + a[12] + ", " + a[13] + ", " + a[14] + ", " + a[15] + ")";
  }
  function frob2(a) {
    return Math.hypot(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
  }
  function add2(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];
    out[6] = a[6] + b[6];
    out[7] = a[7] + b[7];
    out[8] = a[8] + b[8];
    out[9] = a[9] + b[9];
    out[10] = a[10] + b[10];
    out[11] = a[11] + b[11];
    out[12] = a[12] + b[12];
    out[13] = a[13] + b[13];
    out[14] = a[14] + b[14];
    out[15] = a[15] + b[15];
    return out;
  }
  function subtract2(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    out[4] = a[4] - b[4];
    out[5] = a[5] - b[5];
    out[6] = a[6] - b[6];
    out[7] = a[7] - b[7];
    out[8] = a[8] - b[8];
    out[9] = a[9] - b[9];
    out[10] = a[10] - b[10];
    out[11] = a[11] - b[11];
    out[12] = a[12] - b[12];
    out[13] = a[13] - b[13];
    out[14] = a[14] - b[14];
    out[15] = a[15] - b[15];
    return out;
  }
  function multiplyScalar2(out, a, b) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b;
    out[6] = a[6] * b;
    out[7] = a[7] * b;
    out[8] = a[8] * b;
    out[9] = a[9] * b;
    out[10] = a[10] * b;
    out[11] = a[11] * b;
    out[12] = a[12] * b;
    out[13] = a[13] * b;
    out[14] = a[14] * b;
    out[15] = a[15] * b;
    return out;
  }
  function multiplyScalarAndAdd2(out, a, b, scale5) {
    out[0] = a[0] + b[0] * scale5;
    out[1] = a[1] + b[1] * scale5;
    out[2] = a[2] + b[2] * scale5;
    out[3] = a[3] + b[3] * scale5;
    out[4] = a[4] + b[4] * scale5;
    out[5] = a[5] + b[5] * scale5;
    out[6] = a[6] + b[6] * scale5;
    out[7] = a[7] + b[7] * scale5;
    out[8] = a[8] + b[8] * scale5;
    out[9] = a[9] + b[9] * scale5;
    out[10] = a[10] + b[10] * scale5;
    out[11] = a[11] + b[11] * scale5;
    out[12] = a[12] + b[12] * scale5;
    out[13] = a[13] + b[13] * scale5;
    out[14] = a[14] + b[14] * scale5;
    out[15] = a[15] + b[15] * scale5;
    return out;
  }
  function exactEquals2(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3] && a[4] === b[4] && a[5] === b[5] && a[6] === b[6] && a[7] === b[7] && a[8] === b[8] && a[9] === b[9] && a[10] === b[10] && a[11] === b[11] && a[12] === b[12] && a[13] === b[13] && a[14] === b[14] && a[15] === b[15];
  }
  function equals2(a, b) {
    var a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    var a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    var a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11];
    var a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
    var b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    var b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    var b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11];
    var b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
    return Math.abs(a0 - b0) <= EPSILON * Math.max(1, Math.abs(a0), Math.abs(b0)) && Math.abs(a1 - b1) <= EPSILON * Math.max(1, Math.abs(a1), Math.abs(b1)) && Math.abs(a2 - b2) <= EPSILON * Math.max(1, Math.abs(a2), Math.abs(b2)) && Math.abs(a3 - b3) <= EPSILON * Math.max(1, Math.abs(a3), Math.abs(b3)) && Math.abs(a4 - b4) <= EPSILON * Math.max(1, Math.abs(a4), Math.abs(b4)) && Math.abs(a5 - b5) <= EPSILON * Math.max(1, Math.abs(a5), Math.abs(b5)) && Math.abs(a6 - b6) <= EPSILON * Math.max(1, Math.abs(a6), Math.abs(b6)) && Math.abs(a7 - b7) <= EPSILON * Math.max(1, Math.abs(a7), Math.abs(b7)) && Math.abs(a8 - b8) <= EPSILON * Math.max(1, Math.abs(a8), Math.abs(b8)) && Math.abs(a9 - b9) <= EPSILON * Math.max(1, Math.abs(a9), Math.abs(b9)) && Math.abs(a10 - b10) <= EPSILON * Math.max(1, Math.abs(a10), Math.abs(b10)) && Math.abs(a11 - b11) <= EPSILON * Math.max(1, Math.abs(a11), Math.abs(b11)) && Math.abs(a12 - b12) <= EPSILON * Math.max(1, Math.abs(a12), Math.abs(b12)) && Math.abs(a13 - b13) <= EPSILON * Math.max(1, Math.abs(a13), Math.abs(b13)) && Math.abs(a14 - b14) <= EPSILON * Math.max(1, Math.abs(a14), Math.abs(b14)) && Math.abs(a15 - b15) <= EPSILON * Math.max(1, Math.abs(a15), Math.abs(b15));
  }
  var mul2 = multiply2;
  var sub2 = subtract2;

  // node_modules/gl-matrix/esm/quat.js
  var quat_exports = {};
  __export(quat_exports, {
    add: () => add4,
    calculateW: () => calculateW,
    clone: () => clone4,
    conjugate: () => conjugate,
    copy: () => copy4,
    create: () => create5,
    dot: () => dot3,
    equals: () => equals4,
    exactEquals: () => exactEquals4,
    exp: () => exp,
    fromEuler: () => fromEuler,
    fromMat3: () => fromMat3,
    fromValues: () => fromValues5,
    getAngle: () => getAngle,
    getAxisAngle: () => getAxisAngle,
    identity: () => identity3,
    invert: () => invert3,
    len: () => len3,
    length: () => length3,
    lerp: () => lerp2,
    ln: () => ln,
    mul: () => mul4,
    multiply: () => multiply4,
    normalize: () => normalize3,
    pow: () => pow,
    random: () => random2,
    rotateX: () => rotateX2,
    rotateY: () => rotateY2,
    rotateZ: () => rotateZ2,
    rotationTo: () => rotationTo,
    scale: () => scale4,
    set: () => set4,
    setAxes: () => setAxes,
    setAxisAngle: () => setAxisAngle,
    slerp: () => slerp,
    sqlerp: () => sqlerp,
    sqrLen: () => sqrLen2,
    squaredLength: () => squaredLength2,
    str: () => str4
  });

  // node_modules/gl-matrix/esm/vec3.js
  function create3() {
    var out = new ARRAY_TYPE(3);
    if (ARRAY_TYPE != Float32Array) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    }
    return out;
  }
  function length(a) {
    var x = a[0];
    var y = a[1];
    var z = a[2];
    return Math.hypot(x, y, z);
  }
  function fromValues3(x, y, z) {
    var out = new ARRAY_TYPE(3);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    return out;
  }
  function normalize(out, a) {
    var x = a[0];
    var y = a[1];
    var z = a[2];
    var len4 = x * x + y * y + z * z;
    if (len4 > 0) {
      len4 = 1 / Math.sqrt(len4);
    }
    out[0] = a[0] * len4;
    out[1] = a[1] * len4;
    out[2] = a[2] * len4;
    return out;
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function cross(out, a, b) {
    var ax = a[0], ay = a[1], az = a[2];
    var bx = b[0], by = b[1], bz = b[2];
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  }
  var len = length;
  var forEach = function() {
    var vec = create3();
    return function(a, stride, offset, count, fn, arg) {
      var i, l;
      if (!stride) {
        stride = 3;
      }
      if (!offset) {
        offset = 0;
      }
      if (count) {
        l = Math.min(count * stride + offset, a.length);
      } else {
        l = a.length;
      }
      for (i = offset; i < l; i += stride) {
        vec[0] = a[i];
        vec[1] = a[i + 1];
        vec[2] = a[i + 2];
        fn(vec, vec, arg);
        a[i] = vec[0];
        a[i + 1] = vec[1];
        a[i + 2] = vec[2];
      }
      return a;
    };
  }();

  // node_modules/gl-matrix/esm/vec4.js
  var vec4_exports = {};
  __export(vec4_exports, {
    add: () => add3,
    ceil: () => ceil,
    clone: () => clone3,
    copy: () => copy3,
    create: () => create4,
    cross: () => cross2,
    dist: () => dist,
    distance: () => distance,
    div: () => div,
    divide: () => divide,
    dot: () => dot2,
    equals: () => equals3,
    exactEquals: () => exactEquals3,
    floor: () => floor,
    forEach: () => forEach2,
    fromValues: () => fromValues4,
    inverse: () => inverse,
    len: () => len2,
    length: () => length2,
    lerp: () => lerp,
    max: () => max,
    min: () => min,
    mul: () => mul3,
    multiply: () => multiply3,
    negate: () => negate,
    normalize: () => normalize2,
    random: () => random,
    round: () => round,
    scale: () => scale3,
    scaleAndAdd: () => scaleAndAdd,
    set: () => set3,
    sqrDist: () => sqrDist,
    sqrLen: () => sqrLen,
    squaredDistance: () => squaredDistance,
    squaredLength: () => squaredLength,
    str: () => str3,
    sub: () => sub3,
    subtract: () => subtract3,
    transformMat4: () => transformMat4,
    transformQuat: () => transformQuat,
    zero: () => zero
  });
  function create4() {
    var out = new ARRAY_TYPE(4);
    if (ARRAY_TYPE != Float32Array) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
    }
    return out;
  }
  function clone3(a) {
    var out = new ARRAY_TYPE(4);
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    return out;
  }
  function fromValues4(x, y, z, w) {
    var out = new ARRAY_TYPE(4);
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = w;
    return out;
  }
  function copy3(out, a) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    return out;
  }
  function set3(out, x, y, z, w) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = w;
    return out;
  }
  function add3(out, a, b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    return out;
  }
  function subtract3(out, a, b) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    return out;
  }
  function multiply3(out, a, b) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
    out[3] = a[3] * b[3];
    return out;
  }
  function divide(out, a, b) {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];
    out[2] = a[2] / b[2];
    out[3] = a[3] / b[3];
    return out;
  }
  function ceil(out, a) {
    out[0] = Math.ceil(a[0]);
    out[1] = Math.ceil(a[1]);
    out[2] = Math.ceil(a[2]);
    out[3] = Math.ceil(a[3]);
    return out;
  }
  function floor(out, a) {
    out[0] = Math.floor(a[0]);
    out[1] = Math.floor(a[1]);
    out[2] = Math.floor(a[2]);
    out[3] = Math.floor(a[3]);
    return out;
  }
  function min(out, a, b) {
    out[0] = Math.min(a[0], b[0]);
    out[1] = Math.min(a[1], b[1]);
    out[2] = Math.min(a[2], b[2]);
    out[3] = Math.min(a[3], b[3]);
    return out;
  }
  function max(out, a, b) {
    out[0] = Math.max(a[0], b[0]);
    out[1] = Math.max(a[1], b[1]);
    out[2] = Math.max(a[2], b[2]);
    out[3] = Math.max(a[3], b[3]);
    return out;
  }
  function round(out, a) {
    out[0] = Math.round(a[0]);
    out[1] = Math.round(a[1]);
    out[2] = Math.round(a[2]);
    out[3] = Math.round(a[3]);
    return out;
  }
  function scale3(out, a, b) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    return out;
  }
  function scaleAndAdd(out, a, b, scale5) {
    out[0] = a[0] + b[0] * scale5;
    out[1] = a[1] + b[1] * scale5;
    out[2] = a[2] + b[2] * scale5;
    out[3] = a[3] + b[3] * scale5;
    return out;
  }
  function distance(a, b) {
    var x = b[0] - a[0];
    var y = b[1] - a[1];
    var z = b[2] - a[2];
    var w = b[3] - a[3];
    return Math.hypot(x, y, z, w);
  }
  function squaredDistance(a, b) {
    var x = b[0] - a[0];
    var y = b[1] - a[1];
    var z = b[2] - a[2];
    var w = b[3] - a[3];
    return x * x + y * y + z * z + w * w;
  }
  function length2(a) {
    var x = a[0];
    var y = a[1];
    var z = a[2];
    var w = a[3];
    return Math.hypot(x, y, z, w);
  }
  function squaredLength(a) {
    var x = a[0];
    var y = a[1];
    var z = a[2];
    var w = a[3];
    return x * x + y * y + z * z + w * w;
  }
  function negate(out, a) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = -a[3];
    return out;
  }
  function inverse(out, a) {
    out[0] = 1 / a[0];
    out[1] = 1 / a[1];
    out[2] = 1 / a[2];
    out[3] = 1 / a[3];
    return out;
  }
  function normalize2(out, a) {
    var x = a[0];
    var y = a[1];
    var z = a[2];
    var w = a[3];
    var len4 = x * x + y * y + z * z + w * w;
    if (len4 > 0) {
      len4 = 1 / Math.sqrt(len4);
    }
    out[0] = x * len4;
    out[1] = y * len4;
    out[2] = z * len4;
    out[3] = w * len4;
    return out;
  }
  function dot2(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  }
  function cross2(out, u, v, w) {
    var A = v[0] * w[1] - v[1] * w[0], B = v[0] * w[2] - v[2] * w[0], C = v[0] * w[3] - v[3] * w[0], D = v[1] * w[2] - v[2] * w[1], E = v[1] * w[3] - v[3] * w[1], F = v[2] * w[3] - v[3] * w[2];
    var G = u[0];
    var H = u[1];
    var I = u[2];
    var J = u[3];
    out[0] = H * F - I * E + J * D;
    out[1] = -(G * F) + I * C - J * B;
    out[2] = G * E - H * C + J * A;
    out[3] = -(G * D) + H * B - I * A;
    return out;
  }
  function lerp(out, a, b, t) {
    var ax = a[0];
    var ay = a[1];
    var az = a[2];
    var aw = a[3];
    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);
    out[2] = az + t * (b[2] - az);
    out[3] = aw + t * (b[3] - aw);
    return out;
  }
  function random(out, scale5) {
    scale5 = scale5 || 1;
    var v1, v2, v3, v4;
    var s1, s2;
    do {
      v1 = RANDOM() * 2 - 1;
      v2 = RANDOM() * 2 - 1;
      s1 = v1 * v1 + v2 * v2;
    } while (s1 >= 1);
    do {
      v3 = RANDOM() * 2 - 1;
      v4 = RANDOM() * 2 - 1;
      s2 = v3 * v3 + v4 * v4;
    } while (s2 >= 1);
    var d = Math.sqrt((1 - s1) / s2);
    out[0] = scale5 * v1;
    out[1] = scale5 * v2;
    out[2] = scale5 * v3 * d;
    out[3] = scale5 * v4 * d;
    return out;
  }
  function transformMat4(out, a, m) {
    var x = a[0], y = a[1], z = a[2], w = a[3];
    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
    out[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
    return out;
  }
  function transformQuat(out, a, q) {
    var x = a[0], y = a[1], z = a[2];
    var qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    var ix = qw * x + qy * z - qz * y;
    var iy = qw * y + qz * x - qx * z;
    var iz = qw * z + qx * y - qy * x;
    var iw = -qx * x - qy * y - qz * z;
    out[0] = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    out[1] = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    out[2] = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    out[3] = a[3];
    return out;
  }
  function zero(out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    return out;
  }
  function str3(a) {
    return "vec4(" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + ")";
  }
  function exactEquals3(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }
  function equals3(a, b) {
    var a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    var b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    return Math.abs(a0 - b0) <= EPSILON * Math.max(1, Math.abs(a0), Math.abs(b0)) && Math.abs(a1 - b1) <= EPSILON * Math.max(1, Math.abs(a1), Math.abs(b1)) && Math.abs(a2 - b2) <= EPSILON * Math.max(1, Math.abs(a2), Math.abs(b2)) && Math.abs(a3 - b3) <= EPSILON * Math.max(1, Math.abs(a3), Math.abs(b3));
  }
  var sub3 = subtract3;
  var mul3 = multiply3;
  var div = divide;
  var dist = distance;
  var sqrDist = squaredDistance;
  var len2 = length2;
  var sqrLen = squaredLength;
  var forEach2 = function() {
    var vec = create4();
    return function(a, stride, offset, count, fn, arg) {
      var i, l;
      if (!stride) {
        stride = 4;
      }
      if (!offset) {
        offset = 0;
      }
      if (count) {
        l = Math.min(count * stride + offset, a.length);
      } else {
        l = a.length;
      }
      for (i = offset; i < l; i += stride) {
        vec[0] = a[i];
        vec[1] = a[i + 1];
        vec[2] = a[i + 2];
        vec[3] = a[i + 3];
        fn(vec, vec, arg);
        a[i] = vec[0];
        a[i + 1] = vec[1];
        a[i + 2] = vec[2];
        a[i + 3] = vec[3];
      }
      return a;
    };
  }();

  // node_modules/gl-matrix/esm/quat.js
  function create5() {
    var out = new ARRAY_TYPE(4);
    if (ARRAY_TYPE != Float32Array) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    }
    out[3] = 1;
    return out;
  }
  function identity3(out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 1;
    return out;
  }
  function setAxisAngle(out, axis, rad) {
    rad = rad * 0.5;
    var s = Math.sin(rad);
    out[0] = s * axis[0];
    out[1] = s * axis[1];
    out[2] = s * axis[2];
    out[3] = Math.cos(rad);
    return out;
  }
  function getAxisAngle(out_axis, q) {
    var rad = Math.acos(q[3]) * 2;
    var s = Math.sin(rad / 2);
    if (s > EPSILON) {
      out_axis[0] = q[0] / s;
      out_axis[1] = q[1] / s;
      out_axis[2] = q[2] / s;
    } else {
      out_axis[0] = 1;
      out_axis[1] = 0;
      out_axis[2] = 0;
    }
    return rad;
  }
  function getAngle(a, b) {
    var dotproduct = dot3(a, b);
    return Math.acos(2 * dotproduct * dotproduct - 1);
  }
  function multiply4(out, a, b) {
    var ax = a[0], ay = a[1], az = a[2], aw = a[3];
    var bx = b[0], by = b[1], bz = b[2], bw = b[3];
    out[0] = ax * bw + aw * bx + ay * bz - az * by;
    out[1] = ay * bw + aw * by + az * bx - ax * bz;
    out[2] = az * bw + aw * bz + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
  }
  function rotateX2(out, a, rad) {
    rad *= 0.5;
    var ax = a[0], ay = a[1], az = a[2], aw = a[3];
    var bx = Math.sin(rad), bw = Math.cos(rad);
    out[0] = ax * bw + aw * bx;
    out[1] = ay * bw + az * bx;
    out[2] = az * bw - ay * bx;
    out[3] = aw * bw - ax * bx;
    return out;
  }
  function rotateY2(out, a, rad) {
    rad *= 0.5;
    var ax = a[0], ay = a[1], az = a[2], aw = a[3];
    var by = Math.sin(rad), bw = Math.cos(rad);
    out[0] = ax * bw - az * by;
    out[1] = ay * bw + aw * by;
    out[2] = az * bw + ax * by;
    out[3] = aw * bw - ay * by;
    return out;
  }
  function rotateZ2(out, a, rad) {
    rad *= 0.5;
    var ax = a[0], ay = a[1], az = a[2], aw = a[3];
    var bz = Math.sin(rad), bw = Math.cos(rad);
    out[0] = ax * bw + ay * bz;
    out[1] = ay * bw - ax * bz;
    out[2] = az * bw + aw * bz;
    out[3] = aw * bw - az * bz;
    return out;
  }
  function calculateW(out, a) {
    var x = a[0], y = a[1], z = a[2];
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = Math.sqrt(Math.abs(1 - x * x - y * y - z * z));
    return out;
  }
  function exp(out, a) {
    var x = a[0], y = a[1], z = a[2], w = a[3];
    var r = Math.sqrt(x * x + y * y + z * z);
    var et = Math.exp(w);
    var s = r > 0 ? et * Math.sin(r) / r : 0;
    out[0] = x * s;
    out[1] = y * s;
    out[2] = z * s;
    out[3] = et * Math.cos(r);
    return out;
  }
  function ln(out, a) {
    var x = a[0], y = a[1], z = a[2], w = a[3];
    var r = Math.sqrt(x * x + y * y + z * z);
    var t = r > 0 ? Math.atan2(r, w) / r : 0;
    out[0] = x * t;
    out[1] = y * t;
    out[2] = z * t;
    out[3] = 0.5 * Math.log(x * x + y * y + z * z + w * w);
    return out;
  }
  function pow(out, a, b) {
    ln(out, a);
    scale4(out, out, b);
    exp(out, out);
    return out;
  }
  function slerp(out, a, b, t) {
    var ax = a[0], ay = a[1], az = a[2], aw = a[3];
    var bx = b[0], by = b[1], bz = b[2], bw = b[3];
    var omega, cosom, sinom, scale0, scale1;
    cosom = ax * bx + ay * by + az * bz + aw * bw;
    if (cosom < 0) {
      cosom = -cosom;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    if (1 - cosom > EPSILON) {
      omega = Math.acos(cosom);
      sinom = Math.sin(omega);
      scale0 = Math.sin((1 - t) * omega) / sinom;
      scale1 = Math.sin(t * omega) / sinom;
    } else {
      scale0 = 1 - t;
      scale1 = t;
    }
    out[0] = scale0 * ax + scale1 * bx;
    out[1] = scale0 * ay + scale1 * by;
    out[2] = scale0 * az + scale1 * bz;
    out[3] = scale0 * aw + scale1 * bw;
    return out;
  }
  function random2(out) {
    var u1 = RANDOM();
    var u2 = RANDOM();
    var u3 = RANDOM();
    var sqrt1MinusU1 = Math.sqrt(1 - u1);
    var sqrtU1 = Math.sqrt(u1);
    out[0] = sqrt1MinusU1 * Math.sin(2 * Math.PI * u2);
    out[1] = sqrt1MinusU1 * Math.cos(2 * Math.PI * u2);
    out[2] = sqrtU1 * Math.sin(2 * Math.PI * u3);
    out[3] = sqrtU1 * Math.cos(2 * Math.PI * u3);
    return out;
  }
  function invert3(out, a) {
    var a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    var dot4 = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    var invDot = dot4 ? 1 / dot4 : 0;
    out[0] = -a0 * invDot;
    out[1] = -a1 * invDot;
    out[2] = -a2 * invDot;
    out[3] = a3 * invDot;
    return out;
  }
  function conjugate(out, a) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = a[3];
    return out;
  }
  function fromMat3(out, m) {
    var fTrace = m[0] + m[4] + m[8];
    var fRoot;
    if (fTrace > 0) {
      fRoot = Math.sqrt(fTrace + 1);
      out[3] = 0.5 * fRoot;
      fRoot = 0.5 / fRoot;
      out[0] = (m[5] - m[7]) * fRoot;
      out[1] = (m[6] - m[2]) * fRoot;
      out[2] = (m[1] - m[3]) * fRoot;
    } else {
      var i = 0;
      if (m[4] > m[0])
        i = 1;
      if (m[8] > m[i * 3 + i])
        i = 2;
      var j = (i + 1) % 3;
      var k = (i + 2) % 3;
      fRoot = Math.sqrt(m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k] + 1);
      out[i] = 0.5 * fRoot;
      fRoot = 0.5 / fRoot;
      out[3] = (m[j * 3 + k] - m[k * 3 + j]) * fRoot;
      out[j] = (m[j * 3 + i] + m[i * 3 + j]) * fRoot;
      out[k] = (m[k * 3 + i] + m[i * 3 + k]) * fRoot;
    }
    return out;
  }
  function fromEuler(out, x, y, z) {
    var halfToRad = 0.5 * Math.PI / 180;
    x *= halfToRad;
    y *= halfToRad;
    z *= halfToRad;
    var sx = Math.sin(x);
    var cx = Math.cos(x);
    var sy = Math.sin(y);
    var cy = Math.cos(y);
    var sz = Math.sin(z);
    var cz = Math.cos(z);
    out[0] = sx * cy * cz - cx * sy * sz;
    out[1] = cx * sy * cz + sx * cy * sz;
    out[2] = cx * cy * sz - sx * sy * cz;
    out[3] = cx * cy * cz + sx * sy * sz;
    return out;
  }
  function str4(a) {
    return "quat(" + a[0] + ", " + a[1] + ", " + a[2] + ", " + a[3] + ")";
  }
  var clone4 = clone3;
  var fromValues5 = fromValues4;
  var copy4 = copy3;
  var set4 = set3;
  var add4 = add3;
  var mul4 = multiply4;
  var scale4 = scale3;
  var dot3 = dot2;
  var lerp2 = lerp;
  var length3 = length2;
  var len3 = length3;
  var squaredLength2 = squaredLength;
  var sqrLen2 = squaredLength2;
  var normalize3 = normalize2;
  var exactEquals4 = exactEquals3;
  var equals4 = equals3;
  var rotationTo = function() {
    var tmpvec3 = create3();
    var xUnitVec3 = fromValues3(1, 0, 0);
    var yUnitVec3 = fromValues3(0, 1, 0);
    return function(out, a, b) {
      var dot4 = dot(a, b);
      if (dot4 < -0.999999) {
        cross(tmpvec3, xUnitVec3, a);
        if (len(tmpvec3) < 1e-6)
          cross(tmpvec3, yUnitVec3, a);
        normalize(tmpvec3, tmpvec3);
        setAxisAngle(out, tmpvec3, Math.PI);
        return out;
      } else if (dot4 > 0.999999) {
        out[0] = 0;
        out[1] = 0;
        out[2] = 0;
        out[3] = 1;
        return out;
      } else {
        cross(tmpvec3, a, b);
        out[0] = tmpvec3[0];
        out[1] = tmpvec3[1];
        out[2] = tmpvec3[2];
        out[3] = 1 + dot4;
        return normalize3(out, out);
      }
    };
  }();
  var sqlerp = function() {
    var temp1 = create5();
    var temp2 = create5();
    return function(out, a, b, c, d, t) {
      slerp(temp1, a, d, t);
      slerp(temp2, b, c, t);
      slerp(out, temp1, temp2, 2 * t * (1 - t));
      return out;
    };
  }();
  var setAxes = function() {
    var matr = create();
    return function(out, view, right, up) {
      matr[0] = right[0];
      matr[3] = right[1];
      matr[6] = right[2];
      matr[1] = up[0];
      matr[4] = up[1];
      matr[7] = up[2];
      matr[2] = -view[0];
      matr[5] = -view[1];
      matr[8] = -view[2];
      return normalize3(out, fromMat3(out, matr));
    };
  }();

  // src/demos/sdf-physics.ts
  var PALETTE = [
    [0.19215686274509805, 0.2235294117647059, 0.23529411764705882, 1],
    [0.12941176470588237, 0.4627450980392157, 1, 1],
    [0.2, 0.6313725490196078, 0.9921568627450981, 1],
    [0.9921568627450981, 0.792156862745098, 0.25098039215686274, 1],
    [0.9686274509803922, 0.596078431372549, 0.1411764705882353, 1]
  ];
  var SWEEP_RADIUS = 0.03;
  var DONUT_RADIUS = 0.07;
  var PARTICLE_RADIUS = SWEEP_RADIUS + DONUT_RADIUS;
  var SCAN_THREADS = 256;
  var SCAN_ITEMS = 4;
  var NUM_PARTICLE_BLOCKS = 1;
  var NUM_PARTICLES = NUM_PARTICLE_BLOCKS * SCAN_THREADS * SCAN_ITEMS;
  var PARTICLE_WORKGROUP_SIZE = SCAN_THREADS;
  var RAY_STEPS = 32;
  var RAY_TOLER = 1e-3;
  var BG_RAY_STEPS = 256;
  var BG_RAY_TOLER = 1e-4;
  var BG_TMIN = 0;
  var BG_TMAX = 1e3;
  var BG_COLOR = PALETTE[0];
  var RADIUS_PADDING = 1.5;
  var DT = 0.1;
  var SUBSTEPS = 10;
  var SUB_DT = DT / SUBSTEPS;
  var JACOBI_POS = 0.25;
  var JACOBI_ROT = 0.25;
  var GRAVITY = 0.5;
  var POS_DAMPING = 0.05;
  var ROT_DAMPING = 0.01;
  var PARTICLE_INERTIA_TENSOR = mat3_exports.identity(mat3_exports.create());
  var PARTICLE_INV_INERTIA_TENSOR = mat3_exports.invert(mat3_exports.create(), PARTICLE_INERTIA_TENSOR);
  var MAX_BUCKET_SIZE = 16;
  var GRID_SPACING = 2 * PARTICLE_RADIUS;
  var COLLISION_TABLE_SIZE = NUM_PARTICLES;
  var HASH_VEC = [
    1,
    Math.ceil(Math.pow(COLLISION_TABLE_SIZE, 1 / 3)),
    Math.ceil(Math.pow(COLLISION_TABLE_SIZE, 2 / 3))
  ];
  var CONTACTS_PER_PARTICLE = 16;
  var CONTACT_TABLE_SIZE = CONTACTS_PER_PARTICLE * NUM_PARTICLES;
  var COMMON_SHADER_FUNCS = `
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

fn bucketHash (p:vec3<i32>) -> u32 {
  var h = (p.x * ${HASH_VEC[0]}) + (p.y * ${HASH_VEC[1]}) + (p.z * ${HASH_VEC[2]});
  if h < 0 {
    return ${COLLISION_TABLE_SIZE}u - (u32(-h) % ${COLLISION_TABLE_SIZE}u);
  } else {
    return u32(h) % ${COLLISION_TABLE_SIZE}u;
  }
}

fn particleBucket (p:vec3<f32>) -> vec3<i32> {
  return vec3<i32>(floor(p * ${(1 / GRID_SPACING).toFixed(3)}));
}

fn particleHash (p:vec3<f32>) -> u32 {
  return bucketHash(particleBucket(p));
} 
`;
  async function main() {
    const adapter = mustHave(await navigator.gpu.requestAdapter());
    const device = await adapter.requestDevice();
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    const canvas = makeCanvas();
    const context = mustHave(canvas.getContext("webgpu"));
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "opaque"
    });
    const depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    const backroundShader = device.createShaderModule({
      label: "bgRenderShader",
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
    result.color = vec4(${BG_COLOR.join(", ")});
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
    });
    const renderShader = device.createShaderModule({
      label: "particleRenderShader",
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
  var rayDirection = viewCenter + vec4(${RADIUS_PADDING * PARTICLE_RADIUS} * uv.x, ${RADIUS_PADDING * PARTICLE_RADIUS} * uv.y, 0., 0.);
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
  
  var tmid = length(rayDirectionInterp);
  var rayDirection = rayDirectionInterp / tmid;
  var rayDist = traceRay(rayOrigin, rayDirection, tmid - ${RADIUS_PADDING * PARTICLE_RADIUS}, tmid + ${RADIUS_PADDING * PARTICLE_RADIUS});
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
    });
    const PHYSICS_COMMON = `
const inertiaTensor = mat3x3<f32>(${Array.prototype.join.call(PARTICLE_INERTIA_TENSOR)});
const invInertiaTensor = mat3x3<f32>(${Array.prototype.join.call(PARTICLE_INV_INERTIA_TENSOR)});

fn qMultiply(a:vec4<f32>, b:vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y,
    a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
    a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  );
}

fn qMatrix (q:vec4<f32>) -> mat3x3<f32> {
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

  return mat3x3<f32>(
    1. - (yy + zz),
    xy + wz,
    xz - wy,

    xy - wz,
    1. - (xx + zz),
    yz + wx,

    xz + wy,
    yz - wx,
    1. - (xx + yy),
  );
}
`;
    const particlePredictShader = device.createShaderModule({
      label: "particlePredict",
      code: `
${PHYSICS_COMMON}

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
  v.y = v.y - ${(GRAVITY * SUB_DT).toFixed(3)};
  predictedPosition[id] = position[id] + v * ${SUB_DT.toFixed(3)};
  positionUpdate[id] = vec4<f32>(0.);

  var q = rotation[id];
  var omega = angVelocity[id].xyz;

  var nextOmega = omega - ${SUB_DT} * invInertiaTensor * cross(omega, inertiaTensor * omega);
  var nextQ = q + ${0.5 * SUB_DT} * qMultiply(vec4(omega, 0.), q);

  predictedRotation[id] = normalize(nextQ);
  rotationUpdate[id] = vec4<f32>(0.);
}`
    });
    const particleUpdateShader = device.createShaderModule({
      label: "particleUpdate",
      code: `
${PHYSICS_COMMON}

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
  var prevPosition = p.xyz;
  var nextPosition = predictedPosition[id].xyz + ${JACOBI_POS} * positionUpdate[id].xyz;
  velocity[id] = vec4(${Math.exp(-SUB_DT * POS_DAMPING)} * (nextPosition - prevPosition) * ${(1 / SUB_DT).toFixed(3)}, 0.);
  position[id] = vec4(nextPosition, p.w);

  var prevQ = rotation[id];
  var nextQ = normalize(predictedRotation[id] + ${JACOBI_ROT} * rotationUpdate[id]);

  var dQ = qMultiply(nextQ, vec4(-prevQ.xyz, prevQ.w));
  if dQ.w < 0. {
    angVelocity[id] = vec4(-${2 / SUB_DT * Math.exp(-SUB_DT * ROT_DAMPING)} * dQ.xyz, 0.);
  } else {
    angVelocity[id] = vec4(${2 / SUB_DT * Math.exp(-SUB_DT * ROT_DAMPING)} * dQ.xyz, 0.);
  }
  rotation[id] = nextQ;
}
`
    });
    const clearBufferPipeline = device.createComputePipeline({
      label: "clearBufferPipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "clearBufferShader",
          code: `
@binding(0) @group(0) var<storage, read_write> buffer : array<u32>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE}, 1, 1) fn clearGrids (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  buffer[globalVec.x] = 0u;
}`
        }),
        entryPoint: "clearGrids"
      }
    });
    const gridCountPipeline = device.createComputePipeline({
      label: "gridCountPipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "gridCountShader",
          code: `
${COMMON_SHADER_FUNCS}

@binding(0) @group(0) var<storage, read> positions : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read_write> hashCounts : array<atomic<u32>>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn countParticles (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;
  var bucket = particleHash(positions[id].xyz);
  atomicAdd(&hashCounts[bucket], 1u);
}`
        }),
        entryPoint: "countParticles"
      }
    });
    const gridCopyParticlePipeline = device.createComputePipeline({
      label: "gridCopyParticles",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "gridCopyShader",
          code: `
${COMMON_SHADER_FUNCS}

@binding(0) @group(0) var<storage, read> positions : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read_write> hashCounts : array<atomic<u32>>;
@binding(2) @group(0) var<storage, read_write> particleIds : array<u32>;

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn copyParticleIds (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;
  var bucket = particleHash(positions[id].xyz);
  var offset = atomicSub(&hashCounts[bucket], 1u) - 1u;
  particleIds[offset] = id;
}
`
        }),
        entryPoint: "copyParticleIds"
      }
    });
    const contactCommonCode = `
  ${COMMON_SHADER_FUNCS}

@binding(0) @group(0) var<storage, read> positions : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read> hashCounts : array<u32>;
@binding(2) @group(0) var<storage, read> particleIds : array<u32>;

struct BucketContents {
  ids : array<i32, ${MAX_BUCKET_SIZE}>,
  xyz : array<vec3<f32>, ${MAX_BUCKET_SIZE}>,
  count : u32,
}

fn readBucketNeighborhood (centerId:u32) -> array<array<array<BucketContents, 2>, 2>, 2> {
  var result : array<array<array<BucketContents, 2>, 2>, 2>;

  for (var i = 0; i < 2; i = i + 1) {
    for (var j = 0; j < 2; j = j + 1) {
      for (var k = 0; k < 2; k = k + 1) {
        var bucketId = (centerId + bucketHash(vec3<i32>(i, j, k))) % ${COLLISION_TABLE_SIZE}u;
        var bucketStart = hashCounts[bucketId];
        var bucketEnd = ${NUM_PARTICLES}u;
        if bucketId < ${COLLISION_TABLE_SIZE - 1} {
          bucketEnd = hashCounts[bucketId + 1];
        }
        result[i][j][k].count = min(bucketEnd - bucketStart, ${MAX_BUCKET_SIZE}u);
        for (var n = 0u; n < ${MAX_BUCKET_SIZE}u; n = n + 1u) {
          var p = bucketStart + n;
          if p >= bucketEnd {
            result[i][j][k].ids[n] = -1;
          } else {
            result[i][j][k].ids[n] = i32(particleIds[p]);
          }
        }
        for (var n = 0u; n < ${MAX_BUCKET_SIZE}u; n = n + 1u) {
          if (n >= result[i][j][k].count) {
            break;
          }
          result[i][j][k].xyz[n] = positions[result[i][j][k].ids[n]].xyz;
        }
      }
    }
  }

  return result;
}

fn testOverlap (a:vec3<f32>, b:vec3<f32>) -> f32 {
  var d = a - b;
  return dot(d, d) - ${4 * PARTICLE_RADIUS * PARTICLE_RADIUS};
}
`;
    const contactCountPipeline = device.createComputePipeline({
      label: "contactCountPipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "contactCountShader",
          code: `
${contactCommonCode}

@binding(3) @group(0) var<storage, read_write> contactCount : array<u32>;


fn countBucketContacts (a:BucketContents, b:BucketContents) -> u32 {
  var count = 0u;
  for (var i = 0u; i < ${MAX_BUCKET_SIZE}u; i = i + 1u) {
    if (i >= a.count) {
      break;
    }
    for (var j = 0u; j < ${MAX_BUCKET_SIZE}u; j = j + 1u) {
      if (j >= b.count) {
        break;
      }
      if (testOverlap(a.xyz[i], b.xyz[j]) <= 0.) {
        count = count + 1u;
      }
    }
  }
  return count;
}

fn countCenterContacts (a:BucketContents) -> u32 {
  var count = 0u;
  for (var i = 0u; i < ${MAX_BUCKET_SIZE}u; i = i + 1u) {
    if (i >= a.count) {
      break;
    }
    for (var j = 0u; j < i; j = j + 1u) {
      if (testOverlap(a.xyz[i], a.xyz[j]) <= 0.) {
        count = count + 1u;
      }
    }
  }
  return count;
}

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn countContacts (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;
  var buckets = readBucketNeighborhood(id);

  contactCount[id] = countCenterContacts(buckets[0][0][0]) +
    countBucketContacts(buckets[0][0][0], buckets[0][0][1]) +
    countBucketContacts(buckets[0][0][0], buckets[0][1][0]) +
    countBucketContacts(buckets[0][0][0], buckets[0][1][1]) +
    countBucketContacts(buckets[0][0][0], buckets[1][0][0]) +
    countBucketContacts(buckets[0][0][0], buckets[1][0][1]) +
    countBucketContacts(buckets[0][0][0], buckets[1][1][0]) +
    countBucketContacts(buckets[0][0][0], buckets[1][1][1]);
}`
        }),
        entryPoint: "countContacts"
      }
    });
    const contactListPipeline = device.createComputePipeline({
      label: "contactListPipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "contactListShader",
          code: `
${contactCommonCode}

@binding(3) @group(0) var<storage, read> contactCount : array<u32>;
@binding(4) @group(0) var<storage, read_write> contactList : array<vec2<i32>>;

fn emitBucketContacts (a:BucketContents, b:BucketContents, offset:u32) -> u32 {
  if (offset >= ${CONTACT_TABLE_SIZE}u) {
    return offset;
  }
  var shift = offset;
  for (var i = 0u; i < ${MAX_BUCKET_SIZE}u; i = i + 1u) {
    if (i >= a.count) {
      break;
    }
    for (var j = 0u; j < ${MAX_BUCKET_SIZE}u; j = j + 1u) {
      if (j >= b.count) {
        break;
      }
      if (testOverlap(a.xyz[i], b.xyz[j]) <= 0.) {
        contactList[shift] = vec2<i32>(a.ids[i], b.ids[j]);
        shift = shift + 1u;
        if (shift >= ${CONTACT_TABLE_SIZE}u) {
          return shift;
        }
      }
    }
  }
  return shift;
}

fn emitCenterContacts (a:BucketContents, offset:u32) -> u32 {
  if (offset >= ${CONTACT_TABLE_SIZE}u) {
    return offset;
  }
  var shift = offset;
  for (var i = 1u; i < ${MAX_BUCKET_SIZE}u; i = i + 1u) {
    if (i >= a.count) {
      break;
    }
    for (var j = 0u; j < i; j = j + 1u) {
      if (testOverlap(a.xyz[i], a.xyz[j]) <= 0.) {
        contactList[shift] = vec2<i32>(a.ids[i], a.ids[j]);
        shift = shift + 1u;
        if (shift >= ${CONTACT_TABLE_SIZE}u) {
          return shift;
        }
      }
    }
  }
  return shift;
}

@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn countContacts (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;
  var buckets = readBucketNeighborhood(id);
  var offset = 0u;
  if id > 0u {
    offset = contactCount[id - 1u];
  }

  offset = emitCenterContacts(buckets[0][0][0], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[0][0][1], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[0][1][0], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[0][1][1], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[1][0][0], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[1][0][1], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[1][1][0], offset);
  offset = emitBucketContacts(buckets[0][0][0], buckets[1][1][1], offset);  
}`
        }),
        entryPoint: "countContacts"
      }
    });
    const debugContactShader = device.createShaderModule({
      label: "renderContactShader",
      code: `
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
@binding(1) @group(1) var<storage, read> contactId : array<u32>;

@vertex fn vertMain (@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
  var pos = position[contactId[vertexIndex]].xyz;
  return uniforms.proj * uniforms.view * vec4(pos, 1.);
}

@fragment fn fragMain () -> @location(0) vec4<f32> {
  return vec4(1., 0., 0., 1.);
}
`
    });
    const solveTerrainPositionPipeline = device.createComputePipeline({
      label: "solveTerrainPositionPipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "solveTerrainPositionShader",
          code: `
${COMMON_SHADER_FUNCS}
${PHYSICS_COMMON}

@binding(0) @group(0) var<storage, read_write> position : array<vec4<f32>>;
@binding(1) @group(0) var<storage, read_write> rotation : array<vec4<f32>>;

fn terrainGrad (pos : vec3<f32>) -> vec3<f32> {
  var e = vec2<f32>(1.0,-1.0)*0.5773;
  const eps = 0.0005;
  return normalize( e.xyy*terrainSDF( pos + e.xyy*eps ) + 
            e.yyx*terrainSDF( pos + e.yyx*eps ) + 
            e.yxy*terrainSDF( pos + e.yxy*eps ) + 
					  e.xxx*terrainSDF( pos + e.xxx*eps ) );
}


@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE},1,1) fn solveTerrainPosition (@builtin(global_invocation_id) globalVec : vec3<u32>) {
  var id = globalVec.x;
  var p = position[id].xyz;
  var d0 = terrainSDF(p);
  if d0 >= ${PARTICLE_RADIUS} {
    return;
  }

  var q = rotation[id];
  var R = qMatrix(q);

  position[id] = vec4(p.xyz - terrainGrad(p) * (d0 - ${PARTICLE_RADIUS}), 0);
  rotation[id] = q;
}
`
        }),
        entryPoint: "solveTerrainPosition"
      }
    });
    const particleGridCountBuffer = device.createBuffer({
      label: "particleGridCount",
      size: 4 * COLLISION_TABLE_SIZE,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleGridIdBuffer = device.createBuffer({
      label: "particleGridEntry",
      size: 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleContactCountBuffer = device.createBuffer({
      label: "particleContactCount",
      size: 4 * COLLISION_TABLE_SIZE,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const contactListBuffer = device.createBuffer({
      label: "contactListBuffer",
      size: 2 * 4 * CONTACT_TABLE_SIZE,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    particleGridCountBuffer.unmap();
    particleGridIdBuffer.unmap();
    particleContactCountBuffer.unmap();
    contactListBuffer.unmap();
    const particlePositionBuffer = device.createBuffer({
      label: "particlePosition",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleRotationBuffer = device.createBuffer({
      label: "particleRotation",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleColorBuffer = device.createBuffer({
      label: "particleColor",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleVelocityBuffer = device.createBuffer({
      label: "particleVelocity",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleAngularVelocityBuffer = device.createBuffer({
      label: "particleAngularVelocity",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particlePositionPredictionBuffer = device.createBuffer({
      label: "particlePositionPrediction",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particlePositionCorrectionBuffer = device.createBuffer({
      label: "particlePositionCorrection",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleRotationPredictionBuffer = device.createBuffer({
      label: "particleRotationPrediction",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    const particleRotationCorrectionBuffer = device.createBuffer({
      label: "particleRotationCorrection",
      size: 4 * 4 * NUM_PARTICLES,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    {
      const particlePositionData = new Float32Array(particlePositionBuffer.getMappedRange());
      const particleRotationData = new Float32Array(particleRotationBuffer.getMappedRange());
      const particleColorData = new Float32Array(particleColorBuffer.getMappedRange());
      const particleVelocityData = new Float32Array(particleVelocityBuffer.getMappedRange());
      const particleAngularVelocityData = new Float32Array(particleAngularVelocityBuffer.getMappedRange());
      for (let i = 0; i < NUM_PARTICLES; ++i) {
        const color = PALETTE[1 + i % (PALETTE.length - 1)];
        for (let j = 0; j < 4; ++j) {
          particlePositionData[4 * i + j] = 10.5 * (2 * Math.random() - 1);
          if (j == 1) {
            particlePositionData[4 * i + j] = 0;
          }
          particleColorData[4 * i + j] = color[j];
          particleRotationData[4 * i + j] = Math.random() - 0.5;
          particleVelocityData[4 * i + j] = 0;
          particleAngularVelocityData[4 * i + j] = Math.random() - 0.5;
        }
        const q = particleRotationData.subarray(4 * i, 4 * (i + 1));
        quat_exports.normalize(q, q);
      }
    }
    particlePositionBuffer.unmap();
    particleRotationBuffer.unmap();
    particleColorBuffer.unmap();
    particleVelocityBuffer.unmap();
    particleAngularVelocityBuffer.unmap();
    particlePositionPredictionBuffer.unmap();
    particlePositionCorrectionBuffer.unmap();
    particleRotationPredictionBuffer.unmap();
    particleRotationCorrectionBuffer.unmap();
    const [clearGridBindGroup, clearContactBindGroup, clearContactListBindGroup] = [particleGridCountBuffer, particleContactCountBuffer, contactListBuffer].map((buffer) => device.createBindGroup({
      layout: clearBufferPipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: {
          buffer
        }
      }]
    }));
    const gridCountBindGroup = device.createBindGroup({
      layout: gridCountPipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: {
          buffer: particlePositionBuffer
        }
      }, {
        binding: 1,
        resource: {
          buffer: particleGridCountBuffer
        }
      }]
    });
    const gridCopyBindGroup = device.createBindGroup({
      layout: gridCopyParticlePipeline.getBindGroupLayout(0),
      entries: [{
        binding: 0,
        resource: {
          buffer: particlePositionBuffer
        }
      }, {
        binding: 1,
        resource: {
          buffer: particleGridCountBuffer
        }
      }, {
        binding: 2,
        resource: {
          buffer: particleGridIdBuffer
        }
      }]
    });
    const contactCountBindGroup = device.createBindGroup({
      layout: contactCountPipeline.getBindGroupLayout(0),
      entries: [
        particlePositionBuffer,
        particleGridCountBuffer,
        particleGridIdBuffer,
        particleContactCountBuffer
      ].map((buffer, binding) => {
        return {
          binding,
          resource: {
            buffer
          }
        };
      })
    });
    const contactListBindGroup = device.createBindGroup({
      layout: contactListPipeline.getBindGroupLayout(0),
      entries: [
        particlePositionBuffer,
        particleGridCountBuffer,
        particleGridIdBuffer,
        particleContactCountBuffer,
        contactListBuffer
      ].map((buffer, binding) => {
        return {
          binding,
          resource: {
            buffer
          }
        };
      })
    });
    const gridCountScan = new WebGPUScan({
      device,
      threadsPerGroup: SCAN_THREADS,
      itemsPerThread: SCAN_ITEMS,
      dataType: "u32",
      dataSize: 4,
      dataFunc: "A + B",
      dataUnit: "0u"
    });
    const gridCountScanPass = await gridCountScan.createPass(COLLISION_TABLE_SIZE, particleGridCountBuffer);
    const contactCountScanPass = await gridCountScan.createPass(COLLISION_TABLE_SIZE, particleContactCountBuffer);
    const spriteQuadUV = device.createBuffer({
      size: 2 * 4 * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true
    });
    new Float32Array(spriteQuadUV.getMappedRange()).set([
      -1,
      -1,
      -1,
      1,
      1,
      -1,
      1,
      1
    ]);
    spriteQuadUV.unmap();
    const renderUniformBindGroupLayout = device.createBindGroupLayout({
      label: "renderUniformBindGroupLayout",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
        buffer: {
          type: "uniform",
          hasDynamicOffset: false
        }
      }]
    });
    const renderParticleBindGroupLayout = device.createBindGroupLayout({
      label: "renderParticleBindGroupLayout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
            hasDynamicOffset: false
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
            hasDynamicOffset: false
          }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
            hasDynamicOffset: false
          }
        }
      ]
    });
    const debugContactsBindGroupLayout = device.createBindGroupLayout({
      label: "renderContactsBindGroupLayout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
            hasDynamicOffset: false
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: {
            type: "read-only-storage",
            hasDynamicOffset: false
          }
        }
      ]
    });
    const renderUniformData = new Float32Array(1024);
    let uniformPtr = 0;
    function nextUniform(size) {
      const result = renderUniformData.subarray(uniformPtr, uniformPtr + size);
      uniformPtr += size;
      return result;
    }
    const view = nextUniform(16);
    const projection2 = nextUniform(16);
    const projectionInv = nextUniform(16);
    const fog = nextUniform(4);
    const lightDir = nextUniform(4);
    const eye = nextUniform(4);
    const renderUniformBuffer = device.createBuffer({
      size: renderUniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const renderParticlePipeline = device.createRenderPipeline({
      label: "renderParticlePipeline",
      layout: device.createPipelineLayout({
        label: "renderLayout",
        bindGroupLayouts: [
          renderUniformBindGroupLayout,
          renderParticleBindGroupLayout
        ]
      }),
      vertex: {
        module: renderShader,
        entryPoint: "vertMain",
        buffers: [{
          arrayStride: 2 * 4,
          attributes: [{
            shaderLocation: 0,
            offset: 0,
            format: "float32x2"
          }]
        }]
      },
      fragment: {
        module: renderShader,
        entryPoint: "fragMain",
        targets: [{ format: presentationFormat }]
      },
      primitive: {
        topology: "triangle-strip"
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus"
      }
    });
    const renderBackgroundPipeline = device.createRenderPipeline({
      label: "renderBackgroundPipeline",
      layout: device.createPipelineLayout({
        label: "renderBackgroundPipelineLayout",
        bindGroupLayouts: [
          renderUniformBindGroupLayout
        ]
      }),
      vertex: {
        module: backroundShader,
        entryPoint: "vertMain"
      },
      fragment: {
        module: backroundShader,
        entryPoint: "fragMain",
        targets: [{ format: presentationFormat }]
      },
      primitive: {
        topology: "triangle-strip"
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "always",
        format: "depth24plus"
      }
    });
    const renderContactPipeline = device.createRenderPipeline({
      label: "renderContactPipeline",
      layout: device.createPipelineLayout({
        label: "renderContactPipelineLayout",
        bindGroupLayouts: [
          renderUniformBindGroupLayout,
          debugContactsBindGroupLayout
        ]
      }),
      vertex: {
        module: debugContactShader,
        entryPoint: "vertMain"
      },
      fragment: {
        module: debugContactShader,
        entryPoint: "fragMain",
        targets: [{ format: presentationFormat }]
      },
      primitive: {
        topology: "line-list"
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "always",
        format: "depth24plus"
      }
    });
    const predictPipeline = device.createComputePipeline({
      label: "particlePredictPipeline",
      layout: "auto",
      compute: {
        module: particlePredictShader,
        entryPoint: "predictPositions"
      }
    });
    const updatePipeline = device.createComputePipeline({
      label: "particleUpdatePipeline",
      layout: "auto",
      compute: {
        module: particleUpdateShader,
        entryPoint: "updatePositions"
      }
    });
    const predictBindGroup = device.createBindGroup({
      label: "predictBindGroup",
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
        };
      })
    });
    const updateBindGroup = device.createBindGroup({
      label: "updatePositionBindGroup",
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
        };
      })
    });
    const solveTerrainPositionBindGroup = device.createBindGroup({
      label: "solveTerrainPositionBindGroup",
      layout: solveTerrainPositionPipeline.getBindGroupLayout(0),
      entries: [
        particlePositionPredictionBuffer,
        particleRotationPredictionBuffer
      ].map((buffer, binding) => {
        return {
          binding,
          resource: { buffer }
        };
      })
    });
    const renderUniformBindGroup = device.createBindGroup({
      label: "uniformBindGroup",
      layout: renderUniformBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: renderUniformBuffer
          }
        }
      ]
    });
    const renderParticleBindGroup = device.createBindGroup({
      label: "renderParticleBindGroup",
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
    });
    const renderContactBindGroup = device.createBindGroup({
      label: "renderContactBindGroup",
      layout: debugContactsBindGroupLayout,
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
            buffer: contactListBuffer
          }
        }
      ]
    });
    function frame(tick) {
      mat4_exports.perspective(projection2, Math.PI / 4, canvas.width / canvas.height, 0.01, 50);
      mat4_exports.invert(projectionInv, projection2);
      const theta = 0;
      vec4_exports.set(eye, 8 * Math.cos(theta), 3, 8 * Math.sin(theta), 0);
      mat4_exports.lookAt(view, eye, [0, -0.5, 0], [0, 1, 0]);
      vec4_exports.copy(fog, PALETTE[0]);
      vec4_exports.set(lightDir, -1, -1, -0.2, 0);
      vec4_exports.normalize(lightDir, lightDir);
      device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformData.buffer, 0, renderUniformData.byteLength);
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "load",
            storeOp: "store"
          }
        ],
        depthStencilAttachment: {
          view: depthTexture.createView(),
          depthLoadOp: "load",
          depthStoreOp: "store"
        }
      });
      passEncoder.setBindGroup(0, renderUniformBindGroup);
      passEncoder.setBindGroup(1, renderParticleBindGroup);
      passEncoder.setVertexBuffer(0, spriteQuadUV);
      passEncoder.setPipeline(renderBackgroundPipeline);
      passEncoder.draw(4);
      passEncoder.setPipeline(renderParticlePipeline);
      passEncoder.draw(4, NUM_PARTICLES);
      passEncoder.setBindGroup(1, renderContactBindGroup);
      passEncoder.setPipeline(renderContactPipeline);
      passEncoder.draw(2 * CONTACT_TABLE_SIZE);
      passEncoder.end();
      const computePass = commandEncoder.beginComputePass();
      const NGROUPS = NUM_PARTICLES / PARTICLE_WORKGROUP_SIZE;
      computePass.setPipeline(clearBufferPipeline);
      computePass.setBindGroup(0, clearGridBindGroup);
      computePass.dispatchWorkgroups(NGROUPS);
      computePass.setBindGroup(0, clearContactBindGroup);
      computePass.dispatchWorkgroups(NGROUPS);
      computePass.setBindGroup(0, clearContactListBindGroup);
      computePass.dispatchWorkgroups(CONTACT_TABLE_SIZE / PARTICLE_WORKGROUP_SIZE);
      computePass.setBindGroup(0, gridCountBindGroup);
      computePass.setPipeline(gridCountPipeline);
      computePass.dispatchWorkgroups(NGROUPS);
      gridCountScanPass.run(computePass);
      computePass.setBindGroup(0, gridCopyBindGroup);
      computePass.setPipeline(gridCopyParticlePipeline);
      computePass.dispatchWorkgroups(NGROUPS);
      computePass.setBindGroup(0, contactCountBindGroup);
      computePass.setPipeline(contactCountPipeline);
      computePass.dispatchWorkgroups(NGROUPS);
      contactCountScanPass.run(computePass);
      computePass.setBindGroup(0, contactListBindGroup);
      computePass.setPipeline(contactListPipeline);
      computePass.dispatchWorkgroups(NGROUPS);
      for (let i = 0; i < SUBSTEPS; ++i) {
        computePass.setBindGroup(0, predictBindGroup);
        computePass.setPipeline(predictPipeline);
        computePass.dispatchWorkgroups(NGROUPS);
        computePass.setBindGroup(0, solveTerrainPositionBindGroup);
        computePass.setPipeline(solveTerrainPositionPipeline);
        computePass.dispatchWorkgroups(NGROUPS);
        computePass.setBindGroup(0, updateBindGroup);
        computePass.setPipeline(updatePipeline);
        computePass.dispatchWorkgroups(NGROUPS);
      }
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  main().catch((err) => console.error(err));
})();
