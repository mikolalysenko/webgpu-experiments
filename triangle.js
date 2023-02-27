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

  // src/demos/triangle.ts
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
    const shaderModule = device.createShaderModule({
      code: `
@fragment
fn fragMain() -> @location(0) vec4<f32> {
    return vec4(1.0, 0.0, 0.0, 1.0);
}

@vertex
fn vertMain(
    @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5)
    );
    return vec4<f32>(pos[VertexIndex], 0.0, 1.0);
}`
    });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vertMain"
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fragMain",
        targets: [
          {
            format: presentationFormat
          }
        ]
      },
      primitive: {
        topology: "triangle-list"
      }
    });
    function frame() {
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store"
          }
        ]
      });
      passEncoder.setPipeline(pipeline);
      passEncoder.draw(3);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      requestAnimationFrame(frame);
    }
    frame();
  }
  main().catch((err) => console.error(err));
})();
