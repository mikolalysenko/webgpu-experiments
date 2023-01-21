# Project log

## 01-21-2023

* Had to force myself to do something today.  Wrote a really trivial blocked matrix multiply algorithm.  Nice example of a self contained compute shader.
* Hit a bit of trouble where I couldn't create a buffer with STORAGE and MAP_READ set.  Ended up just allocating 2 buffers and copying as a workaround, but this is pretty inefficient.

## 01-16-2023

* Decided to try implementing [Hunar's particle life](https://github.com/hunar4321/particle-life) using compute shaders.  Still doesn't quite look right but overall structure seems fine.
* Right now I do 16 passes to handle all pairwise interactions but it might be more efficient to pack all the particles together and do it in 1 shot.
* Not really sure yet how to best select workgroup sizes, need to experiment a bit more.
* So far, I really like WebGPU!  The API is very thoughtfully designed and once this is more widely supported I can see it fully replacing WebGL for basically everything.  The people behind this have done an amazing job!  I'm very impressed with how carefully considered and elegantly everything works. A++++ job

## 01-15-2023

* Working through documentation on WebGPU API.  Built another trivial demo with a rotating icosahedron to test my knowledge of uniforms and indexed rendering.
* Added a demo with point sprites to try emulating WebGL-style `GL_POINTS`.  Unfortunately the performance seems really bad, but maybe I am doing something wrong. Will ask around for advice.

## 01-14-2023

* Set up repository and build scripts.  Got a triangle rendering based on https://austin-eng.com/webgpu-samples/samples/helloTriangle.
