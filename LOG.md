# Project log

## 02-20-2023

## 02-19-2023

* Implemented SDF impostor rendering and set up a basic framework for integrating and accumulating/applying Jacobi corrections to particles.  Next step is adding collision detection.

## 02-18-2023

* Started building an SDF based rigid body physics simulator.  Will be using XPBD for the objects and the scan stuff I've already put together for the dynamics.  I still haven't decided how I'm going to handle contact resolution, since Gauss-Seidel isn't too easy to parallelize.  If I can't figure out an easy way to implement it using atomics/locks or some kind of ad-hoc graph coloring, then I'll probably just skip it and do Jacobi based resolution.  Hope that will be fine for a demo....

## 02-11-2023

* Tried optimizing prefix sum a bit using a work efficient scan and adding Harris et al.'s bank conflict avoidance trick.  Didn't observe much benefit, its likely that the overhead of dispatching from webgpu/js is the dominant factor.
* It may be possible that using render bundles can further improve performance, but at this point it should be good enough as a foundation for particle collision detection.

## 02-05-2023

* Picking this project back up again.  A bit behind schedule this weekend due to building a new PC taking a bit longer than expected.  System is set up and stable now.
* Changed the GFLOPs calculation in the matrix multiply demo
* Tried changing around the matrix multiply code, I'm pretty sure bank conflicts are causing some performance loss but was taking too much time to figure out the correct index arithmetic to solve the problem.

## 01-23-2023

* Started working on a really terrible, not-very-parllel, non-work efficient prefix sum.
* Current idea is based on sqrt decomposition of the fenwick tree into 2 levels.  We compute prefix sums on each level by brute force for now
* In the future should try replacing the brute force steps with Blelloch-style parallel prefix sum.
* Not very happy with current performance or code complexity, need to figure out a simpler way to do this for future plans...
* Refactored UI boilerplate for benchmarks into a common module

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
