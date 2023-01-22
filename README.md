# [Mikola's WebGPU experiments](https://mikolalysenko.github.io/webgpu-experiments/index.html)

I'm learning WebGPU.  Here's what I've built so far:

* [Hello world triangle](https://mikolalysenko.github.io/webgpu-experiments/triangle.html)
* [Rotating icosahedron](https://mikolalysenko.github.io/webgpu-experiments/icosahedron.html)
* [Point sprites](https://mikolalysenko.github.io/webgpu-experiments/sprites.html)
* [Particle life](https://mikolalysenko.github.io/webgpu-experiments/particle-life.html)
* [Matrix multiplication](https://mikolalysenko.github.io/webgpu-experiments/matrix-mult.html)
* [Prefix sum](https://mikolalysenko.github.io/webgpu-experiments/prefix-sum.html)

For project history, check the [log](LOG.md).

# Development

Clone this repo, and using node.js/npm run:

```
npm ci
```

Once all dependencies are initialized there are two basic commands:

* `npm run watch`: Sets up a live reloading server for working on the demos
* `npm run build`: Builds all the demos

And one very dangerous command:

* `npm run gh-pages`: Which builds all the files and pushes them to gh-pages

All the code is in `src/demos`, take a look if you are curious

# License
(c) 2023 Mikola Lysenko.  MIT License