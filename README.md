# [Mik's WebGPU experiments](https://mikolalysenko.github.io/webgpu-experiments/index.html)

Minimal experiments in learning WebGPU.

* [Hello world triangle](https://mikolalysenko.github.io/webgpu-experiments/triangle.html)
* [Rotating icosahedron](https://mikolalysenko.github.io/webgpu-experiments/icosahedron.html)
* [Point sprites](https://mikolalysenko.github.io/webgpu-experiments/sprites.html)

Not much to see here yet, more demos coming soon.  Check [log](LOG.md) for updates.

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