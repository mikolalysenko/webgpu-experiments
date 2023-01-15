export function mustHave<T>(x:T|null|undefined) : T {
  if (!x) {
    document.body.innerHTML = `Your browser does not support WebGPU`	
    throw new Error('WebGPU not supported')
  }
  return x
}
mustHave(navigator.gpu)

export function makeCanvas () {
  const canvas = document.createElement('canvas')
  Object.assign(canvas.style, {
    width: '100%',
    height: '100%',
    position: 'absolute',
    left: '0',
    top: '0',
    margin: '0',
    padding: '0'
  })
  canvas.width = window.innerWidth * window.devicePixelRatio
  canvas.height = window.innerHeight * window.devicePixelRatio
  document.body.appendChild(canvas)
  return canvas
}
