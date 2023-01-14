export function mustHave<T>(x:T|null|undefined) : T {
  if (!x) {
    document.body.innerHTML = `Your browser does not support WebGPU`	
    throw new Error('WebGPU not supported')
  }
  return x
}
mustHave(navigator.gpu)

function isCanvas (element:HTMLElement|null) : element is HTMLCanvasElement {
  if (!element) {
    return false
  }
  return /^canvas$/i.test(element.tagName)
}

const _canvas = document.getElementById('mainCanvas')
if (!isCanvas(_canvas)) {
  throw new Error('Error loading main canvas element')
}
_canvas.width = window.innerWidth
_canvas.height = window.innerHeight
export const canvas:HTMLCanvasElement = _canvas