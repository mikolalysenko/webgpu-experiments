import * as fs from 'node:fs/promises'
import * as path from 'node:path'

const ROOT_DIR = path.join(__dirname, '..')
const DIST_DIR = path.join(ROOT_DIR, 'dist')
const DEMO_DIR = path.join(ROOT_DIR, 'src/demos')

const htmlTemplate = (name:string) => 
`<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>One of Mik's WebGPU experiments : ${name}</title>
  </head>
  <body>
    <canvas id="mainCanvas" style="width:100%;height:100%;padding:0 !important;margin:0 !important;left:0;top:0;position:absolute;"></canvas>
	<script src="/${name}.js"></script>
  </body>
</html>
`

async function main () {
  // initialize www dir
  try {
    await fs.mkdir(DIST_DIR, {
      recursive: true
    })
  } catch {}
  // write all the files to the destination dir
  for (const file of await fs.readdir(DEMO_DIR)) {
    const basename = path.basename(file, '.ts')
    await fs.writeFile(
      path.join(DIST_DIR, `${basename}.html`),
      htmlTemplate(basename)
    )
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})

