"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    canvas: $("observatoryCanvas"),
    status: $("observatoryStatus"),
    empty: $("observatoryEmpty"),
    scene: $("observatoryScene"),
    field: $("observatoryField"),
    density: $("observatoryDensity"),
    time: $("observatoryTime"),
    reset: $("observatoryReset"),
    evidence: $("observatoryEvidence"),
    residency: $("observatoryResidency"),
    coordinates: $("observatoryCoordinates"),
    probe: $("observatoryProbe"),
    precision: $("observatoryPrecision"),
    warning: $("observatoryWarning"),
    fieldLabel: $("observatoryFieldLabel"),
    clock: $("observatoryClock"),
  };
  if (!ui.canvas) return;

  const state = {
    device: null,
    context: null,
    format: null,
    volumePipeline: null,
    meshPipeline: null,
    bindGroup: null,
    uniformBuffer: null,
    pageBuffer: null,
    fineTexture: null,
    coarseTexture: null,
    depthTexture: null,
    scenes: [],
    sceneId: null,
    manifest: null,
    frame: null,
    field: null,
    fineLevel: null,
    coarseLevel: null,
    chunkCache: new Map(),
    resident: null,
    residentCount: 0,
    loadGeneration: 0,
    camera: { yaw: 0.62, pitch: 0.31, distance: 1.72 },
    drag: null,
    meshBuffers: [],
    raf: 0,
    refreshTimer: 0,
  };

  const shader = `
struct SceneUniforms {
  cameraPosition: vec4f,
  cameraRight: vec4f,
  cameraUp: vec4f,
  cameraForward: vec4f,
  fineDimensions: vec4f,
  coarseDimensions: vec4f,
  chunkDimensions: vec4f,
  chunkCounts: vec4f,
  rangeParameters: vec4f,
  viewportParameters: vec4f,
  gridOrigin: vec4f,
  gridLengths: vec4f,
};

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(0) @binding(1) var fineVolume: texture_3d<f32>;
@group(0) @binding(2) var coarseVolume: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> pageTable: array<u32>;

fn scalarLoad(
  volume: texture_3d<f32>,
  dimensions: vec3u,
  coordinate: vec3i
) -> f32 {
  let bounded = clamp(coordinate, vec3i(0), vec3i(dimensions) - vec3i(1));
  return textureLoad(volume, bounded, 0).r;
}

fn trilinear(
  volume: texture_3d<f32>,
  dimensions: vec3u,
  coordinate: vec3f
) -> f32 {
  let scaled = clamp(coordinate, vec3f(0.0), vec3f(1.0))
    * vec3f(dimensions - vec3u(1));
  let base = vec3i(floor(scaled));
  let fraction = fract(scaled);
  let x00 = mix(
    scalarLoad(volume, dimensions, base),
    scalarLoad(volume, dimensions, base + vec3i(1, 0, 0)),
    fraction.x
  );
  let x10 = mix(
    scalarLoad(volume, dimensions, base + vec3i(0, 1, 0)),
    scalarLoad(volume, dimensions, base + vec3i(1, 1, 0)),
    fraction.x
  );
  let x01 = mix(
    scalarLoad(volume, dimensions, base + vec3i(0, 0, 1)),
    scalarLoad(volume, dimensions, base + vec3i(1, 0, 1)),
    fraction.x
  );
  let x11 = mix(
    scalarLoad(volume, dimensions, base + vec3i(0, 1, 1)),
    scalarLoad(volume, dimensions, base + vec3i(1, 1, 1)),
    fraction.x
  );
  return mix(mix(x00, x10, fraction.y), mix(x01, x11, fraction.y), fraction.z);
}

fn sampleField(coordinate: vec3f) -> f32 {
  let fineDimensions = vec3u(scene.fineDimensions.xyz);
  let coarseDimensions = vec3u(scene.coarseDimensions.xyz);
  let chunkDimensions = vec3u(scene.chunkDimensions.xyz);
  let counts = vec3u(scene.chunkCounts.xyz);
  let voxel = min(
    vec3u(clamp(coordinate, vec3f(0.0), vec3f(0.999999))
      * vec3f(fineDimensions)),
    fineDimensions - vec3u(1)
  );
  let brick = min(voxel / chunkDimensions, counts - vec3u(1));
  let page = (brick.x * counts.y + brick.y) * counts.z + brick.z;
  if (pageTable[page] != 0u) {
    return trilinear(fineVolume, fineDimensions, coordinate);
  }
  return trilinear(coarseVolume, coarseDimensions, coordinate);
}

fn palette(value: f32, signedField: bool) -> vec3f {
  if (signedField) {
    let magnitude = abs(value * 2.0 - 1.0);
    let neutral = vec3f(0.82, 0.90, 0.94);
    let cold = vec3f(0.05, 0.35, 0.95);
    let hot = vec3f(1.0, 0.18, 0.08);
    return mix(neutral, select(cold, hot, value >= 0.5), pow(magnitude, 0.72));
  }
  let a = vec3f(0.025, 0.035, 0.11);
  let b = vec3f(0.0, 0.75, 0.96);
  let c = vec3f(1.0, 0.29, 0.08);
  return mix(mix(a, b, smoothstep(0.0, 0.62, value)), c, smoothstep(0.58, 1.0, value));
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn fullscreenVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var result: VertexOutput;
  result.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  result.uv = positions[vertexIndex] * 0.5 + 0.5;
  return result;
}

fn intersectBox(origin: vec3f, direction: vec3f) -> vec2f {
  let safeDirection = select(
    direction,
    vec3f(1.0e-6),
    abs(direction) < vec3f(1.0e-6)
  );
  let first = (vec3f(-0.5) - origin) / safeDirection;
  let second = (vec3f(0.5) - origin) / safeDirection;
  let nearValue = max(max(min(first.x, second.x), min(first.y, second.y)), min(first.z, second.z));
  let farValue = min(min(max(first.x, second.x), max(first.y, second.y)), max(first.z, second.z));
  return vec2f(nearValue, farValue);
}

@fragment
fn volumeFragment(input: VertexOutput) -> @location(0) vec4f {
  let aspect = scene.viewportParameters.x;
  let tangent = scene.viewportParameters.y;
  let rayDirection = normalize(
    scene.cameraForward.xyz
    + (input.uv.x * 2.0 - 1.0) * aspect * tangent * scene.cameraRight.xyz
    + (1.0 - input.uv.y * 2.0) * tangent * scene.cameraUp.xyz
  );
  let hit = intersectBox(scene.cameraPosition.xyz, rayDirection);
  if (hit.y <= max(hit.x, 0.0)) {
    discard;
  }
  var distance = max(hit.x, 0.0);
  let span = hit.y - distance;
  let stepLength = max(span / 300.0, 0.0018);
  var accumulated = vec4f(0.0);
  let minimum = scene.rangeParameters.x;
  let maximum = scene.rangeParameters.y;
  let density = scene.rangeParameters.z;
  let signedField = scene.rangeParameters.w > 0.5;
  for (var sampleIndex = 0; sampleIndex < 384; sampleIndex++) {
    if (distance > hit.y || accumulated.a > 0.985) {
      break;
    }
    let point = scene.cameraPosition.xyz + rayDirection * distance;
    let raw = sampleField(point + vec3f(0.5));
    var normalized = clamp((raw - minimum) / max(maximum - minimum, 1.0e-20), 0.0, 1.0);
    var opacitySignal = normalized;
    if (signedField) {
      let center = max(abs(minimum), abs(maximum));
      let signedValue = clamp(raw / max(center, 1.0e-20), -1.0, 1.0);
      normalized = signedValue * 0.5 + 0.5;
      opacitySignal = abs(signedValue);
    }
    let localAlpha = (1.0 - exp(-density * opacitySignal * opacitySignal * stepLength * 8.0));
    let color = palette(normalized, signedField);
    let contributionAlpha = (1.0 - accumulated.a) * localAlpha;
    accumulated = vec4f(
      accumulated.rgb + contributionAlpha * color,
      accumulated.a + contributionAlpha
    );
    distance += stepLength;
  }
  let glow = 0.025 * vec3f(0.04, 0.34, 0.46);
  return vec4f(accumulated.rgb + glow * accumulated.a, accumulated.a);
}

struct MeshOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex
fn meshVertex(
  @location(0) worldPosition: vec3f,
  @builtin(instance_index) instance: u32
) -> MeshOutput {
  let normalized = (worldPosition - scene.gridOrigin.xyz) / scene.gridLengths.xyz - vec3f(0.5);
  let relative = normalized - scene.cameraPosition.xyz;
  let viewDepth = max(dot(relative, scene.cameraForward.xyz), 0.001);
  let tangent = scene.viewportParameters.y;
  let aspect = scene.viewportParameters.x;
  let projected = vec2f(
    dot(relative, scene.cameraRight.xyz) / (viewDepth * tangent * aspect),
    dot(relative, scene.cameraUp.xyz) / (viewDepth * tangent)
  );
  var result: MeshOutput;
  result.position = vec4f(projected, clamp(viewDepth / 5.0, 0.0, 1.0), 1.0);
  result.color = select(
    vec4f(0.2, 0.9, 1.0, 0.24),
    vec4f(1.0, 0.45, 0.12, 0.32),
    (instance & 1u) == 1u
  );
  return result;
}

@fragment
fn meshFragment(input: MeshOutput) -> @location(0) vec4f {
  return input.color;
}
`;

  function setStatus(text, error = false) {
    ui.status.textContent = text;
    ui.status.classList.toggle("error", error);
  }

  async function api(path) {
    const response = await fetch(path, { cache: "no-store" });
    const contentType = response.headers.get("content-type") || "";
    const body = await response.text();
    if (!contentType.includes("application/json")) {
      throw new Error(
        `${path} returned HTTP ${response.status} ${contentType || "without a content type"}; ` +
        "restart Tesseract Studio so its Observatory API matches the browser client",
      );
    }
    let payload;
    try {
      payload = JSON.parse(body);
    } catch (error) {
      throw new Error(`${path} returned invalid JSON: ${error.message}`);
    }
    if (!response.ok || payload.ok === false) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    return payload;
  }

  async function initializeGPU() {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found");
    state.device = await adapter.requestDevice();
    state.device.lost.then((info) => setStatus(`GPU DEVICE LOST: ${info.message}`, true));
    state.context = ui.canvas.getContext("webgpu");
    state.format = navigator.gpu.getPreferredCanvasFormat();
    state.context.configure({
      device: state.device,
      format: state.format,
      alphaMode: "premultiplied",
    });
    const module = state.device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    const compilationErrors = compilation.messages.filter(
      (message) => message.type === "error",
    );
    if (compilationErrors.length) {
      throw new Error(
        `Observatory shader compilation failed:\n${compilationErrors
          .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`)
          .join("\n")}`,
      );
    }
    state.volumePipeline = await state.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: { module, entryPoint: "fullscreenVertex" },
      fragment: {
        module,
        entryPoint: "volumeFragment",
        targets: [{
          format: state.format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
    state.meshPipeline = await state.device.createRenderPipelineAsync({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "meshVertex",
        buffers: [{
          arrayStride: 12,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        }],
      },
      fragment: {
        module,
        entryPoint: "meshFragment",
        targets: [{
          format: state.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
    state.uniformBuffer = state.device.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  function resize() {
    const scale = Math.min(devicePixelRatio || 1, 1.75);
    const width = Math.max(2, Math.floor(ui.canvas.clientWidth * scale));
    const height = Math.max(2, Math.floor(ui.canvas.clientHeight * scale));
    if (
      ui.canvas.width === width
      && ui.canvas.height === height
      && (!state.device || state.depthTexture)
    ) return;
    ui.canvas.width = width;
    ui.canvas.height = height;
    if (state.device) {
      state.depthTexture?.destroy();
      state.depthTexture = state.device.createTexture({
        size: [width, height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }

  function cameraVectors() {
    const { yaw, pitch, distance } = state.camera;
    const position = [
      distance * Math.cos(pitch) * Math.sin(yaw),
      distance * Math.sin(pitch),
      distance * Math.cos(pitch) * Math.cos(yaw),
    ];
    const forwardLength = Math.hypot(...position);
    const forward = position.map((value) => -value / forwardLength);
    let right = [forward[2], 0, -forward[0]];
    const rightLength = Math.hypot(...right) || 1;
    right = right.map((value) => value / rightLength);
    const up = [
      right[1] * forward[2] - right[2] * forward[1],
      right[2] * forward[0] - right[0] * forward[2],
      right[0] * forward[1] - right[1] * forward[0],
    ];
    return { position, forward, right, up };
  }

  function updateUniforms() {
    if (!state.field || !state.fineLevel) return;
    const camera = cameraVectors();
    const axes = state.manifest.coordinates.axes;
    const values = new Float32Array(48);
    const put = (offset, source, w = 0) => {
      values.set(source, offset);
      values[offset + 3] = w;
    };
    put(0, camera.position, 1);
    put(4, camera.right);
    put(8, camera.up);
    put(12, camera.forward);
    put(16, state.fineLevel.shape);
    put(20, state.coarseLevel.shape);
    put(24, state.fineLevel.chunk_shape);
    put(28, state.fineLevel.chunk_counts);
    const minimum = state.fineLevel.minimum;
    const maximum = state.fineLevel.maximum;
    put(32, [minimum, maximum, Number(ui.density.value)], state.field.display?.signed ? 1 : 0);
    put(36, [ui.canvas.width / Math.max(ui.canvas.height, 1), Math.tan(Math.PI / 7)]);
    put(40, axes.map((axis) => axis.origin));
    put(44, axes.map((axis) => axis.spacing * axis.size));
    state.device.queue.writeBuffer(state.uniformBuffer, 0, values);
  }

  function createTexture(shape) {
    return state.device.createTexture({
      size: shape,
      dimension: "3d",
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  function brickActualShape(level, coordinate) {
    return level.shape.map((size, axis) =>
      Math.min(level.chunk_shape[axis], size - coordinate[axis] * level.chunk_shape[axis])
    );
  }

  function chunkKey(coordinate) {
    return coordinate.join("/");
  }

  function chunkURL(level, coordinate) {
    return `/api/scenes/${state.sceneId}/files/${level.path}/c/${coordinate.join("/")}`;
  }

  async function fetchChunk(level, coordinate, generation) {
    const cacheKey = `${level.path}:${chunkKey(coordinate)}`;
    if (state.chunkCache.has(cacheKey)) return state.chunkCache.get(cacheKey);
    const response = await fetch(chunkURL(level, coordinate));
    if (!response.ok) throw new Error(`Chunk fetch failed (${response.status})`);
    const data = new Float32Array(await response.arrayBuffer());
    if (generation !== state.loadGeneration) return null;
    state.chunkCache.set(cacheKey, data);
    return data;
  }

  function uploadChunk(texture, level, coordinate, data) {
    const shape = brickActualShape(level, coordinate);
    const bytesPerRow = Math.ceil((shape[0] * 4) / 256) * 256;
    const padded = new Uint8Array(bytesPerRow * shape[1] * shape[2]);
    const output = new DataView(padded.buffer);
    for (let x = 0; x < shape[0]; x += 1) {
      for (let y = 0; y < shape[1]; y += 1) {
        for (let z = 0; z < shape[2]; z += 1) {
          const sourceIndex = (x * shape[1] + y) * shape[2] + z;
          const destination = z * bytesPerRow * shape[1] + y * bytesPerRow + x * 4;
          output.setFloat32(destination, data[sourceIndex], true);
        }
      }
    }
    state.device.queue.writeTexture(
      {
        texture,
        origin: coordinate.map((value, axis) => value * level.chunk_shape[axis]),
      },
      padded,
      { offset: 0, bytesPerRow, rowsPerImage: shape[1] },
      shape
    );
  }

  function allChunks(level) {
    const result = [];
    for (let x = 0; x < level.chunk_counts[0]; x += 1) {
      for (let y = 0; y < level.chunk_counts[1]; y += 1) {
        for (let z = 0; z < level.chunk_counts[2]; z += 1) result.push([x, y, z]);
      }
    }
    const center = level.chunk_counts.map((value) => (value - 1) * 0.5);
    result.sort((left, right) => {
      const distance = (value) => value.reduce(
        (sum, item, axis) => sum + (item - center[axis]) ** 2, 0
      );
      return distance(left) - distance(right);
    });
    return result;
  }

  function updatePageTable() {
    state.device.queue.writeBuffer(state.pageBuffer, 0, state.resident);
    ui.residency.textContent = `${state.residentCount} / ${state.resident.length}`;
  }

  async function streamFineBricks(generation) {
    for (const coordinate of allChunks(state.fineLevel)) {
      if (generation !== state.loadGeneration) return;
      const data = await fetchChunk(state.fineLevel, coordinate, generation);
      if (!data || generation !== state.loadGeneration) return;
      uploadChunk(state.fineTexture, state.fineLevel, coordinate, data);
      const [x, y, z] = coordinate;
      const counts = state.fineLevel.chunk_counts;
      const index = (x * counts[1] + y) * counts[2] + z;
      if (!state.resident[index]) {
        state.resident[index] = 1;
        state.residentCount += 1;
        updatePageTable();
      }
      scheduleRender();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    setStatus("FULL-RESOLUTION RESIDENT");
  }

  async function loadCoarseLevel(generation) {
    for (const coordinate of allChunks(state.coarseLevel)) {
      const data = await fetchChunk(state.coarseLevel, coordinate, generation);
      if (!data || generation !== state.loadGeneration) return false;
      uploadChunk(state.coarseTexture, state.coarseLevel, coordinate, data);
    }
    return true;
  }

  async function loadField() {
    if (!state.device || !state.frame) return;
    state.loadGeneration += 1;
    const generation = state.loadGeneration;
    state.field = state.frame.fields.find((field) => field.id === ui.field.value);
    if (!state.field) return;
    const levels = state.field.multiscales;
    state.fineLevel = levels[0];
    state.coarseLevel = levels[levels.length - 1];
    state.chunkCache.clear();
    state.fineTexture?.destroy();
    state.coarseTexture?.destroy();
    state.fineTexture = createTexture(state.fineLevel.shape);
    state.coarseTexture = createTexture(state.coarseLevel.shape);
    const pageCount = state.fineLevel.chunk_counts.reduce((a, b) => a * b, 1);
    state.resident = new Uint32Array(Math.max(pageCount, 1));
    state.residentCount = 0;
    state.pageBuffer?.destroy();
    state.pageBuffer = state.device.createBuffer({
      size: Math.max(4, state.resident.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    updatePageTable();
    state.bindGroup = state.device.createBindGroup({
      layout: state.volumePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.uniformBuffer } },
        { binding: 1, resource: state.fineTexture.createView() },
        { binding: 2, resource: state.coarseTexture.createView() },
        { binding: 3, resource: { buffer: state.pageBuffer } },
      ],
    });
    ui.fieldLabel.textContent = `${state.field.symbol} · ${state.field.name}`.toUpperCase();
    ui.precision.textContent = `${state.field.data_type} · ${state.field.accuracy.lossy ? "lossy" : "lossless base"}`;
    setStatus("LOADING COARSE VOLUME");
    const loaded = await loadCoarseLevel(generation);
    if (!loaded) return;
    setStatus("STREAMING FINE BRICKS");
    scheduleRender();
    streamFineBricks(generation).catch((error) => setStatus(error.message, true));
  }

  async function loadMeshes() {
    for (const mesh of state.meshBuffers) {
      mesh.vertex.destroy();
      mesh.index.destroy();
    }
    state.meshBuffers = [];
    if (!state.manifest?.geometry?.length) return;
    const samples = state.manifest.worldtubes?.flatMap((tube) => tube.samples || []) || [];
    const selected = samples.length
      ? samples[Math.min(Number(ui.time.value), samples.length - 1)]?.surface
      : null;
    const objects = state.manifest.geometry.filter((item) =>
      selected ? item.id === selected : item.step === state.frame.step
    ).slice(0, 4);
    for (let objectIndex = 0; objectIndex < objects.length; objectIndex += 1) {
      const object = objects[objectIndex];
      const [verticesResponse, indicesResponse] = await Promise.all([
        fetch(`/api/scenes/${state.sceneId}/files/${object.vertices.path}`),
        fetch(`/api/scenes/${state.sceneId}/files/${object.indices.path}`),
      ]);
      if (!verticesResponse.ok || !indicesResponse.ok) continue;
      const vertices = await verticesResponse.arrayBuffer();
      const indices = await indicesResponse.arrayBuffer();
      const vertex = state.device.createBuffer({
        size: Math.max(4, vertices.byteLength),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      const index = state.device.createBuffer({
        size: Math.max(4, indices.byteLength),
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      state.device.queue.writeBuffer(vertex, 0, vertices);
      state.device.queue.writeBuffer(index, 0, indices);
      state.meshBuffers.push({
        vertex,
        index,
        count: object.indices.count,
        instance: objectIndex,
      });
    }
    scheduleRender();
  }

  function render() {
    state.raf = 0;
    if (!state.device || !state.bindGroup || !state.depthTexture) return;
    resize();
    updateUniforms();
    const encoder = state.device.createCommandEncoder();
    const colorView = state.context.getCurrentTexture().createView();
    const volumePass = encoder.beginRenderPass({
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0.008, g: 0.016, b: 0.027, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    volumePass.setPipeline(state.volumePipeline);
    volumePass.setBindGroup(0, state.bindGroup);
    volumePass.draw(3);
    volumePass.end();
    if (state.meshBuffers.length) {
      const meshBindGroup = state.device.createBindGroup({
        layout: state.meshPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: state.uniformBuffer } }],
      });
      const meshPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          loadOp: "load",
          storeOp: "store",
        }],
        depthStencilAttachment: {
          view: state.depthTexture.createView(),
          depthClearValue: 1,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      meshPass.setPipeline(state.meshPipeline);
      meshPass.setBindGroup(0, meshBindGroup);
      for (const mesh of state.meshBuffers) {
        meshPass.setVertexBuffer(0, mesh.vertex);
        meshPass.setIndexBuffer(mesh.index, "uint32");
        meshPass.drawIndexed(mesh.count, 1, 0, 0, mesh.instance);
      }
      meshPass.end();
    }
    state.device.queue.submit([encoder.finish()]);
  }

  function scheduleRender() {
    if (!state.raf) state.raf = requestAnimationFrame(render);
  }

  async function loadScene(identifier) {
    state.sceneId = identifier;
    setStatus("READING SCENE CONTRACT");
    const payload = await api(`/api/scenes/${identifier}/manifest`);
    state.manifest = payload.scene;
    state.frame = state.manifest.frames[state.manifest.frames.length - 1];
    ui.field.innerHTML = state.frame.fields
      .map((field) => `<option value="${field.id}">${field.symbol} · ${field.name}</option>`)
      .join("");
    const samples = state.manifest.worldtubes?.flatMap((tube) => tube.samples || []) || [];
    ui.time.max = Math.max(0, state.manifest.frames.length - 1, samples.length - 1);
    ui.time.value = ui.time.max;
    ui.clock.textContent = `t = ${Number(state.frame.time).toPrecision(7)} · step ${state.frame.step}`;
    ui.coordinates.textContent = `${state.manifest.coordinates.chart} · ${state.manifest.coordinates.gauge.slicing}`;
    ui.warning.textContent = state.manifest.coordinates.gauge.coordinate_warning;
    ui.empty.classList.remove("visible");
    await Promise.all([loadField(), loadMeshes()]);
  }

  async function selectFrame(index) {
    const frames = state.manifest?.frames || [];
    const frame = frames[Math.min(index, frames.length - 1)];
    if (!frame) return;
    state.frame = frame;
    const selectedField = ui.field.value;
    ui.field.innerHTML = state.frame.fields
      .map((field) => `<option value="${field.id}">${field.symbol} · ${field.name}</option>`)
      .join("");
    if (state.frame.fields.some((field) => field.id === selectedField)) {
      ui.field.value = selectedField;
    }
    ui.clock.textContent = `t = ${Number(frame.time).toPrecision(7)} · step ${frame.step}`;
    await Promise.all([loadField(), loadMeshes()]);
  }

  async function refreshLiveScene() {
    if (!state.sceneId || !state.manifest?.provenance?.frames_may_be_dropped) return;
    const payload = await api(`/api/scenes/${state.sceneId}/manifest`);
    const next = payload.scene;
    if (next.frames.length <= state.manifest.frames.length) return;
    const wasFollowing = Number(ui.time.value) === Number(ui.time.max);
    state.manifest = next;
    ui.time.max = Math.max(0, next.frames.length - 1);
    if (wasFollowing) {
      ui.time.value = ui.time.max;
      await selectFrame(Number(ui.time.value));
    }
  }

  function installInteraction() {
    ui.canvas.addEventListener("pointerdown", (event) => {
      ui.canvas.setPointerCapture(event.pointerId);
      state.drag = { x: event.clientX, y: event.clientY, moved: false };
    });
    ui.canvas.addEventListener("pointermove", (event) => {
      if (!state.drag) return;
      const dx = event.clientX - state.drag.x;
      const dy = event.clientY - state.drag.y;
      state.drag.x = event.clientX;
      state.drag.y = event.clientY;
      state.drag.moved ||= Math.abs(dx) + Math.abs(dy) > 2;
      state.camera.yaw -= dx * 0.006;
      state.camera.pitch = Math.max(-1.35, Math.min(1.35, state.camera.pitch + dy * 0.006));
      scheduleRender();
    });
    ui.canvas.addEventListener("pointerup", async (event) => {
      const drag = state.drag;
      state.drag = null;
      if (!drag?.moved) await probe(event);
    });
    ui.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.camera.distance = Math.max(0.72, Math.min(4.5, state.camera.distance * Math.exp(event.deltaY * 0.001)));
      scheduleRender();
    }, { passive: false });
    ui.reset.addEventListener("click", () => {
      state.camera = { yaw: 0.62, pitch: 0.31, distance: 1.72 };
      scheduleRender();
    });
    ui.density.addEventListener("input", scheduleRender);
    ui.field.addEventListener("change", () => loadField().catch((error) => setStatus(error.message, true)));
    ui.scene.addEventListener("change", () => loadScene(ui.scene.value).catch((error) => setStatus(error.message, true)));
    ui.time.addEventListener("input", () => {
      selectFrame(Number(ui.time.value)).catch((error) => setStatus(error.message, true));
    });
    ui.evidence.addEventListener("click", () => {
      const provenance = state.manifest?.provenance;
      if (!provenance) return;
      ui.warning.textContent =
        `Checkpoint ${provenance.source_checkpoint_sha256.slice(0, 16)}… · ` +
        `${state.manifest.evidence.sources.length} evidence source(s) · ` +
        `derived=${provenance.derived_product}, restart=${provenance.restart_capable}`;
    });
    addEventListener("resize", () => { resize(); scheduleRender(); });
  }

  async function probe(event) {
    if (!state.field || !state.fineLevel) return;
    const bounds = ui.canvas.getBoundingClientRect();
    const normalized = [
      Math.max(0, Math.min(0.999999, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(0.999999, 1 - (event.clientY - bounds.top) / bounds.height)),
      0.5,
    ];
    const voxel = normalized.map((value, axis) => Math.floor(value * state.fineLevel.shape[axis]));
    const coordinate = voxel.map((value, axis) => Math.floor(value / state.fineLevel.chunk_shape[axis]));
    const data = await fetchChunk(state.fineLevel, coordinate, state.loadGeneration);
    if (!data) return;
    const shape = brickActualShape(state.fineLevel, coordinate);
    const local = voxel.map((value, axis) => value - coordinate[axis] * state.fineLevel.chunk_shape[axis]);
    const index = (local[0] * shape[1] + local[1]) * shape[2] + local[2];
    const axes = state.manifest.coordinates.axes;
    const position = voxel.map((value, axis) => axes[axis].origin + (value + 0.5) * axes[axis].spacing);
    ui.probe.textContent =
      `${state.field.symbol}=${data[index].toExponential(5)} at ` +
      `(${position.map((value) => value.toFixed(3)).join(", ")})`;
  }

  async function start() {
    installInteraction();
    resize();
    try {
      await initializeGPU();
      resize();
      const payload = await api("/api/scenes");
      state.scenes = payload.scenes;
      if (!state.scenes.length) {
        ui.empty.classList.add("visible");
        setStatus("NO SCENES", true);
        return;
      }
      ui.scene.innerHTML = state.scenes
        .map((scene) => `<option value="${scene.id}">${scene.title}</option>`)
        .join("");
      await loadScene(state.scenes[0].id);
      state.refreshTimer = setInterval(() => {
        refreshLiveScene().catch((error) => setStatus(error.message, true));
      }, 3000);
    } catch (error) {
      ui.empty.classList.add("visible");
      ui.empty.querySelector("strong").textContent = "OBSERVATORY UNAVAILABLE";
      ui.empty.querySelector("span").textContent = error.message;
      setStatus(error.message.toUpperCase(), true);
    }
  }

  start();
})();
