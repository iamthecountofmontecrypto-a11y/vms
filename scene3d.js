// 3D bridge scene (optional; loaded only when the "3D" toggle is on).
// Twin Sails (left) and Poole Bridge (right) as detailed 3D models: physically based water, night
// lighting with bloom, lattice girders, hydraulic rams, barrier arms, traffic that queues for
// lifts, boats, a waterfront, and the last-minute stunt car.
// Units are metres; roads run along x, the channel runs along z, y is up.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

// ---------------------------------------------------------------- shared timing (matches the 2D scene)
const LIFT_DELAY = 1.4, LIFT_TIME = 5.0;
function bezier(p1x, p1y, p2x, p2y){
  const bx = t => 3*(1-t)*(1-t)*t*p1x + 3*(1-t)*t*t*p2x + t*t*t;
  const by = t => 3*(1-t)*(1-t)*t*p1y + 3*(1-t)*t*t*p2y + t*t*t;
  return x => {
    if (x <= 0) return 0; if (x >= 1) return 1;
    let lo = 0, hi = 1;
    for (let i = 0; i < 30; i++){ const m = (lo + hi) / 2; if (bx(m) < x) lo = m; else hi = m; }
    return by((lo + hi) / 2);
  };
}
const EASE = bezier(.6, 0, .22, 1.05);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const STATE_COLORS = { BOTH: 0x2de2c4, POOLE: 0x7b5cff, TWIN: 0xff6fa8, OTHER: 0xffd166 };
let seed = 11;                                   // deterministic "random" so the town looks the same every load
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

// ---------------------------------------------------------------- water: swell, ripples, shaders
// Swell: Gerstner waves [heading (deg; 90 = towards the camera), wavelength (m), amplitude (m)].
// Speeds follow deep-water dispersion (w = sqrt(g k)); steepness is shared so crests sharpen but
// never loop. The first VERTEX_WAVES are long enough to displace the mesh; the rest only shade.
const VERTEX_WAVES = 3;
const WAVES = [[100, 16, .17, .3], [72, 11, .12, 2.1], [124, 7.2, .08, 4.4], [88, 4.7, .05, 1.2], [55, 3.2, .032, 5.3], [140, 2.3, .02, 3.7]]
  .map(([deg, L, A, phi], _, all) => {
    const a = deg * Math.PI / 180, k = 2 * Math.PI / L;
    return { dx: Math.cos(a), dz: Math.sin(a), k, A, w: Math.sqrt(9.81 * k), phi, Q: .6 / (k * A * all.length) };
  });
function waveHeight(x, z, t){                    // matches the displaced mesh (vertex waves only)
  let h = 0;
  for (let i = 0; i < VERTEX_WAVES; i++){ const w = WAVES[i]; h += w.A * Math.sin(w.k * (w.dx * x + w.dz * z) - w.w * t + w.phi); }
  return h;
}
const f5 = v => v.toFixed(5);
const WAVE_GLSL = `
const int NW = ${WAVES.length};
const int NVW = ${VERTEX_WAVES};
const vec4 WA[NW] = vec4[NW](${WAVES.map(w => `vec4(${f5(w.dx)}, ${f5(w.dz)}, ${f5(w.k)}, ${f5(w.A)})`).join(", ")});
const vec3 WB[NW] = vec3[NW](${WAVES.map(w => `vec3(${f5(w.w)}, ${f5(w.phi)}, ${f5(w.Q)})`).join(", ")});
`;
// Wind ripples: a tileable normal map built from a power-law spectrum of sines on integer wave
// vectors (so it wraps seamlessly), biased along the wind. Mipmapped, so distant water averages
// its normals; the shortened average is read back as extra roughness (Toksvig).
function rippleNormals(){
  const N = 256, waves = [];
  let sd = 7; const r = () => ((sd = (sd * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 64; i++){
    let kx, ky; do { kx = Math.round((r() * 2 - 1) * 30); ky = Math.round((r() * 2 - 1) * 30); } while (kx * kx + ky * ky < 5);
    const k = Math.hypot(kx, ky), along = Math.abs(ky) / k;
    waves.push([kx, ky, Math.pow(k, -1.8) * (.3 + .7 * along * along), r() * Math.PI * 2]);
  }
  const gx = new Float32Array(N * N), gy = new Float32Array(N * N); let sum2 = 0;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++){
    let ax = 0, ay = 0;
    for (const [kx, ky, a, p] of waves){ const c = a * 2 * Math.PI / N * Math.cos(2 * Math.PI * (kx * x + ky * y) / N + p); ax += c * kx; ay += c * ky; }
    gx[y * N + x] = ax; gy[y * N + x] = ay; sum2 += ax * ax + ay * ay;
  }
  const scale = .22 / Math.sqrt(sum2 / (N * N)), data = new Uint8Array(N * N * 4);   // rms slope ~0.22
  for (let i = 0; i < N * N; i++){
    const nx = -gx[i] * scale, ny = -gy[i] * scale, l = Math.hypot(nx, ny, 1);
    data[i * 4] = (nx / l * .5 + .5) * 255; data[i * 4 + 1] = (ny / l * .5 + .5) * 255; data[i * 4 + 2] = (1 / l * .5 + .5) * 255; data[i * 4 + 3] = 255;
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter; t.needsUpdate = true;
  return t;
}
// Water mesh: a grid whose spacing is fine in front of the camera and coarse towards the horizon.
function waterGrid(){
  const NX = 280, NZ = 220, pos = new Float32Array((NX + 1) * (NZ + 1) * 3), idx = [];
  const warp = u => .25 * u + .75 * u ** 5;                   // u in [-1, 1]
  let k = 0;
  for (let j = 0; j <= NZ; j++){
    const z = 45 + 650 * warp(j / NZ * 2 - 1);
    for (let i = 0; i <= NX; i++){ pos[k++] = 620 * warp(i / NX * 2 - 1); pos[k++] = 0; pos[k++] = z; }
  }
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++){
    const a = j * (NX + 1) + i, b = a + 1, c = a + NX + 1, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(idx);
  g.computeBoundingSphere(); g.boundingSphere.radius += 2;
  return g;
}
const MAX_OB = 24, MAX_BOATS = 12, MAX_RIPPLES = 4;
const WATER_VS = /* glsl */`
uniform float time;
varying vec3 vWorld;
varying vec2 vGrid;
varying float vCrest;
#include <common>
#include <fog_pars_vertex>
${WAVE_GLSL}
void main(){
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vec2 p = wp.xz;
  float fade = 1.0 - smoothstep(180.0, 420.0, length(p - cameraPosition.xz));
  vec3 d = vec3(0.0);
  float crest = 0.0;
  for (int i = 0; i < NVW; i++){
    float ph = WA[i].z * dot(WA[i].xy, p) - WB[i].x * time + WB[i].y;
    float s = sin(ph), c = cos(ph);
    d.xz += WB[i].z * WA[i].w * WA[i].xy * c;
    d.y += WA[i].w * s;
    crest += WA[i].w * s;
  }
  wp.xyz += d * fade;
  vGrid = p; vWorld = wp.xyz; vCrest = crest * fade;
  vec4 mvPosition = viewMatrix * wp;
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}`;
const WATER_FS = /* glsl */`
#define MAX_OB ${MAX_OB}
#define MAX_BOATS ${MAX_BOATS}
#define MAX_RIPPLES ${MAX_RIPPLES}
uniform sampler2D mirror;
uniform mat4 textureMatrix;
uniform sampler2D ripplesTex;
uniform float time;
uniform vec3 moonDir;
uniform vec3 moonColor;         // the moon's irradiance (a small, bright disc)
uniform vec3 deepColor;
uniform vec3 scatterColor;
uniform vec3 ambient;
uniform vec4 obA[MAX_OB];        // centre x, z, half extents along the box axes
uniform vec2 obB[MAX_OB];        // the box's x axis in world x/z
uniform int obN;
uniform vec4 boatA[MAX_BOATS];   // position x, z, heading x, z
uniform vec4 boatB[MAX_BOATS];   // half length, half beam, speed (m/s)
uniform int boatN;
uniform vec4 ripples[MAX_RIPPLES]; // x, z, age (s), strength
varying vec3 vWorld;
varying vec2 vGrid;
varying float vCrest;
#include <common>
#include <fog_pars_fragment>
#include <lights_pars_begin>
${WAVE_GLSL}

float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p){ return 0.55 * vnoise(p) + 0.3 * vnoise(p * 2.13 + 7.1) + 0.15 * vnoise(p * 4.37 + 3.3); }

// Boat hulls, bow waves, Kelvin wakes and splash rings: x = surface height (m), y = foam
vec2 dynamicSurface(vec2 p){
  float h = 0.0, foam = 0.0;
  for (int i = 0; i < MAX_BOATS; i++){
    if (i >= boatN) break;
    vec2 d = p - boatA[i].xy, f = boatA[i].zw;
    float hl = boatB[i].x, hb = boatB[i].y, sp = boatB[i].z;
    float reach = hl + 8.0 + sp * 10.0;
    if (dot(d, d) > reach * reach) continue;
    float along = dot(d, f), lat = dot(d, vec2(-f.y, f.x));
    float t = clamp(along, -hl + hb, hl - hb);
    float dh = length(d - f * t) - hb;                         // distance from the hull (a capsule)
    float spf = clamp(sp / 8.0, 0.0, 1.0);
    float bow = smoothstep(-hl * 0.2, hl, along);
    float near = exp(-max(dh, 0.0) / (0.5 + 0.8 * spf));
    foam += near * (0.3 + 0.9 * spf * (0.35 + bow));             // waterline froth, heaviest at the bow
    h += 0.22 * spf * bow * exp(-max(dh, 0.0) / 1.1);           // bow wave
    float behind = -along - hl;                                // distance astern of the transom
    if (behind > 0.0 && sp > 0.3){
      float spread = 0.354 * behind + hb * 0.7;                // Kelvin wedge: 19.47 degrees
      float fade = exp(-behind / (10.0 + sp * 5.0));
      float arm = abs(abs(lat) - spread);
      foam += spf * fade * exp(-arm / (0.45 + 0.07 * behind)) * 1.1;                                  // the two wake arms
      foam += spf * exp(-behind / (6.0 + sp * 2.5)) * exp(-abs(lat) / (hb * 0.8 + 0.1 * behind)) * 1.5;  // prop wash
      float inside = 1.0 - smoothstep(spread - 0.5, spread + 1.5, abs(lat));
      h += spf * fade * (0.1 * inside * sin(1.5 * (0.6 * behind + 0.8 * abs(lat))) + 0.14 * exp(-arm / 0.8));
    }
  }
  for (int i = 0; i < MAX_RIPPLES; i++){
    vec4 r = ripples[i];
    if (r.w <= 0.0) continue;
    float d = length(p - r.xy), front = r.z * 5.5, env = exp(-r.z * 0.5) * r.w;
    float bx = (d - front) / (1.2 + r.z * 0.5), band = exp(-bx * bx);
    h += env * 0.35 * band * sin(d * 2.8 - r.z * 13.0);
    foam += env * (band * 0.9 * exp(-r.z * 0.6) + exp(-d / (1.5 + r.z)) * exp(-r.z * 0.9) * 1.4);
  }
  return vec2(h, foam);
}
float obstacleDist(vec2 p){                                   // distance to piers, towers, piles and quays
  float dmin = 1e4;
  for (int i = 0; i < MAX_OB; i++){
    if (i >= obN) break;
    vec2 q = p - obA[i].xy;
    q = vec2(dot(q, obB[i]), dot(q, vec2(-obB[i].y, obB[i].x)));
    vec2 e = abs(q) - obA[i].zw;
    dmin = min(dmin, length(max(e, 0.0)) + min(max(e.x, e.y), 0.0));
  }
  return dmin;
}
float ggxSpec(vec3 N, vec3 V, vec3 L, float a){                // GGX, Smith-Schlick, Fresnel (F0 = 0.02); times N.L
  vec3 H = normalize(L + V);
  float NL = max(dot(N, L), 0.0), NV = max(dot(N, V), 1e-3), NH = max(dot(N, H), 0.0), VH = max(dot(V, H), 0.0);
  float a2 = a * a, dd = NH * NH * (a2 - 1.0) + 1.0, D = a2 / (PI * dd * dd);
  float k = a * 0.5, G = NL / (NL * (1.0 - k) + k) * NV / (NV * (1.0 - k) + k);
  float F = 0.02 + 0.98 * pow(clamp(1.0 - VH, 0.0, 1.0), 5.0);
  return D * G * F / (4.0 * NV * max(NL, 1e-3)) * NL;
}

void main(){
  vec3 toEye = cameraPosition - vWorld;
  float dist = length(toEye);
  vec3 V = toEye / dist;
  vec2 p = vGrid;
  float foot = max(length(dFdx(p)), length(dFdy(p))) + 1e-4;   // metres of water under this pixel

  // 1. swell: analytic Gerstner slopes; waves too short for the pixel fade out and become roughness
  vec2 slope = vec2(0.0);
  float ny = 1.0, variance = 0.0;
  for (int i = 0; i < NW; i++){
    float L = 6.2831853 / WA[i].z;
    float lod = smoothstep(1.5, 5.0, L / foot);
    float ph = WA[i].z * dot(WA[i].xy, p) - WB[i].x * time + WB[i].y;
    float kA = WA[i].z * WA[i].w;
    slope += WA[i].xy * kA * cos(ph) * lod;
    ny -= WB[i].z * kA * sin(ph) * lod;
    variance += 0.5 * kA * kA * (1.0 - lod);
  }
  slope /= max(ny, 0.3);

  // 2. wind ripples: two drifting layers of the tileable normal map (Toksvig roughness from the mips)
  vec3 t1 = texture2D(ripplesTex, p / 9.0 + time * vec2(0.013, 0.021)).xyz * 2.0 - 1.0;
  vec3 t2 = texture2D(ripplesTex, mat2(0.8, -0.6, 0.6, 0.8) * p / 3.4 + time * vec2(-0.031, 0.017)).xyz * 2.0 - 1.0;
  float l1 = length(t1), l2 = length(t2);
  slope -= 0.7 * t1.xy / max(t1.z, 0.2) + 0.4 * t2.xy / max(t2.z, 0.2);
  variance += 0.5 * (0.49 * (1.0 - l1) / l1 + 0.16 * (1.0 - l2) / l2);

  // 3. boats and splashes: a height field, differentiated numerically
  vec2 dyn = dynamicSurface(p);
  const float E = 0.18;
  slope += vec2(dynamicSurface(p + vec2(E, 0.0)).x - dyn.x, dynamicSurface(p + vec2(0.0, E)).x - dyn.x) / E;

  vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));
  float NV = dot(N, V);
  if (NV < 0.02){ N = normalize(N + V * (0.02 - NV)); NV = 0.02; }   // no back-facing micro-normals at grazing angles
  float rough = sqrt(0.0025 + variance);

  // 4. reflection: planar mirror, looked up along the perturbed reflected ray against a proxy
  //    distance (so ripples shift the image physically: mostly vertically at grazing angles), then
  //    blurred vertically by the roughness to draw the long light pillars of night-time water
  vec3 R = reflect(-V, N); R.y = abs(R.y);
  vec4 pc = textureMatrix * vec4(vWorld + R * clamp(dist * 0.3, 5.0, 40.0), 1.0);
  vec2 uv = pc.xy / pc.w;
  float spread = clamp(rough * 0.03, 0.001, 0.012);
  vec3 refl = texture2D(mirror, clamp(uv, 0.001, 0.999)).rgb * 0.36;
  refl += texture2D(mirror, clamp(uv + vec2(0.0, spread), 0.001, 0.999)).rgb * 0.2;
  refl += texture2D(mirror, clamp(uv - vec2(0.0, spread), 0.001, 0.999)).rgb * 0.2;
  refl += texture2D(mirror, clamp(uv + vec2(0.0, spread * 2.5), 0.001, 0.999)).rgb * 0.12;
  refl += texture2D(mirror, clamp(uv - vec2(0.0, spread * 2.5), 0.001, 0.999)).rgb * 0.12;
  float F = 0.02 + 0.98 * pow(clamp(1.0 - NV, 0.0, 1.0), 5.0);

  // 5. direct light: the moon's glitter path, plus a GGX glint and underwater glow for every lamp
  vec3 spec = moonColor * ggxSpec(N, V, moonDir, max(rough, 0.045));
  vec3 vPos = (viewMatrix * vec4(vWorld, 1.0)).xyz, vN = normalize((viewMatrix * vec4(N, 0.0)).xyz), vV = normalize(-vPos);
  vec3 vUp = normalize((viewMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
  float aL = max(rough, 0.06);
  vec3 irradiance = ambient;
  IncidentLight il;
  #if NUM_POINT_LIGHTS > 0
  #pragma unroll_loop_start
  for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
    getPointLightInfo( pointLights[ i ], vPos, il );
    spec += il.color * ggxSpec( vN, vV, il.direction, aL );
    irradiance += il.color * max( dot( vUp, il.direction ), 0.0 );
  }
  #pragma unroll_loop_end
  #endif
  #if NUM_SPOT_LIGHTS > 0
  #pragma unroll_loop_start
  for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
    getSpotLightInfo( spotLights[ i ], vPos, il );
    spec += il.color * ggxSpec( vN, vV, il.direction, aL );
    irradiance += il.color * max( dot( vUp, il.direction ), 0.0 );
  }
  #pragma unroll_loop_end
  #endif
  spec = min(spec, vec3(12.0));

  // 6. the water body: nearly black, with light scattered back out under lamps and through crests
  float crest = pow(clamp(vCrest * 2.2 + 0.2, 0.0, 1.0), 2.0);
  vec3 body = deepColor + scatterColor * (irradiance * 0.05 + 0.1 * crest * ambient);

  // 7. foam: lapping at every pier and quay, around hulls, in wakes and splash rings
  float od = obstacleDist(p);
  float n1 = fbm(p * 0.9 + vec2(time * 0.11, -time * 0.07)), n2 = fbm(p * 2.6 - vec2(time * 0.05, time * 0.23));
  float lap = exp(-max(od, 0.0) / 2.2) * (0.7 + 0.3 * sin(od * 2.0 - time * 1.9 + n1 * 4.0));
  float foam = smoothstep(0.0, 0.5, (lap * 1.2 + dyn.y) * (n1 + 0.45) - 0.4 * n2 - 0.06);
  foam *= 1.0 - smoothstep(250.0, 450.0, dist);
  vec3 foamCol = vec3(0.8, 0.86, 0.94) * (irradiance * 2.0 + 0.03);   // bright, rough and diffuse

  vec3 col = F * refl + (1.0 - F) * body + spec * (1.0 - foam);
  col = mix(col, foamCol, foam * 0.9);
  if (any(isnan(col)) || any(isinf(col))) col = vec3(0.0);
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}`;

// ---------------------------------------------------------------- procedural textures
function canvasTex(w, h, draw, srgb = true){
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
const towerFacade = () => canvasTex(256, 320, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, w, 0);
  gr.addColorStop(0, "#5aa895"); gr.addColorStop(.5, "#7cc8b3"); gr.addColorStop(1, "#4d9584");
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
  g.globalAlpha = .18; g.fillStyle = "#123b33";                       // weathering
  for (let i = 0; i < 70; i++) g.fillRect(rand() * w, rand() * h, 6 + rand() * 30, 2 + rand() * 10);
  g.globalAlpha = .12; g.fillStyle = "#e8fff8";
  for (let i = 0; i < 30; i++) g.fillRect(rand() * w, rand() * h, 2 + rand() * 12, 20 + rand() * 60);
  g.globalAlpha = .35; g.strokeStyle = "#1f4a41"; g.lineWidth = 2;   // cladding seams
  for (let x = 32; x < w; x += 40){ g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
  for (let y = 40; y < h; y += 56){ g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  g.globalAlpha = 1;
  g.fillStyle = "#2e5ea8"; g.fillRect(w * .62, h * .28, w * .2, h * .5);
  g.fillStyle = "#1d3f7a"; g.strokeStyle = "#e8c45a"; g.lineWidth = 4;     // borough coat of arms
  g.beginPath(); g.moveTo(w * .18, h * .3); g.lineTo(w * .44, h * .3); g.lineTo(w * .44, h * .45);
  g.quadraticCurveTo(w * .44, h * .56, w * .31, h * .6); g.quadraticCurveTo(w * .18, h * .56, w * .18, h * .45); g.closePath(); g.fill(); g.stroke();
  g.fillStyle = "#e8c45a"; g.fillRect(w * .22, h * .36, w * .18, 5); g.fillRect(w * .22, h * .42, w * .18, 5);
  g.fillStyle = "#3d6e62"; g.fillRect(w * .3, h * .82, w * .14, h * .18);
});
const roadTex = (w, lanes = true) => {
  const t = canvasTex(512, 128, (g, cw, ch) => {
    g.fillStyle = "#2a2e36"; g.fillRect(0, 0, cw, ch);
    g.globalAlpha = .25; for (let i = 0; i < 900; i++){ g.fillStyle = rand() < .5 ? "#3a3f4a" : "#1c1f25"; g.fillRect(rand() * cw, rand() * ch, 2, 2); }
    g.globalAlpha = 1;
    g.fillStyle = "#8d94a3"; g.fillRect(0, 0, cw, ch * .12); g.fillRect(0, ch * .88, cw, ch * .12);       // footpaths
    g.fillStyle = "#e9edf3"; g.fillRect(0, ch * .15, cw, 3); g.fillRect(0, ch * .85 - 3, cw, 3);           // edge lines
    if (lanes) for (let x = 0; x < cw; x += 64) g.fillRect(x, ch / 2 - 2, 36, 4);                          // centre dashes
  });
  t.wrapS = THREE.RepeatWrapping; return t;
};
const stripeTex = () => canvasTex(128, 16, (g, w, h) => {
  g.fillStyle = "#f4f6fa"; g.fillRect(0, 0, w, h); g.fillStyle = "#d7263d";
  for (let x = -h; x < w; x += 32){ g.beginPath(); g.moveTo(x, h); g.lineTo(x + 16, 0); g.lineTo(x + 32, 0); g.lineTo(x + 16, h); g.fill(); }
});
const chevronTex = () => canvasTex(128, 64, (g, w, h) => {
  g.fillStyle = "#f2c230"; g.fillRect(0, 0, w, h); g.fillStyle = "#111";
  for (let x = -h; x < w; x += 28){ g.beginPath(); g.moveTo(x, h); g.lineTo(x + h, 0); g.lineTo(x + h + 12, 0); g.lineTo(x + 12, h); g.fill(); }
});
const warnSignTex = () => canvasTex(128, 128, (g, w) => {
  g.clearRect(0, 0, w, w);
  g.beginPath(); g.moveTo(w / 2, 8); g.lineTo(w - 8, w - 14); g.lineTo(8, w - 14); g.closePath();
  g.fillStyle = "#fff"; g.fill(); g.lineWidth = 14; g.strokeStyle = "#d7263d"; g.lineJoin = "round"; g.stroke();
  g.fillStyle = "#111"; g.fillRect(w * .3, w * .62, w * .4, 6); g.fillRect(w * .36, w * .5, w * .12, 12); g.fillRect(w * .52, w * .5, w * .12, 12);
});
function facadeTex(cols, rows, base){
  const W = 256, H = 256, lit = [];
  const t = canvasTex(W, H, (g) => {
    g.fillStyle = base; g.fillRect(0, 0, W, H);
    const cw = W / cols, rh = H / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++){
      const on = rand() < .38; lit.push(on);
      g.fillStyle = on ? (rand() < .7 ? "#ffd58a" : "#cfe6ff") : "#0b1020";
      g.fillRect(c * cw + cw * .22, r * rh + rh * .25, cw * .56, rh * .5);
    }
  });
  let k = 0;
  const e = canvasTex(W, H, (g) => {
    g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
    const cw = W / cols, rh = H / rows;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++){
      if (lit[k++]){ g.fillStyle = "#ffe0a0"; g.fillRect(c * cw + cw * .22, r * rh + rh * .25, cw * .56, rh * .5); }
    }
  });
  return { map: t, emissiveMap: e };
}
const stoneTex = () => { const t = canvasTex(256, 64, (g, w, h) => {
  g.fillStyle = "#3b3f48"; g.fillRect(0, 0, w, h);
  for (let y = 0; y < h; y += 16) for (let x = (y / 16) % 2 ? -16 : 0; x < w; x += 32){
    g.fillStyle = `hsl(220, 8%, ${22 + rand() * 10}%)`; g.fillRect(x + 1, y + 1, 30, 14);
  }
}); t.wrapS = THREE.RepeatWrapping; return t; };

// ---------------------------------------------------------------- builders
const std = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: .7, metalness: .1, ...o });
const glow = (color, intensity = 3) => new THREE.MeshStandardMaterial({ color: 0x000000, emissive: color, emissiveIntensity: intensity });
function box(w, h, d, mat, x = 0, y = 0, z = 0){
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); return m;
}
function boxGeo(w, h, d, x = 0, y = 0, z = 0){ const g = new THREE.BoxGeometry(w, h, d); g.translate(x, y, z); return g; }
function beamGeo(a, b, t = .18, t2 = t){      // a box spanning point a -> point b
  const d = new THREE.Vector3().subVectors(b, a), len = d.length();
  const geo = new THREE.BoxGeometry(len, t, t2);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), d.normalize());
  geo.applyMatrix4(new THREE.Matrix4().compose(a.clone().add(b).multiplyScalar(.5), q, new THREE.Vector3(1, 1, 1)));
  return geo;
}
const merged = (geos, mat) => new THREE.Mesh(mergeGeometries(geos, false), mat);
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const fixed = o => { o.userData.fixed = true; return o; };     // foundations: charred by meteors, never knocked down
function orientBetween(mesh, a, b){          // cylinder (axis y, height 1) stretched from a to b
  const d = new THREE.Vector3().subVectors(b, a), len = d.length();
  mesh.position.copy(a).addScaledVector(d, .5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  mesh.scale.set(1, Math.max(.001, len), 1);
}
function lamp(x, z, deckY, color = 0xffd98a, withLight = false){
  const g = new THREE.Group(), metal = std(0x4a5263, { metalness: .6, roughness: .4 });
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(.09, .14, 5, 10), metal); pole.position.set(x, deckY + 2.5, z); g.add(pole);
  const arm = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(V(x, deckY + 4.9, z), V(x, deckY + 5.6, z - Math.sign(z) * .2), V(x, deckY + 5.3, z - Math.sign(z) * 1.4)), 10, .06, 6), metal); g.add(arm);
  const hood = new THREE.Mesh(new THREE.CylinderGeometry(.18, .38, .28, 12), metal); hood.position.set(x, deckY + 5.2, z - Math.sign(z) * 1.4); g.add(hood);
  const head = new THREE.Mesh(new THREE.CylinderGeometry(.32, .32, .06, 12), glow(color, 7)); head.position.set(x, deckY + 5.05, z - Math.sign(z) * 1.4); g.add(head);
  if (withLight){ const l = new THREE.PointLight(color, 22, 18, 2); l.position.set(x, deckY + 4.8, z - Math.sign(z) * 1.4); g.add(l); }
  return g;
}
function makeCar(color){
  const g = new THREE.Group();
  const s = new THREE.Shape();
  s.moveTo(-2.1, .32); s.lineTo(2.02, .32); s.quadraticCurveTo(2.16, .38, 2.14, .72); s.lineTo(1.98, .86);
  s.quadraticCurveTo(1.3, .96, .88, .99); s.lineTo(.32, 1.44); s.lineTo(-1.18, 1.46); s.quadraticCurveTo(-1.72, 1.4, -1.98, 1.0);
  s.lineTo(-2.12, .92); s.closePath();
  const body = new THREE.ExtrudeGeometry(s, { depth: 1.66, bevelEnabled: true, bevelThickness: .06, bevelSize: .05, bevelSegments: 2 });
  body.translate(0, 0, -.83);
  g.add(new THREE.Mesh(body, new THREE.MeshPhysicalMaterial({ color, roughness: .32, metalness: .55, clearcoat: .9, clearcoatRoughness: .15 })));
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x0c1422, roughness: .08, metalness: .3, clearcoat: 1 });
  const gs = new THREE.Shape(); gs.moveTo(.8, 1.0); gs.lineTo(.3, 1.38); gs.lineTo(-1.12, 1.4); gs.quadraticCurveTo(-1.6, 1.35, -1.82, 1.02); gs.closePath();
  const gg = new THREE.ExtrudeGeometry(gs, { depth: 1.84, bevelEnabled: false }); gg.translate(0, 0, -.92);
  g.add(new THREE.Mesh(gg, glass));
  for (const z of [.58, -.58]){
    g.add(box(.06, .16, .38, glow(0xfff6c8, 6), 2.12, .76, z));
    g.add(box(.06, .14, .36, glow(0xff2a3a, 4), -2.12, .82, z));
  }
  const tyre = std(0x0b0f18, { roughness: .9 }), rim = std(0xb8c0cc, { metalness: .8, roughness: .3 });
  for (const [x, z] of [[1.3, .86], [1.3, -.86], [-1.3, .86], [-1.3, -.86]]){
    const w = new THREE.Mesh(new THREE.CylinderGeometry(.36, .36, .24, 16), tyre); w.rotation.x = Math.PI / 2; w.position.set(x, .36, z); g.add(w);
    const r = new THREE.Mesh(new THREE.CylinderGeometry(.2, .2, .25, 10), rim); r.rotation.x = Math.PI / 2; r.position.set(x, .36, z); g.add(r);
  }
  return g;
}
const BOAT_DIMS = { yacht: [9.4, 2.8, 13.3], superyacht: [18.4, 4.2, 6.4], trawler: [11.4, 3.4, 9.2], tug: [9.4, 3.6, 5.1], rib: [5.9, 2.2, 1.7], launch: [6.9, 2.4, 2.2] };
function makeBoat(kind){
  const g = new THREE.Group();
  g.dims = { len: BOAT_DIMS[kind][0], beam: BOAT_DIMS[kind][1], h: BOAT_DIMS[kind][2] };
  const hull = (len, beam, h, color) => {
    const s = new THREE.Shape(); s.moveTo(-len / 2, -beam / 2); s.lineTo(len * .3, -beam / 2); s.quadraticCurveTo(len / 2, -beam * .3, len / 2 + .4, 0);
    s.quadraticCurveTo(len / 2, beam * .3, len * .3, beam / 2); s.lineTo(-len / 2, beam / 2); s.closePath();
    const geo = new THREE.ExtrudeGeometry(s, { depth: h, bevelEnabled: true, bevelThickness: .08, bevelSize: .08, bevelSegments: 2 }); geo.rotateX(-Math.PI / 2);
    return new THREE.Mesh(geo, std(color, { roughness: .35, metalness: .15 }));
  };
  const nav = (x, z, c) => { const m = new THREE.Mesh(new THREE.SphereGeometry(.14, 8, 6), glow(c, 3.5)); m.position.set(x, 1.4, z); g.add(m); };
  const rail = (len, y, beam) => { for (const z of [beam / 2 - .1, -beam / 2 + .1]) g.add(box(len, .05, .05, std(0xc9d2e0, { metalness: .8, roughness: .3 }), -.3, y, z)); };
  if (kind === "yacht"){
    g.add(hull(9, 2.8, 1.1, 0xe9eef6)); g.add(box(3, .7, 1.8, std(0xcdd6e4), -.5, 1.45, 0)); g.add(box(3.05, .22, 1.85, glow(0xffe2b8, 1.1), -.5, 1.5, 0));
    g.add(box(.12, 12, .12, std(0xdfe6ef, { metalness: .6 }), .2, 7, 0)); g.add(box(4, .08, .08, std(0xdfe6ef), -1.8, 2.4, 0));
    const sail = new THREE.Shape(); sail.moveTo(0, 0); sail.lineTo(0, 10.5); sail.lineTo(-3.8, 0); sail.closePath();
    const sm = new THREE.Mesh(new THREE.ShapeGeometry(sail), std(0xf4f7fb, { side: THREE.DoubleSide, roughness: .9 })); sm.position.set(.1, 1.6, 0); g.add(sm);
    const top = new THREE.Mesh(new THREE.SphereGeometry(.16, 8, 6), glow(0xffffff, 8)); top.position.set(.2, 13, 0); g.add(top);
    rail(6, 1.6, 2.8);
  } else if (kind === "superyacht"){
    g.add(hull(18, 4.2, 1.6, 0xf4f7fb)); g.add(box(11, 1.4, 3.4, std(0xe9eef6), -1, 2.3, 0)); g.add(box(6, 1.1, 2.8, std(0xdfe6ef), -1.5, 3.5, 0));
    g.add(box(11.2, .45, 3.5, glow(0x7fe3ff, 1.5), -1, 2.1, 0)); g.add(box(6.1, .35, 2.9, std(0x0d1a33, { roughness: .1, metalness: .6 }), -1.5, 3.6, 0));
    g.add(box(.3, 1.6, .3, std(0xdfe6ef), -2, 4.8, 0)); rail(12, 3.1, 4.2);
    const under = new THREE.PointLight(0x2de2ff, 30, 14, 2); under.position.set(0, -.5, 0); g.add(under);
  } else if (kind === "trawler"){
    g.add(hull(11, 3.4, 1.8, 0xa8322d)); g.add(box(3, 2.2, 2.6, std(0xe9eef6), -2.5, 2.9, 0)); g.add(box(3.1, .5, 2.7, glow(0xffe2b8, 1.3), -2.5, 3.1, 0));
    g.add(box(.14, 6, .14, std(0xc9d2e0), 1.5, 4.8, 0)); g.add(merged([beamGeo(V(1.5, 7.5, 0), V(-2, 9, 1.8), .08), beamGeo(V(1.5, 7.5, 0), V(-2, 9, -1.8), .08)], std(0xc9d2e0)));
    const flood = new THREE.PointLight(0xfff6c8, 22, 14, 2); flood.position.set(1.5, 7, 0); g.add(flood);
  } else if (kind === "tug"){
    g.add(hull(9, 3.6, 1.7, 0x1a1d27)); g.add(box(3.2, 1.8, 2.6, std(0xe56a2c), -.5, 2.6, 0)); g.add(box(.9, 1.6, .9, std(0x1a1d27), -.5, 4.2, 0));
    g.add(box(3.3, .4, 2.7, glow(0xffe2b8, 1.3), -.5, 2.8, 0));
    for (const x of [-3, -1, 1, 3]) for (const z of [1.85, -1.85]){ const t = new THREE.Mesh(new THREE.TorusGeometry(.32, .14, 8, 12), std(0x222222)); t.position.set(x, 1.2, z); g.add(t); }
  } else if (kind === "rib"){
    g.add(hull(5.5, 2.2, .7, 0x2a2f3c)); g.add(box(1, .9, .9, std(0xcfd6e2), -.3, 1.1, 0));
    for (const z of [1, -1]) g.add(box(4.6, .4, .4, std(0xff7a3c, { roughness: .6 }), -.3, .7, z));
  } else if (kind === "launch"){
    g.add(hull(6.5, 2.4, .9, 0xe9eef6)); g.add(box(2.6, 1, 1.8, std(0x2b5fb3), -.6, 1.3, 0)); g.add(box(2.7, .3, 1.9, glow(0xffe2b8, 1.1), -.6, 1.5, 0));
  }
  nav(0, 1.2, 0xff3040); nav(0, -1.2, 0x41ff8b);
  return g;
}
let facadeMats = null;
function makeBuilding(w, h, d, x, z, rotY = 0, pitched = false){
  const g = new THREE.Group();
  if (!facadeMats) facadeMats = [[5, 4, "#1a2238"], [7, 6, "#232a3f"], [4, 3, "#1d2640"], [8, 5, "#20283d"]].map(([c, r, base]) => {
    const f = facadeTex(c, r, base);
    return new THREE.MeshStandardMaterial({ map: f.map, emissiveMap: f.emissiveMap, emissive: 0xffffff, emissiveIntensity: 1.6, roughness: .8 });
  });
  const faceMat = facadeMats[Math.floor(rand() * facadeMats.length)];
  const side = std(0x161d31, { roughness: .9 });
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [side, side, std(0x10162a), side, faceMat, side]);
  m.position.y = h / 2; g.add(m);
  if (pitched){
    const s = new THREE.Shape(); s.moveTo(-w / 2 - .3, 0); s.lineTo(0, w * .22); s.lineTo(w / 2 + .3, 0); s.closePath();
    const roof = new THREE.Mesh(new THREE.ExtrudeGeometry(s, { depth: d + .6, bevelEnabled: false }), std(0x2a3346, { roughness: .6, metalness: .3 }));
    roof.position.set(0, h, -d / 2 - .3); g.add(roof);
  } else g.add(box(w + .3, .4, d + .3, std(0x0d1222), 0, h + .2, 0));
  g.position.set(x, 0, z); g.rotation.y = rotY;
  return g;
}

// ---------------------------------------------------------------- the scene
export function create(container){
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);
  Object.assign(renderer.domElement.style, { display: "block", width: "100%", height: "100%" });

  const scene = new THREE.Scene();
  // sky gradient, repainted as it turns blood red in alarm mode (k: 0 = night, 1 = red)
  const skyCanvas = document.createElement("canvas"); skyCanvas.width = 4; skyCanvas.height = 256;
  const skyCtx = skyCanvas.getContext("2d"), skyTexture = new THREE.CanvasTexture(skyCanvas);
  skyTexture.colorSpace = THREE.SRGBColorSpace;
  const SKY_STOPS = [[0, 0x02040c, 0x160205], [.6, 0x060b22, 0x520a0e], [.85, 0x0c1433, 0x951a12], [1, 0x131c42, 0xd9491c]];
  const skyA = new THREE.Color(), skyB = new THREE.Color();
  function paintSky(k){
    const gr = skyCtx.createLinearGradient(0, 0, 0, 256);
    for (const [o, a, b] of SKY_STOPS) gr.addColorStop(o, "#" + skyA.setHex(a).lerp(skyB.setHex(b), k).getHexString());
    skyCtx.fillStyle = gr; skyCtx.fillRect(0, 0, 4, 256); skyTexture.needsUpdate = true;
  }
  paintSky(0);
  scene.background = skyTexture;
  scene.fog = new THREE.FogExp2(0x080d26, 0.0075);
  const camera = new THREE.PerspectiveCamera(24, 4, .5, 2500);

  const hemi = new THREE.HemisphereLight(0x5566a8, 0x0a0c18, .9); scene.add(hemi);
  const moonLight = new THREE.DirectionalLight(0xc6d4ff, 1.6); moonLight.position.set(-60, 90, 40); scene.add(moonLight);

  // moon, halo, stars
  const moon = new THREE.Mesh(new THREE.SphereGeometry(9, 32, 16), new THREE.MeshBasicMaterial({ color: 0xfff6dc, fog: false }));
  moon.position.set(60, 95, -420); scene.add(moon);
  moon.layers.set(1); camera.layers.enable(1);        // layer 1: seen by the camera, not by the water's mirror
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(128, 128, (g, w) => {
    const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    r.addColorStop(0, "rgba(255,246,220,.55)"); r.addColorStop(.3, "rgba(255,246,220,.12)"); r.addColorStop(1, "rgba(255,246,220,0)");
    g.fillStyle = r; g.fillRect(0, 0, w, w); }), transparent: true, depthWrite: false, fog: false }));
  halo.scale.set(120, 120, 1); halo.position.copy(moon.position); halo.layers.set(1); scene.add(halo);
  const starGeo = new THREE.BufferGeometry(), starPos = [];
  for (let i = 0; i < 700; i++){
    const th = rand() * Math.PI * 2, ph = rand() * Math.PI * .42 + .08, r = 1400;
    starPos.push(r * Math.cos(th) * Math.cos(ph), r * Math.sin(ph), r * Math.sin(th) * Math.cos(ph));
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
  const starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, sizeAttenuation: false, fog: false, transparent: true, opacity: .8 });
  scene.add(new THREE.Points(starGeo, starMat));

  // water: Gerstner swell on an adaptive grid, a half-float planar mirror, GGX glints from the moon
  // and every lamp, Fresnel, and foam from signed distances to the piers, hulls and wakes
  const lin = (r, g, b) => new THREE.Color().setRGB(r, g, b);
  const mirrorRT = new THREE.WebGLRenderTarget(512, 256, { type: THREE.HalfFloatType });
  const textureMatrix = new THREE.Matrix4(), mirrorCam = new THREE.PerspectiveCamera();
  const rippleTex = rippleNormals(); rippleTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const wu = THREE.UniformsUtils.merge([THREE.UniformsLib.fog, THREE.UniformsLib.lights, {
    time: { value: 0 }, textureMatrix: { value: null }, mirror: { value: null }, ripplesTex: { value: null },
    moonDir: { value: V(0, 0, 0) }, moonColor: { value: lin(.07, .064, .05) }, deepColor: { value: lin(.0015, .004, .009) },
    scatterColor: { value: lin(.02, .12, .1) }, ambient: { value: lin(.045, .055, .085) },
    obA: { value: [] }, obB: { value: [] }, obN: { value: 0 }, boatA: { value: [] }, boatB: { value: [] }, boatN: { value: 0 }, ripples: { value: [] },
  }]);
  wu.textureMatrix.value = textureMatrix; wu.mirror.value = mirrorRT.texture; wu.ripplesTex.value = rippleTex;
  wu.moonDir.value.copy(moon.position).normalize();
  wu.obA.value = Array.from({ length: MAX_OB }, () => new THREE.Vector4()); wu.obB.value = Array.from({ length: MAX_OB }, () => new THREE.Vector2(1, 0));
  wu.boatA.value = Array.from({ length: MAX_BOATS }, () => new THREE.Vector4()); wu.boatB.value = Array.from({ length: MAX_BOATS }, () => new THREE.Vector4());
  wu.ripples.value = Array.from({ length: MAX_RIPPLES }, () => new THREE.Vector4(0, 0, 0, 0));   // w = strength; 0 = unused
  const water = new THREE.Mesh(waterGrid(), renderer.capabilities.isWebGL2
    ? new THREE.ShaderMaterial({ uniforms: wu, vertexShader: WATER_VS, fragmentShader: WATER_FS, fog: true, lights: true })
    : std(0x020915, { roughness: .2, metalness: .6 }));                      // (WebGL1 fallback)
  scene.add(water);
  {
    // planar reflection (as three's Water/Reflector): mirror the camera in y = 0, clip below the plane
    const up = V(0, 1, 0), mwp = V(0, 0, 0), cwp = V(0, 0, 0), view = V(0, 0, 0), look = V(0, 0, 0), target = V(0, 0, 0);
    const rot = new THREE.Matrix4(), plane = new THREE.Plane(), clip = new THREE.Vector4(), q = new THREE.Vector4();
    water.onBeforeRender = (r, sc, cam) => {
      if (!water.material.isShaderMaterial) return;
      mwp.setFromMatrixPosition(water.matrixWorld); cwp.setFromMatrixPosition(cam.matrixWorld);
      view.subVectors(mwp, cwp); if (view.dot(up) > 0) return;
      view.reflect(up).negate().add(mwp);
      rot.extractRotation(cam.matrixWorld);
      look.set(0, 0, -1).applyMatrix4(rot).add(cwp);
      target.subVectors(mwp, look).reflect(up).negate().add(mwp);
      mirrorCam.position.copy(view); mirrorCam.up.set(0, 1, 0).applyMatrix4(rot).reflect(up); mirrorCam.lookAt(target);
      mirrorCam.far = cam.far; mirrorCam.updateMatrixWorld(); mirrorCam.projectionMatrix.copy(cam.projectionMatrix);
      textureMatrix.set(.5, 0, 0, .5, 0, .5, 0, .5, 0, 0, .5, .5, 0, 0, 0, 1).multiply(mirrorCam.projectionMatrix).multiply(mirrorCam.matrixWorldInverse);
      plane.setFromNormalAndCoplanarPoint(up, mwp).applyMatrix4(mirrorCam.matrixWorldInverse);
      clip.set(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
      const pm = mirrorCam.projectionMatrix.elements;
      q.set((Math.sign(clip.x) + pm[8]) / pm[0], (Math.sign(clip.y) + pm[9]) / pm[5], -1, (1 + pm[10]) / pm[14]);
      clip.multiplyScalar(2 / clip.dot(q));
      pm[2] = clip.x; pm[6] = clip.y; pm[10] = clip.z + 1; pm[14] = clip.w;
      const prev = r.getRenderTarget();
      water.visible = false;
      r.setRenderTarget(mirrorRT); r.state.buffers.depth.setMask(true);
      if (r.autoClear === false) r.clear();
      r.render(sc, mirrorCam);
      water.visible = true;
      r.setRenderTarget(prev);
      if (cam.viewport !== undefined) r.state.viewport(cam.viewport);
    };
  }
  let waterTime = 0;
  const ripples = wu.ripples.value;
  function addRipple(x, z, strength){                // splash / explosion rings (oldest slot reused)
    const slot = ripples.find(r => r.w <= 0) || ripples.reduce((a, r) => (r.z > a.z ? r : a));
    slot.set(x, z, 0, strength);
  }

  // far shore: a waterfront of buildings with lit windows
  const shore = new THREE.Group();
  for (let i = 0; i < 26; i++){
    const w = 12 + rand() * 22, h = 6 + rand() * 18;
    shore.add(makeBuilding(w, h, 10, -320 + i * 26 + rand() * 6, -150 - rand() * 30, 0, rand() < .3));
  }
  shore.add(box(900, 3, 60, std(0x080d1c), 0, 1.4, -180));
  scene.add(shore);

  const stone = stoneTex();
  function quay(g, x, deck, W, len){       // stone quay wall with bollards
    const t = stone.clone(); t.needsUpdate = true; t.repeat.set(len / 8, 1);
    g.add(fixed(box(len, deck + 1, W + 24, new THREE.MeshStandardMaterial({ map: t, roughness: .95 }), x, (deck + 1) / 2 - 1, 0)));
    const bol = new THREE.InstancedMesh(new THREE.CylinderGeometry(.22, .28, .7, 10), std(0x15181f, { metalness: .5 }), 6); const m4 = new THREE.Matrix4();
    for (let i = 0; i < 6; i++){ m4.makeTranslation(x - len / 2 + 1.5 + i * (len - 3) / 5, deck + .35, W / 2 + 11); bol.setMatrixAt(i, m4); }
    g.add(bol);
  }
  function roadPlane(g, len, W, x, y, lanes = true){
    const t = roadTex(W, lanes); t.repeat.set(len / 16, 1);
    const p = new THREE.Mesh(new THREE.PlaneGeometry(len, W), new THREE.MeshStandardMaterial({ map: t, roughness: .85 }));
    p.rotation.x = -Math.PI / 2; p.position.set(x, y + .005, 0); g.add(p); return p;
  }
  function barrier(g, x, deck, edgeZ, len){  // striped arm on a post; returns the pivot so it can be lowered
    const post = box(.35, 1.3, .35, std(0xe9edf3), x, deck + .65, edgeZ); g.add(post);
    const pivot = new THREE.Group(); pivot.position.set(x, deck + 1.15, edgeZ); g.add(pivot);
    const arm = new THREE.Mesh(new THREE.BoxGeometry(.12, .12, len), new THREE.MeshStandardMaterial({ map: stripeTex(), roughness: .5 }));
    arm.position.z = -Math.sign(edgeZ) * len / 2; pivot.add(arm);
    const lights = [];
    for (let i = 1; i <= 3; i++){ const l = new THREE.Mesh(new THREE.SphereGeometry(.07, 8, 6), glow(0x3a1016, 1)); l.position.set(0, .1, -Math.sign(edgeZ) * i * len / 4); pivot.add(l); lights.push(l); }
    pivot.userData = { up: Math.sign(edgeZ) * Math.PI / 2, lights };
    pivot.rotation.x = pivot.userData.up;
    return pivot;
  }

  // ------------------------------------------------------------ Twin Sails Bridge
  const twin = { key: "TWIN", group: new THREE.Group(), p: 0, from: 0, target: 0, t0: -99, maxDeg: 88, hinge: 8.8, L: 17.6, W: 12, deck: 6,
                 lanes: [3, -3], stops: [-17, 17], barrierP: 0, barriers: [], rams: [], warns: [] };
  {
    const g = twin.group, { L, W, deck } = twin;
    const concrete = std(0x9aa2b2, { roughness: .9 }), deckMat = std(0xe8edf5, { roughness: .45, metalness: .15 });
    const steel = std(0xc9d2de, { metalness: .75, roughness: .28 });
    for (const sx of [-1, 1]){
      const px = sx * (twin.hinge + 2.2);
      g.add(fixed(box(4.4, deck + 1, W + 2, concrete, px, (deck - 1) / 2 - .5, 0)));                                   // main pier
      for (const sz of [-1, 1]){                                                                                // rounded pier noses
        const nose = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, deck + 1, 20, 1, false, 0, Math.PI), concrete);
        nose.position.set(px, (deck - 1) / 2 - .5, sz * (W / 2 + 1)); nose.rotation.y = sz > 0 ? -Math.PI / 2 : Math.PI / 2; g.add(fixed(nose));
      }
      g.add(merged(Array.from({ length: 7 }, (_, i) => boxGeo(.18, deck - 1, .5, px - sx * 2.3, (deck - 1) / 2 - 1, -W / 2 + 1 + i * (W - 2) / 6)), std(0x2b2621, { roughness: 1 })));   // fender boards
      g.add(box(34, 1, W, deckMat, sx * (twin.hinge + 4.4 + 17), deck - .5, 0));                               // approach span
      g.add(merged([boxGeo(34, 1.1, .5, sx * (twin.hinge + 4.4 + 17), deck - 1.55, W / 2 - 2), boxGeo(34, 1.1, .5, sx * (twin.hinge + 4.4 + 17), deck - 1.55, -W / 2 + 2)], std(0x7d8798, { metalness: .4 })));
      for (const pxx of [22, 34]) g.add(fixed(box(2, deck, 2.4, concrete, sx * pxx, deck / 2 - 1, 0)));
      roadPlane(g, 34, W, sx * (twin.hinge + 4.4 + 17), deck);
      // stainless balustrade: posts, rails and glass
      const postsGeo = [], railGeo = [];
      for (const sz of [-1, 1]){
        for (let i = 0; i <= 22; i++) postsGeo.push(boxGeo(.07, 1.1, .07, sx * (twin.hinge + 4.4 + i * 34 / 22), deck + .55, sz * (W / 2 - .05)));
        railGeo.push(boxGeo(34, .08, .1, sx * (twin.hinge + 4.4 + 17), deck + 1.12, sz * (W / 2 - .05)));
        railGeo.push(boxGeo(34, .05, .05, sx * (twin.hinge + 4.4 + 17), deck + .35, sz * (W / 2 - .05)));
        const glass = box(34, .7, .02, new THREE.MeshPhysicalMaterial({ color: 0x9fd6ff, transparent: true, opacity: .18, roughness: .05, metalness: .1 }), sx * (twin.hinge + 4.4 + 17), deck + .72, sz * (W / 2 - .05));
        g.add(glass);
      }
      g.add(merged(postsGeo.concat(railGeo), steel));
      g.add(lamp(sx * 26, W / 2 - .4, deck, 0xdff3ff, true)); g.add(lamp(sx * 18, -W / 2 + .4, deck, 0xdff3ff, false));
      const wash = new THREE.PointLight(0xcfe0ff, 60, 30, 1.6); wash.position.set(sx * twin.hinge, deck + 4, W / 2 + 5); g.add(wash);
      // rounded light pylon with a warning light
      const pylon = new THREE.Mesh(new THREE.CapsuleGeometry(.45, 5, 6, 12), std(0x7a8494, { metalness: .5, roughness: .4 })); pylon.position.set(sx * 16, deck + 3, -W / 2 - .8); g.add(pylon);
      const warn = new THREE.Mesh(new THREE.SphereGeometry(.3, 10, 8), glow(0x3a1016, 1)); warn.position.set(sx * 16, deck + 5.9, -W / 2 - .8); g.add(warn); twin.warns.push(warn);
      quay(g, sx * 44, deck, W, 12);
      twin.barriers.push(barrier(g, sx * 18.5, deck, sx < 0 ? W / 2 - .4 : -W / 2 + .4, W / 2 - .8));
    }
    // balustrade lights in the sign colour
    twin.dmxMat = glow(STATE_COLORS.BOTH, 4);
    const dmx = new THREE.InstancedMesh(new THREE.SphereGeometry(.12, 8, 6), twin.dmxMat, 40); let n = 0; const m4 = new THREE.Matrix4();
    for (const sx of [-1, 1]) for (let i = 0; i < 10; i++) for (const sz of [-1, 1]){ m4.makeTranslation(sx * (twin.hinge + 5 + i * 3.3), deck + 1.2, sz * (W / 2 - .05)); dmx.setMatrixAt(n++, m4); }
    g.add(dmx);
    // leaves: triangles split diagonally across the road, hinged at each pier
    const leafShape = pts => { const s = new THREE.Shape(); s.moveTo(pts[0][0], pts[0][1]); pts.slice(1).forEach(p => s.lineTo(p[0], p[1])); s.closePath(); return s; };
    const leafMat = std(0xeef2f8, { roughness: .42, metalness: .35 }), ribMat = std(0xb8c2d0, { metalness: .5, roughness: .45 });
    twin.edgeMat = glow(STATE_COLORS.BOTH, 3);
    const makeLeaf = side => {
      const pivot = new THREE.Group(); pivot.position.set(side * twin.hinge, deck, 0); g.add(pivot);
      const dir = side < 0 ? 1 : -1;
      const pts = side < 0 ? [[0, -W / 2], [0, W / 2], [L, W / 2]] : [[0, -W / 2], [0, W / 2], [-L, -W / 2]];
      const geo = new THREE.ExtrudeGeometry(leafShape(pts), { depth: 1.1, bevelEnabled: false }); geo.rotateX(Math.PI / 2);
      pivot.add(new THREE.Mesh(geo, leafMat));
      // cross ribs underneath, each clipped to the triangle
      const ribs = [];
      for (let s = 1.2; s < L - .8; s += 1.4){
        const len = W * (1 - s / L), zc = side < 0 ? W / 2 - len / 2 : -W / 2 + len / 2;
        ribs.push(boxGeo(.28, .45, len, dir * s, -1.32, zc));
      }
      ribs.push(boxGeo(L, .5, .5, dir * L / 2, -1.35, side < 0 ? W / 2 - .3 : -W / 2 + .3));     // edge girder under the mast
      pivot.add(merged(ribs, ribMat));
      // the road surface on the leaf (the triangle, marked like the approaches)
      const rs = new THREE.Mesh(new THREE.ShapeGeometry(leafShape(pts.map(([x, z]) => [x, -z]))), new THREE.MeshStandardMaterial({ color: 0x2a2e36, roughness: .85 }));
      rs.rotation.x = -Math.PI / 2; rs.position.y = .006; pivot.add(rs);
      // mast along the long edge, past the tip, with a glowing white tip
      const zEdge = side < 0 ? W / 2 - .1 : -W / 2 + .1, len = L * 1.12;
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(.13, .22, len, 12), std(0xf2f5fa, { metalness: .6, roughness: .3 })); mast.rotation.z = Math.PI / 2; mast.position.set(dir * len / 2, .38, zEdge); pivot.add(mast);
      const tip = new THREE.Mesh(new THREE.SphereGeometry(.34, 12, 8), glow(0xffffff, 10)); tip.position.set(dir * len, .38, zEdge); pivot.add(tip);
      // balustrade posts along the leaf's long edge
      pivot.add(merged(Array.from({ length: 12 }, (_, i) => boxGeo(.07, 1.1, .07, dir * (.8 + i * (L - 1.6) / 11), .55, zEdge)).concat([boxGeo(L, .08, .1, dir * L / 2, 1.12, zEdge)]), steel));
      // neon strip along the diagonal edge
      const a = V(0, .05, side < 0 ? -W / 2 : W / 2), b = V(dir * L, .05, side < 0 ? W / 2 : -W / 2);
      pivot.add(new THREE.Mesh(beamGeo(a, b, .08), twin.edgeMat));
      return pivot;
    };
    twin.leafL = makeLeaf(-1); twin.leafR = makeLeaf(1);
    // hydraulic rams: barrel anchored in the pier, piston attached under the leaf
    const barrelMat = std(0x3c4454, { metalness: .8, roughness: .3 }), pistonMat = std(0xe6ebf2, { metalness: 1, roughness: .12 });
    for (const [leaf, side] of [[twin.leafL, -1], [twin.leafR, 1]]) for (const z of [-3, 3]){
      const barrel = new THREE.Mesh(new THREE.CylinderGeometry(.3, .3, 1, 14), barrelMat), piston = new THREE.Mesh(new THREE.CylinderGeometry(.14, .14, 1, 12), pistonMat);
      g.add(barrel, piston);
      twin.rams.push({ leaf, side, anchor: V(side * (twin.hinge + 1.4), deck - 4.6, z), attach: V(-side * 3.2, -1.4, z), barrel, piston });
    }
  }
  twin.group.position.set(-35, 0, -6); twin.group.rotation.y = .62;
  scene.add(twin.group);

  // ------------------------------------------------------------ Poole Bridge
  const poole = { key: "POOLE", group: new THREE.Group(), p: 0, from: 0, target: 0, t0: -99, maxDeg: 74, hinge: 12.3, L: 12.3, W: 10, deck: 4,
                  lanes: [2.2, -2.2], stops: [-24, 24], barrierP: 0, barriers: [], rams: [], warns: [] };
  {
    const g = poole.group, { L, W, deck } = poole;
    const facade = towerFacade(), green = std(0x5fae9b, { roughness: .8 });
    const towerMats = [green, green, std(0x5aa593), green, std(0xffffff, { map: facade, roughness: .8 }), std(0xffffff, { map: facade, roughness: .8 })];
    const teal = std(0x2fb3a3, { metalness: .55, roughness: .42, emissive: 0x0a2e2a });
    const signMat = new THREE.MeshStandardMaterial({ map: warnSignTex(), transparent: true, alphaTest: .3, side: THREE.DoubleSide, emissive: 0x331010, roughness: .6 });
    for (const sx of [-1, 1]){
      const cx = sx * (poole.hinge + 4.1);
      for (const sz of [-1, 1]){                                   // a tower each side of the road
        const tz = sz * (W / 2 + 1.8);
        const t = new THREE.Mesh(new THREE.BoxGeometry(8.2, 14, 3.6), towerMats); t.position.set(cx, 6, tz); g.add(fixed(t));
        // vertical ribs, cornice band, parapet and rooftop plant
        const det = [];
        for (const rx of [-3.6, -1.2, 1.2, 3.6]) det.push(boxGeo(.28, 13.2, .16, cx + rx, 5.8, tz + sz * 1.86));
        det.push(boxGeo(8.6, .5, 4, cx, 11.9, tz), boxGeo(8.8, .6, 4.2, cx, 13.3, tz), boxGeo(8.4, .8, .25, cx, 14, tz + 1.95), boxGeo(8.4, .8, .25, cx, 14, tz - 1.95));
        g.add(merged(det, std(0x4f9a87, { roughness: .7 })));
        g.add(box(3.2, .8, 2, std(0x4d9b89), cx, 14.1, tz));
        for (const wx of [-2.4, 1.6]) g.add(box(.6, 1, .05, glow(0xffd98a, 3.2), cx + wx, 10.8, tz + sz * 1.9));
        // signal head with twin red lamps (alternate flashing during a lift)
        g.add(box(1.4, .8, .5, std(0x141a26), cx - sx * 3.2, 15.3, tz));
        g.add(box(.12, 1.2, .12, std(0x141a26), cx - sx * 3.2, 14.6, tz));
        for (const lx of [-.35, .35]){
          const w1 = new THREE.Mesh(new THREE.SphereGeometry(.22, 10, 8), glow(0x3a1016, 1)); w1.position.set(cx - sx * 3.2 + lx, 15.3, tz + sz * .27); g.add(w1); poole.warns.push(w1);
        }
        const up = new THREE.SpotLight(0xfff3c4, 90, 22, .55, .6, 1.6); up.position.set(cx, .5, tz + sz * 6); up.target.position.set(cx, 10, tz); g.add(up, up.target);
      }
      // 3D lattice portal over the road, with warning signs
      const pg = [], zs = W / 2 + .2, n = 8;
      pg.push(beamGeo(V(cx, 11.1, -zs), V(cx, 11.1, zs), .25), beamGeo(V(cx, 12.9, -zs), V(cx, 12.9, zs), .25));
      for (let i = 0; i <= n; i++){
        const z = -zs + i * 2 * zs / n; pg.push(beamGeo(V(cx, 11.1, z), V(cx, 12.9, z), .14));
        if (i < n) pg.push(beamGeo(V(cx, i % 2 ? 12.9 : 11.1, z), V(cx, i % 2 ? 11.1 : 12.9, z + 2 * zs / n), .12));
      }
      g.add(merged(pg, teal));
      for (const sz of [-1, 1]){ const sgn = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.5), signMat); sgn.position.set(cx - sx * .2, 12, sz * 1.8); sgn.rotation.y = sx < 0 ? -Math.PI / 2 : Math.PI / 2; g.add(sgn); }
      g.add(box(44, .9, W, std(0x3b4150, { roughness: .9 }), sx * (poole.hinge + 8.2 + 22), deck - .45, 0));
      roadPlane(g, 44, W, sx * (poole.hinge + 8.2 + 22), deck);
      roadPlane(g, 8.2, W, cx, deck);
      g.add(box(8.2, .9, W, std(0x3b4150, { roughness: .9 }), cx, deck - .45, 0));
      g.add(lamp(sx * 32, W / 2 - .4, deck, 0xffd98a, true)); g.add(lamp(sx * 44, -W / 2 + .4, deck, 0xffd98a, false));
      const piles = new THREE.InstancedMesh(new THREE.CylinderGeometry(.28, .3, deck + 2, 10), std(0x2b2621, { roughness: 1 }), 10); const m4 = new THREE.Matrix4();
      for (let i = 0; i < 10; i++){ m4.makeTranslation(sx * (poole.hinge + .6 + (i % 5) * 1.9), (deck + 2) / 2 - 1.5, (i < 5 ? 1 : -1) * (W / 2 + 4.6)); piles.setMatrixAt(i, m4); }
      g.add(fixed(piles));
      quay(g, sx * 58, deck, W, 14);
      for (let i = 0; i < 3; i++) g.add(fixed(makeBuilding(12, 7 + i * 3, 9, sx * (52 + i * 12), -28 - i * 7, 0, true)));   // waterfront sheds
      poole.barriers.push(barrier(g, sx * 25.5, deck, sx < 0 ? W / 2 - .3 : -W / 2 + .3, W / 2 - .6));
    }
    const chevron = new THREE.MeshStandardMaterial({ map: chevronTex(), roughness: .6 });
    const makeLeaf = side => {
      const pivot = new THREE.Group(); pivot.position.set(side * poole.hinge, deck, 0); g.add(pivot);
      const dir = side < 0 ? 1 : -1;
      pivot.add(box(L, .9, W, std(0x2c55b0, { roughness: .5, metalness: .4 }), dir * L / 2, -.45, 0));
      // underside: longitudinal girders and cross girders
      const under = [boxGeo(L, 1, .5, dir * L / 2, -1.4, W / 2 - 1.2), boxGeo(L, 1, .5, dir * L / 2, -1.4, -W / 2 + 1.2)];
      for (let s = .8; s < L; s += 1.5) under.push(boxGeo(.3, .7, W - 1.2, dir * s, -1.25, 0));
      pivot.add(merged(under, std(0x17307a, { roughness: .6, metalness: .3 })));
      // 3D Pratt-style lattice girders along both edges
      const tg = [], bays = 8, h = 1.9;
      for (const sz of [-1, 1]){
        const z = sz * (W / 2 - .12);
        tg.push(beamGeo(V(0, .1, z), V(dir * L, .1, z), .22), beamGeo(V(dir * .4, h, z), V(dir * (L - .4), h, z), .22));
        for (let i = 0; i <= bays; i++){
          const x = dir * (.4 + i * (L - .8) / bays);
          tg.push(beamGeo(V(x, .1, z), V(x, h, z), .14));
          if (i < bays){ const x2 = dir * (.4 + (i + 1) * (L - .8) / bays); tg.push(i % 2 ? beamGeo(V(x, h, z), V(x2, .1, z), .12) : beamGeo(V(x, .1, z), V(x2, h, z), .12)); }
        }
        tg.push(boxGeo(L, .07, .07, dir * L / 2, h + .4, z));          // handrail on top
      }
      pivot.add(merged(tg, teal));
      roadPlane(pivot, L, W - .5, dir * L / 2, 0).position.x = dir * L / 2;
      const tip = new THREE.Mesh(new THREE.PlaneGeometry(W, 1.6), chevron); tip.position.set(dir * (L + .01), -.8, 0); tip.rotation.y = dir > 0 ? Math.PI / 2 : -Math.PI / 2; pivot.add(tip);
      return pivot;
    };
    poole.leafL = makeLeaf(-1); poole.leafR = makeLeaf(1);
  }
  poole.group.position.set(37, 0, 0); poole.group.rotation.y = -.42;
  scene.add(poole.group);
  const bridges = [twin, poole];
  // Solid obstacles (full height) and deck undersides (only boats taller than "under" hit them),
  // as local x/z boxes. The leaves are checked separately, by height, as they move.
  twin.solids = [[8.8, 13.2, -(twin.W / 2 + 3.2), twin.W / 2 + 3.2], [21, 23, -1.2, 1.2], [33, 35, -1.2, 1.2], [38, 50, -18, 18]];
  twin.decks = [[13.2, 47.2, -twin.W / 2, twin.W / 2, twin.deck - 1]];
  twin.leafThick = 1.6;
  poole.solids = [[12.3, 20.5, poole.W / 2, poole.W / 2 + 3.6], [12.3, 20.5, -(poole.W / 2 + 3.6), -poole.W / 2], [51, 65, -17, 17]];
  poole.decks = [[12.3, 64.5, -poole.W / 2, poole.W / 2, poole.deck - 1]];
  poole.leafThick = 2.0;
  {
    // everything standing in the water, as world-space boxes for the shader's foam line
    twin.group.updateMatrixWorld(); poole.group.updateMatrixWorld();
    let n = 0;
    const put = (cx, cz, hx, hz, ax, az) => { wu.obA.value[n].set(cx, cz, hx, hz); wu.obB.value[n].set(ax, az); n++; };
    const piles = [[poole.hinge + .3, poole.hinge + 8.2, poole.W / 2 + 4.3, poole.W / 2 + 4.9], [poole.hinge + .3, poole.hinge + 8.2, -(poole.W / 2 + 4.9), -(poole.W / 2 + 4.3)]];
    for (const [b, boxes] of [[twin, twin.solids], [poole, poole.solids.concat(piles)]]){
      const ax = Math.cos(b.group.rotation.y), az = -Math.sin(b.group.rotation.y);
      for (const [x0, x1, z0, z1] of boxes) for (const sx of [-1, 1]){
        const c = b.group.localToWorld(V(sx * (x0 + x1) / 2, 0, (z0 + z1) / 2));
        put(c.x, c.z, (x1 - x0) / 2, (z1 - z0) / 2, ax, az);
      }
    }
    put(0, -180, 450, 30, 1, 0);                                             // far shore
    wu.obN.value = n;
  }

  // moored yachts along the quays
  twin.group.updateMatrixWorld(); poole.group.updateMatrixWorld();
  const moored = [];
  for (const [b, x, z] of [[twin, -46, 21], [twin, -41, 25], [poole, 55, 20], [poole, 61, 24]]){
    const m = makeBoat("yacht"); const p = V(x, 0, z); b.group.localToWorld(p); m.position.copy(p); m.rotation.y = b.group.rotation.y + .1; m.scale.setScalar(.8);
    m.userData.phase = rand() * 6; scene.add(m); moored.push(m);
  }

  // ------------------------------------------------------------ traffic (queues at the barriers during a lift)
  for (const b of bridges){
    b.cars = [];
    const colors = [0x20283a, 0x39425a, 0x6b7a90, 0x8a1f2a, 0x2e3a55, 0xd9dee8];
    for (let i = 0; i < 6; i++){
      const lane = i % 2, car = makeCar(colors[i]);
      car.userData = { lane, dir: lane === 0 ? 1 : -1, x: -60 + i * 21, v: 11 + rand() * 3 };
      car.position.set(car.userData.x, b.deck, b.lanes[lane]); car.rotation.y = lane === 0 ? 0 : Math.PI;
      b.group.add(car); b.cars.push(car);
    }
  }
  // A car is "on the span" when any part of it is between the hinges (plus a metre).
  const HALF_CAR = 2.2;
  const onSpan = (b, x) => Math.abs(x) < b.hinge + 1 + HALF_CAR;
  function updateTraffic(b, dt){
    if (b.broken) return;                                     // wrecked: whatever is left stays put
    const closed = b.target === 1 || b.p > .002;
    for (const lane of [0, 1]){
      const cars = b.cars.filter(c => c.userData.lane === lane).sort((a, c) => (c.userData.x - a.userData.x) * (lane === 0 ? 1 : -1));
      cars.forEach((car, i) => {
        const u = car.userData, dir = u.dir;
        let limit = Infinity, speed = u.v;
        const stop = lane === 0 ? b.stops[0] : b.stops[1];
        if (closed){
          const spanEdge = -dir * (b.hinge + 1 + HALF_CAR);          // centre position that just keeps the car off the span
          if (dir * (u.x - stop) < 0) limit = Math.min(limit, dir * (stop - u.x));                    // before the barrier: wait there
          else if (dir * (u.x - spanEdge) <= 0) limit = Math.min(limit, dir * (spanEdge - u.x));      // past it: wait at the span edge
          else if (onSpan(b, u.x)) speed = u.v * 2.2;                                                  // already on it: clear it quickly
        }
        const ahead = cars[i - 1];
        if (ahead && dir * (ahead.userData.x - u.x) > 0) limit = Math.min(limit, dir * (ahead.userData.x - u.x) - 6.5);
        const step = Math.max(0, Math.min(speed * dt, limit));
        u.x += dir * step;
        if (u.x > 62) u.x = -62; if (u.x < -62) u.x = 62;
        car.position.x = u.x;
      });
    }
  }

  // ------------------------------------------------------------ boats
  const boats = [];
  function addBoat(kind, b, path){
    const m = makeBoat(kind);
    m.userData = { b, ...path, t: path.always ? (path.phase || 0) : 0, wait: path.always ? 0 : (path.phase || 0) * 12 };
    scene.add(m); boats.push(m); return m;
  }
  addBoat("rib", null, { from: V(-130, 0, 44), to: V(130, 0, 47), speed: 14, always: true });
  addBoat("launch", null, { from: V(150, 0, 44), to: V(-150, 0, 52), speed: 6, always: true, phase: .4 });
  const channel = (b, dz, lane, kind, speed, phase) => {
    const a = V(lane, 0, 80 * dz), c = V(lane, 0, -80 * dz);
    b.group.localToWorld(a); b.group.localToWorld(c);
    return addBoat(kind, b, { from: a, to: c, speed, phase });
  };
  twin.group.updateMatrixWorld(); poole.group.updateMatrixWorld();
  channel(twin, 1, -2, "yacht", 6, 0); channel(twin, -1, 2.5, "superyacht", 4.5, .5);
  channel(poole, 1, -1.8, "trawler", 5, 0); channel(poole, -1, 2, "tug", 4.2, .45); channel(poole, 1, .5, "yacht", 5.5, .7);
  // Boats float on the swell: heave, pitch and roll come from the wave height at the bow, stern
  // and both beams, low-passed for the hull's inertia.
  const fwdR = V(0, 0, 0);
  function ride(m, dt, base){
    fwdR.set(1, 0, 0).applyQuaternion(m.quaternion); fwdR.y = 0; fwdR.normalize();
    const u = m.userData, sc = m.scale.x, hl = m.dims.len / 2 * sc * .8, hb = m.dims.beam / 2 * sc, x = m.position.x, z = m.position.z;
    const hB = waveHeight(x + fwdR.x * hl, z + fwdR.z * hl, waterTime), hS = waveHeight(x - fwdR.x * hl, z - fwdR.z * hl, waterTime);
    const hP = waveHeight(x - fwdR.z * hb, z + fwdR.x * hb, waterTime), hQ = waveHeight(x + fwdR.z * hb, z - fwdR.x * hb, waterTime);
    const k = 1 - Math.exp(-dt * 4);
    u.heave = (u.heave ?? 0) + ((hB + hS + hP + hQ) / 4 - (u.heave ?? 0)) * k;
    u.pitch = (u.pitch ?? 0) + (Math.atan2(hB - hS, 2 * hl) - (u.pitch ?? 0)) * k;
    u.roll = (u.roll ?? 0) + (-Math.atan2(hP - hQ, 2 * hb) - (u.roll ?? 0)) * k;
    m.position.y = base + u.heave;
    m.rotateZ(u.pitch); m.rotateX(u.roll);
  }
  function feedWater(){
    let n = 0;
    for (const m of boats.concat(moored)){
      if (!m.visible || m.userData.dead || n >= MAX_BOATS) continue;
      const u = m.userData, sc = m.scale.x, moving = u.speed && (u.always || u.t > 0);
      fwdR.set(1, 0, 0).applyQuaternion(m.quaternion); fwdR.y = 0; fwdR.normalize();
      wu.boatA.value[n].set(m.position.x, m.position.z, fwdR.x, fwdR.z);
      wu.boatB.value[n].set(m.dims.len / 2 * sc, m.dims.beam / 2 * sc, moving ? u.speed : 0, 0);
      n++;
    }
    wu.boatN.value = n;
  }
  function updateBoats(dt, t){
    for (const m of boats){
      const u = m.userData, len = u.from.distanceTo(u.to);
      if (u.dead){
        m.visible = false;
        if (t < u.respawnAt) continue;
        u.dead = false; u.t = 0; if (!u.always) u.wait = 3;
      }
      if (u.always){ u.t = (u.t + u.speed * dt / len) % 1; }
      else {
        const open = u.b.p > .85 || u.b.broken, crossing = u.t > 0;
        if (!crossing && open){ u.wait -= dt; if (u.wait <= 0) u.t = 1e-4; }
        if (u.t > 0){ u.t += u.speed * dt / len; if (u.t >= 1){ u.t = 0; u.wait = 6 + rand() * 6; } }
      }
      m.visible = u.always || u.t > 0;
      m.position.lerpVectors(u.from, u.to, u.t);
      m.lookAt(u.to.x, m.position.y, u.to.z); m.rotateY(-Math.PI / 2);
      ride(m, dt, -.2);
    }
    for (const m of moored){
      if (m.userData.dead){ m.visible = t >= m.userData.respawnAt; if (m.visible) m.userData.dead = false; else continue; }
      if (m.userData.yaw === undefined) m.userData.yaw = m.rotation.y;
      m.rotation.set(0, m.userData.yaw, 0); ride(m, dt, -.18);
    }
    checkCollisions(t);
    feedWater();
  }

  // ------------------------------------------------------------ collisions and explosions
  // Each hull is approximated by three circles along its length (in the water plane). Any
  // overlap between two boats blows both up; they respawn later at the start of their route.
  const fwd = V(0, 0, 0);
  function hullCircles(m){
    fwd.set(1, 0, 0).applyQuaternion(m.quaternion); fwd.y = 0; fwd.normalize();
    const sc = m.scale.x, r = m.dims.beam / 2 * sc * .95, off = (m.dims.len / 2 - m.dims.beam / 2) * sc;
    return [-1, 0, 1].map(k => ({ x: m.position.x + fwd.x * off * k, z: m.position.z + fwd.z * off * k, r }));
  }
  const local = V(0, 0, 0);
  function hitsBridge(m, circles){
    for (const b of bridges){
      const th = THREE.MathUtils.degToRad(Math.min(b.maxDeg * b.p, 89.5)), tan = Math.tan(th), reach = b.L * Math.cos(th);
      for (const c of circles){
        local.set(c.x, 0, c.z); b.group.worldToLocal(local);
        const x = local.x, z = local.z, r = c.r;
        const inBox = (x0, x1, z0, z1) => (Math.abs(x) + r > x0 && Math.abs(x) - r < x1 && z + r > z0 && z - r < z1);
        if (b.solids.some(([x0, x1, z0, z1]) => inBox(x0, x1, z0, z1))) return true;
        if (b.decks.some(([x0, x1, z0, z1, under]) => m.dims.h > under && inBox(x0, x1, z0, z1))) return true;
        if (!b.broken && Math.abs(z) < b.W / 2 + r){                                   // under the lifting span: check each leaf's height here
          for (const side of [-1, 1]){
            const hx = side * b.hinge, dist = -side * (x - hx);           // distance from this leaf's hinge towards the channel
            if (dist > -r && dist < reach + r){
              const underside = b.deck - b.leafThick + Math.max(0, dist) * tan;
              if (m.dims.h > underside) return true;
            }
          }
        }
      }
    }
    return false;
  }
  let explosionCount = 0;
  function checkCollisions(t){
    const live = boats.filter(m => m.visible && !m.userData.dead).concat(moored.filter(m => !m.userData.dead));
    const circles = live.map(hullCircles);
    for (let i = 0; i < live.length; i++){
      if (hitsBridge(live[i], circles[i])){
        const m = live[i]; m.userData.dead = true; m.userData.respawnAt = t + 12 + rand() * 8; m.visible = false;
        explode(V(m.position.x, .6, m.position.z), m.dims.len);
        return;
      }
    }
    for (let i = 0; i < live.length; i++) for (let j = i + 1; j < live.length; j++){
      for (const a of circles[i]) for (const b of circles[j]){
        const dx = a.x - b.x, dz = a.z - b.z;
        if (dx * dx + dz * dz < (a.r + b.r) * (a.r + b.r)){
          const hit = V((a.x + b.x) / 2, .6, (a.z + b.z) / 2);
          for (const m of [live[i], live[j]]){ m.userData.dead = true; m.userData.respawnAt = t + 12 + rand() * 8; m.visible = false; }
          explode(hit, Math.max(live[i].dims.len, live[j].dims.len));
          return;
        }
      }
    }
  }
  const smokeTex = canvasTex(64, 64, (g, w) => {
    const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    r.addColorStop(0, "rgba(90,90,100,.9)"); r.addColorStop(.5, "rgba(60,60,70,.5)"); r.addColorStop(1, "rgba(40,40,50,0)");
    g.fillStyle = r; g.fillRect(0, 0, w, w);
  });
  const fireTex = canvasTex(128, 128, (g, w) => {
    const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    r.addColorStop(0, "rgba(255,244,214,1)"); r.addColorStop(.25, "rgba(255,190,90,.95)"); r.addColorStop(.55, "rgba(230,90,30,.55)");
    r.addColorStop(.8, "rgba(120,30,10,.18)"); r.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = r; g.fillRect(0, 0, w, w);
  });
  const fxBox = new THREE.BoxGeometry(1, 1, 1), fxRing = new THREE.RingGeometry(.96, 1.08, 64);
  const debrisMat = std(0x2a2c33, { roughness: .8 }), emberMat = glow(0xff7a1a, 6);
  // flash lights exist from the start (intensity 0) so an explosion never forces shaders to recompile
  const flashLights = [0, 1].map(() => { const l = new THREE.PointLight(0xff8a2a, 0, 90, 1.6); scene.add(l); return l; });
  const meteorLights = [0, 1].map(() => { const l = new THREE.PointLight(0xff7a30, 0, 120, 1.4); scene.add(l); return l; });
  const fireLights = [0, 1, 2].map(() => { const l = new THREE.PointLight(0xff6a20, 0, 45, 1.6); scene.add(l); return l; });
  let nextFlash = 0, shake = 0;
  const explosions = [];
  // Explosion: flash, billowing fireball with secondary blasts, streaking sparks (or a water plume),
  // a low surge of smoke rolling outwards, tumbling debris and a tall smoke column.
  const sparkMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  function explode(pos, size, big = false){
    const k = clamp(size / 10, .7, big ? 2.4 : 1.8), g = new THREE.Group(); g.position.copy(pos); scene.add(g);
    const onWater = pos.y < 2.5;
    const puff = (opacity) => { const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: fireTex, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })); g.add(sp); return sp; };
    const flash = puff(1); flash.scale.setScalar(3 * k);
    const fire = [], subs = [];
    const addFire = (off, n, delay, sizeMul, upMul) => {
      for (let i = 0; i < n; i++){
        const sp = puff(0), a = rand() * Math.PI * 2;
        sp.position.set(off.x + Math.cos(a) * rand() * 1.2 * k, off.y + rand() * 1.2 * k, off.z + Math.sin(a) * rand() * 1.2 * k);
        fire.push({ s: sp, v: V(Math.cos(a) * (2 + rand() * 4) * k, (2 + rand() * 6) * k * upMul, Math.sin(a) * (2 + rand() * 4) * k),
                    delay: delay + rand() * .15, life: .9 + rand() * .7, size: (4 + rand() * 4.5) * k * sizeMul, spin: (rand() - .5) * 2.5 });
      }
    };
    addFire(V(0, 0, 0), 22, 0, 1, 1);
    for (let j = 0; j < (big ? 3 : 2); j++){                          // secondary blasts, a moment later and a little way off
      const d = .12 + rand() * .4; subs.push(d);
      addFire(V((rand() - .5) * 5 * k, rand() * 2 * k, (rand() - .5) * 5 * k), 7, d, .7, .8);
    }
    if (big) addFire(V(0, 1, 0), 10, .05, .9, 2.6);                  // a rising column of fire
    const ring = new THREE.Mesh(fxRing, new THREE.MeshBasicMaterial({ color: 0xffe2b8, transparent: true, opacity: .6, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = -pos.y + .55; g.add(ring);
    addRipple(pos.x, pos.z, (big ? 2 : 1.6) * k);
    // sparks: short additive streaks (head hot, tail dim); on water, part of them is a spray plume
    const NS = big ? 140 : 90, sGeo = new THREE.BufferGeometry(), sparks = [];
    sGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
    sGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(NS * 6), 3));
    for (let i = 0; i < NS; i++){
      const a = rand() * Math.PI * 2, el = .15 + rand() * 1.2, sp = (18 + rand() * 30) * k, water = onWater && i < NS * .45;
      sparks.push({ p: V(0, 0, 0), water, life: water ? 1.2 + rand() * .8 : .6 + rand() * .8,
                    v: water ? V(Math.cos(a) * sp * .12, sp * (.7 + rand() * .7), Math.sin(a) * sp * .12) : V(Math.cos(a) * Math.cos(el) * sp, Math.sin(el) * sp, Math.sin(a) * Math.cos(el) * sp) });
    }
    const sLines = new THREE.LineSegments(sGeo, sparkMat); sLines.frustumCulled = false; g.add(sLines);
    const debris = [];
    for (let i = 0; i < (big ? 30 : 22); i++){
      const ember = i % 3 === 0, d = new THREE.Mesh(fxBox, ember ? emberMat : debrisMat);
      const sz = ember ? .1 + rand() * .15 : (.2 + rand() * .5) * Math.min(k, 1.4);      // embers stay small, or bloom turns them into glowing squares
      d.scale.set(sz, sz * (.4 + rand() * .6), sz * (.5 + rand())); g.add(d);
      const a = rand() * Math.PI * 2, out = (4 + rand() * 9) * k;
      debris.push({ m: d, v: V(Math.cos(a) * out, (8 + rand() * 14) * k, Math.sin(a) * out), w: V(rand() * 8, rand() * 8, rand() * 8) });
    }
    const smoke = [];
    const addSmoke = (p, v, delay, grow, drag, op, life) => {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, transparent: true, opacity: 0, depthWrite: false }));
      sp.position.copy(p); sp.scale.setScalar(2 * k); g.add(sp);
      smoke.push({ s: sp, v, delay, grow, drag, op, life });
    };
    for (let i = 0; i < (big ? 22 : 16); i++)                          // the main cloud, rising
      addSmoke(V((rand() - .5) * 2 * k, rand() * k, (rand() - .5) * 2 * k), V((rand() - .5) * 2, (3 + rand() * 3) * (big ? 1.6 : 1), (rand() - .5) * 2), rand() * .35, 3.2, .35, .75, big ? 6.5 : 4.2);
    for (let i = 0; i < 12; i++){                                       // base surge: a low ring rolling outwards
      const a = i / 12 * Math.PI * 2 + rand() * .3, sp = (12 + rand() * 8) * k;
      addSmoke(V(Math.cos(a) * k, .4 - pos.y, Math.sin(a) * k), V(Math.cos(a) * sp, .6, Math.sin(a) * sp), .05 + rand() * .1, 2.6, 1.6, onWater ? .45 : .6, 3.4);
    }
    if (onWater) smoke.slice(-12).forEach(p => p.s.material.color.setRGB(.85, .9, .95));   // spray, not dust
    const light = flashLights[nextFlash++ % flashLights.length]; light.position.copy(pos).setY(3);
    shake = Math.max(shake, big ? 1.3 : .7 * k); explosionCount++;
    explosions.push({ g, t: 0, k, big, fire, flash, ring, debris, smoke, light, subs, sparks, sGeo });
  }
  function updateExplosions(dt){
    for (let i = explosions.length - 1; i >= 0; i--){
      const e = explosions[i]; e.t += dt; const t = e.t, k = e.k;
      e.flash.scale.setScalar((3 + t * 40) * k); e.flash.material.opacity = clamp(1 - t / .22, 0, 1);
      for (const f of e.fire){
        const ft = t - f.delay; if (ft < 0) continue;
        const u = ft / f.life;
        f.s.position.addScaledVector(f.v, dt); f.v.multiplyScalar(1 - dt * 2.2); f.v.y += dt * 1.5;
        f.s.scale.setScalar(f.size * (.35 + Math.min(1, u * 2.2)));
        f.s.material.rotation += f.spin * dt;
        f.s.material.opacity = u < 1 ? clamp(Math.min(ft * 12, 1) * (1 - u) * (1 - u * .4), 0, 1) : 0;
        f.s.material.color.setHSL(.08 - u * .05, 1, clamp(.75 - u * .45, .2, .75));
      }
      e.ring.scale.setScalar((1 + t * 16) * k); e.ring.material.opacity = clamp(.6 - t * .45, 0, .6);
      let li = 2600 * k * Math.exp(-t * 3.2);
      for (const d of e.subs) if (t > d) li += 1800 * k * Math.exp(-(t - d) * 7);
      e.light.intensity = li * (.85 + Math.random() * .3);
      const pa = e.sGeo.attributes.position.array, ca = e.sGeo.attributes.color.array;
      for (let j = 0; j < e.sparks.length; j++){
        const sp = e.sparks[j], o = j * 6;
        let a = clamp(1 - t / sp.life, 0, 1);
        if (a > 0){
          sp.v.y -= (sp.water ? 14 : 20) * dt; sp.v.multiplyScalar(1 - (sp.water ? .5 : 1.4) * dt); sp.p.addScaledVector(sp.v, dt);
          if (sp.p.y + e.g.position.y < 0 && sp.v.y < 0){ sp.life = 0; a = 0; }
        }
        const tl = sp.water ? .1 : .05;
        pa[o] = sp.p.x; pa[o + 1] = sp.p.y; pa[o + 2] = sp.p.z;
        pa[o + 3] = sp.p.x - sp.v.x * tl; pa[o + 4] = sp.p.y - sp.v.y * tl; pa[o + 5] = sp.p.z - sp.v.z * tl;
        if (sp.water){ ca[o] = .9 * a; ca[o + 1] = 1 * a; ca[o + 2] = 1.1 * a; ca[o + 3] = ca[o + 4] = ca[o + 5] = .25 * a; }
        else { ca[o] = 5 * a; ca[o + 1] = 2.8 * a; ca[o + 2] = 1.1 * a; ca[o + 3] = 1.2 * a; ca[o + 4] = .25 * a; ca[o + 5] = .04 * a; }
      }
      e.sGeo.attributes.position.needsUpdate = true; e.sGeo.attributes.color.needsUpdate = true;
      for (const d of e.debris){
        if (d.m.position.y + e.g.position.y > 0 || d.v.y > 0){
          d.v.y -= 16 * dt; d.m.position.addScaledVector(d.v, dt);
          d.m.rotation.x += d.w.x * dt; d.m.rotation.y += d.w.y * dt; d.m.rotation.z += d.w.z * dt;
        } else { d.m.position.y -= .6 * dt; d.m.scale.multiplyScalar(1 - dt * .6); }
      }
      for (const p of e.smoke){
        const st = t - p.delay; if (st < 0) continue;
        p.s.position.addScaledVector(p.v, dt); p.v.multiplyScalar(1 - dt * p.drag);
        p.s.scale.setScalar((2 + st * p.grow) * k); p.s.material.opacity = clamp(Math.min(st * 3, 1) * (1 - st / p.life), 0, 1) * p.op;
      }
      if (t > (e.big ? 7 : 4.5)){
        scene.remove(e.g); e.light.intensity = 0; e.sGeo.dispose();
        [e.flash, e.ring].forEach(m => m.material.dispose()); e.fire.forEach(f => f.s.material.dispose()); e.smoke.forEach(p => p.s.material.dispose());
        explosions.splice(i, 1);
      }
    }
  }

  // ------------------------------------------------------------ alarm: red sky, meteor shower, bridges wrecked, fires
  let alarm = false, alarmK = 0, clockA = 0, nextMeteor = 0, nextML = 0, look0 = null, look1 = null;
  const meteors3 = [], wrecks = [], burners = [], flamePool = [], smokePool = [];
  const tailTex = canvasTex(16, 128, (c2, w, h) => {
    const gr = c2.createLinearGradient(0, 0, 0, h);
    gr.addColorStop(0, "rgba(255,50,20,0)"); gr.addColorStop(.55, "rgba(255,120,40,.55)"); gr.addColorStop(1, "rgba(255,244,220,1)");
    c2.fillStyle = gr; c2.fillRect(0, 0, w, h);
  });
  const tailGeo = new THREE.ConeGeometry(1, 1, 16, 1, true);          // apex = the end of the tail
  const tailOuter = new THREE.MeshBasicMaterial({ map: tailTex, transparent: true, opacity: .6, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
  const tailCore = tailOuter.clone(); tailCore.opacity = 1;
  tailOuter.color.setRGB(.9, .3, .1); tailCore.color.setRGB(1.7, 1.25, .85);
  const headMat = new THREE.SpriteMaterial({ map: fireTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  headMat.color.setRGB(2.6, 2.1, 1.5);
  const headGlowMat = new THREE.SpriteMaterial({ map: fireTex, transparent: true, opacity: .8, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  headGlowMat.color.setRGB(1, .35, .1);
  const bx = new THREE.Box3(), c0 = V(0, 0, 0), sz = V(0, 0, 0), UP = V(0, 1, 0);

  function captureLook(){
    look0 = { fog: scene.fog.color.clone(), hs: hemi.color.clone(), hg: hemi.groundColor.clone(), ml: moonLight.color.clone(), moon: moon.material.color.clone(),
              halo: halo.material.color.clone(), amb: wu.ambient.value.clone(), glint: wu.moonColor.value.clone(), deep: wu.deepColor.value.clone(), bloom: bloom.strength };
    look1 = { fog: new THREE.Color(0x3a0a0c), hs: new THREE.Color(0xc0402c), hg: new THREE.Color(0x200505), ml: new THREE.Color(0xff7a52), moon: new THREE.Color(0xff3a1c),
              halo: new THREE.Color(0xff4a2a), amb: lin(.16, .03, .025), glint: lin(.12, .03, .015), deep: lin(.006, .0012, .001), bloom: 1.05 };
  }
  function applyLook(k){
    paintSky(k);
    scene.fog.color.copy(look0.fog).lerp(look1.fog, k);
    hemi.color.copy(look0.hs).lerp(look1.hs, k); hemi.groundColor.copy(look0.hg).lerp(look1.hg, k);
    moonLight.color.copy(look0.ml).lerp(look1.ml, k);
    moon.material.color.copy(look0.moon).lerp(look1.moon, k); halo.material.color.copy(look0.halo).lerp(look1.halo, k);
    wu.ambient.value.copy(look0.amb).lerp(look1.amb, k); wu.moonColor.value.copy(look0.glint).lerp(look1.glint, k); wu.deepColor.value.copy(look0.deep).lerp(look1.deep, k);
    bloom.strength = look0.bloom + (look1.bloom - look0.bloom) * k;
    starMat.opacity = .8 - .5 * k;
  }
  // Everything a meteor can knock off a bridge: each child of the bridge group (lamps, barriers,
  // cars, rails, portals...) and each part of the lifting leaves -- but not the fixed foundations.
  function collectPieces(b){
    b.pieces = [];
    const add = o => { if (!o.isLight && !o.userData.fixed) b.pieces.push(o); };
    for (const c of b.group.children){
      if (c === b.leafL || c === b.leafR) c.children.forEach(add); else add(c);
    }
  }
  function makePools(){
    for (let i = 0; i < 200; i++){
      const f = new THREE.Sprite(new THREE.SpriteMaterial({ map: fireTex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
      f.visible = false; f.userData = { v: V(0, 0, 0) }; scene.add(f); flamePool.push(f);
    }
    for (let i = 0; i < 180; i++){
      const m = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, transparent: true, opacity: 0, depthWrite: false }));
      m.visible = false; m.userData = { v: V(0, 0, 0), col: new THREE.Color() }; scene.add(m); smokePool.push(m);
    }
    flamePool.next = smokePool.next = 0;
  }
  function emit(pool, p, vx, vy, vz, life, s0, s1, extra){       // ring buffer: the oldest particle is reused
    const sp = pool[pool.next]; pool.next = (pool.next + 1) % pool.length;
    const u = sp.userData;
    sp.position.copy(p); u.v.set(vx, vy, vz); u.age = 0; u.life = life; u.s0 = s0; u.s1 = s1; u.spin = (rand() - .5) * 2;
    u.op = extra && extra.op !== undefined ? extra.op : .55; if (u.col) u.col.setRGB(...(extra && extra.col || [.22, .17, .17]));
    sp.scale.setScalar(s0); sp.visible = true;
  }
  function updateParticles(dt){
    for (const sp of flamePool){
      if (!sp.visible) continue;
      const u = sp.userData; u.age += dt; const x = u.age / u.life;
      if (x >= 1){ sp.visible = false; continue; }
      sp.position.addScaledVector(u.v, dt); u.v.multiplyScalar(1 - dt * 1.5); u.v.y += dt * 2;
      sp.scale.setScalar(u.s0 + (u.s1 - u.s0) * x); sp.material.rotation += u.spin * dt;
      sp.material.opacity = Math.min(1, u.age * 12) * (1 - x);
      sp.material.color.setHSL(.09 - x * .07, 1, clamp(.7 - x * .45, .15, .7));
    }
    for (const sp of smokePool){
      if (!sp.visible) continue;
      const u = sp.userData; u.age += dt; const x = u.age / u.life;
      if (x >= 1){ sp.visible = false; continue; }
      sp.position.addScaledVector(u.v, dt); u.v.multiplyScalar(1 - dt * .35);
      sp.scale.setScalar(u.s0 + (u.s1 - u.s0) * Math.sqrt(x)); sp.material.rotation += u.spin * dt * .3;
      sp.material.opacity = Math.min(1, u.age * 2.5) * (1 - x) * u.op; sp.material.color.copy(u.col);
    }
  }
  function pickTarget(){
    if (rand() < .75){
      const b = bridges[rand() < .5 ? 0 : 1], live = (b.pieces || []).filter(o => !o.userData.broken);
      if (live.length){ bx.setFromObject(live[Math.floor(rand() * live.length)]); if (!bx.isEmpty()) return { pos: bx.getCenter(V(0, 0, 0)), bridge: true }; }
    }
    return { pos: V((rand() - .5) * 200, 0, -40 + rand() * 80), bridge: false };
  }
  function spawnMeteor(){
    const tgt = pickTarget();
    const dir = V((rand() - .5) * .9, -1, (rand() - .2) * .6).normalize();   // mostly down, drifting across and towards us
    const dist = 170 + rand() * 60, pos = tgt.pos.clone().addScaledVector(dir, -dist);
    const grp = new THREE.Group(); grp.quaternion.setFromUnitVectors(UP, dir.clone().negate());
    const len = 34 + rand() * 18, outer = new THREE.Mesh(tailGeo, tailOuter), core = new THREE.Mesh(tailGeo, tailCore);
    outer.scale.set(2.6, len, 2.6); outer.position.y = len / 2; core.scale.set(.8, len * .75, .8); core.position.y = len * .375;
    const head = new THREE.Sprite(headMat), glowS = new THREE.Sprite(headGlowMat);
    head.scale.setScalar(4); glowS.scale.setScalar(12);
    grp.add(outer, core, head, glowS); grp.position.copy(pos); scene.add(grp);
    meteors3.push({ grp, pos, dir, speed: 95 + rand() * 30, dist, trav: 0, tgt, light: meteorLights[nextML++ % meteorLights.length], acc: 0, sacc: 0 });
  }
  function meteorImpact(m, now){
    const p = m.pos; scene.remove(m.grp); m.light.intensity = 0;
    explode(p.clone(), 20, true);
    breakAt(p, 9.5);
    if (m.tgt.bridge || rand() < .35) addBurner(p);
    for (const bt of boats.concat(moored)){                          // boats caught underneath
      if (!bt.visible || bt.userData.dead || bt.position.distanceTo(p) > 9) continue;
      bt.userData.dead = true; bt.userData.respawnAt = now + 14 + rand() * 8; bt.visible = false;
    }
  }
  function breakAt(p, R){
    for (const b of bridges){
      if (!b.pieces) continue;
      let hit = false;
      for (const o of b.pieces){
        if (o.userData.broken || (b.stunt && b.stunt.car === o)) continue;
        bx.setFromObject(o);
        if (bx.isEmpty() || bx.distanceToPoint(p) > R) continue;
        detach(b, o, p); hit = true;
      }
      if (hit){ b.broken = true; b.decks = []; cleanupStunt(b); }
      for (const o of b.group.children){
        if (!o.userData.fixed || o.userData.charred) continue;
        bx.setFromObject(o); if (bx.distanceToPoint(p) < R + 3) char(o);
      }
    }
  }
  function char(o){                                                   // scorch a foundation that the blast reached
    o.userData.charred = true;
    o.traverse(m => {
      if (!m.isMesh) return;
      const burn = x => { const n = x.clone(); if (n.color) n.color.multiplyScalar(.3); if (n.emissive) n.emissiveIntensity *= .35; return n; };
      m.material = Array.isArray(m.material) ? m.material.map(burn) : burn(m.material);
    });
  }
  function detach(b, o, p){
    o.userData.broken = true;
    const ci = b.cars.indexOf(o); if (ci >= 0) b.cars.splice(ci, 1);
    bx.getCenter(c0); bx.getSize(sz);
    const size = Math.max(sz.x, sz.y, sz.z), k = 1 / (1 + size / 10);
    const pivot = new THREE.Group(); pivot.position.copy(c0); scene.add(pivot); pivot.attach(o);   // spin about the piece's own centre
    o.traverse(c => { if (c.isLight) c.intensity = 0; });            // lamps go out (the light objects stay, so no shader rebuild)
    const out = c0.clone().sub(p); out.y = 0;
    if (out.lengthSq() < 1e-4) out.set(rand() - .5, 0, rand() - .5);
    out.normalize().multiplyScalar((9 + rand() * 14) * k).setY((8 + rand() * 12) * k);
    wrecks.push({ pivot, v: out, w: V(rand() - .5, rand() - .5, rand() - .5).multiplyScalar(10 * (k + .15)), size, wet: false, done: false, smoke: 0 });
  }
  function updateWrecks(dt){
    for (const w of wrecks){
      if (w.done) continue;
      const pv = w.pivot;
      if (!w.wet){
        w.v.y -= 22 * dt;
        w.smoke += dt;
        if (w.smoke > .07){ w.smoke = 0; emit(smokePool, pv.position, 0, 1, 0, 2.2, 1.5, 6, { op: .45 }); emit(flamePool, pv.position, 0, 0, 0, .35, 1.2 + w.size * .08, .5); }
        if (pv.position.y < 0){
          w.wet = true;
          addRipple(pv.position.x, pv.position.z, Math.min(1.6, .5 + w.size * .06));
          for (let i = 0; i < 10; i++) emit(smokePool, pv.position, (rand() - .5) * 6, 5 + rand() * 7, (rand() - .5) * 6, 1.3, 1, 5 + w.size * .2, { op: .6, col: [.85, .9, .95] });
        }
      } else {
        w.v.multiplyScalar(Math.exp(-dt * 3)); w.w.multiplyScalar(Math.exp(-dt * 1.2));
        w.v.y = Math.min(w.v.y, 0) - (1 + w.size * .05) * dt * 3;
      }
      pv.position.addScaledVector(w.v, dt);
      pv.rotation.x += w.w.x * dt; pv.rotation.y += w.w.y * dt; pv.rotation.z += w.w.z * dt;
      if (pv.position.y < -(w.size + 6)) w.done = true;
    }
  }
  function addBurner(p){
    if (burners.length >= 12) return;
    // burn on whatever is left standing nearby, or on the water as a burning slick
    let at = V(p.x, .25, p.z), best = 14;
    for (const b of bridges) for (const o of b.group.children){
      if (!o.userData.fixed) continue;
      bx.setFromObject(o); const d = bx.distanceToPoint(p);
      if (d < best){ best = d; at = V(clamp(p.x, bx.min.x, bx.max.x), bx.max.y, clamp(p.z, bx.min.z, bx.max.z)); }
    }
    burners.push({ pos: at, acc: 0, sacc: 0, light: fireLights[burners.length] || null, seed: rand() * 10, r: 1 + rand() * 1.5 });
  }
  const jit = V(0, 0, 0);
  function updateBurners(dt, now){
    for (const b of burners){
      b.acc += dt; b.sacc += dt;
      while (b.acc > .03){ b.acc -= .03; jit.set((rand() - .5) * 2 * b.r, 0, (rand() - .5) * 2 * b.r).add(b.pos); emit(flamePool, jit, (rand() - .5), 3 + rand() * 3, (rand() - .5), .6 + rand() * .4, 2.6 + rand(), .8); }
      while (b.sacc > .16){ b.sacc -= .16; jit.copy(b.pos).y += 2.5; emit(smokePool, jit, .8 + rand() * .6, 2.4 + rand(), (rand() - .5) * .6, 5, 2.5, 15, { op: .5 }); }
      if (b.light){ b.light.position.copy(b.pos).y += 2; b.light.intensity = 260 + 120 * Math.sin(now * 17 + b.seed) + 80 * Math.sin(now * 29 + b.seed * 2); }
    }
  }
  function updateAlarm(dt, now){
    if (alarmK < 1){ alarmK = Math.min(1, alarmK + dt / 3); applyLook(alarmK); }
    clockA += dt;
    if (clockA >= nextMeteor && meteors3.length < 5){ spawnMeteor(); nextMeteor = clockA + .8 + rand() * 1.2; }
    for (let i = meteors3.length - 1; i >= 0; i--){
      const m = meteors3[i], step = m.speed * dt;
      m.trav += step; m.pos.addScaledVector(m.dir, step);
      if (m.trav >= m.dist){ m.pos.copy(m.tgt.pos); meteors3.splice(i, 1); meteorImpact(m, now); continue; }
      m.grp.position.copy(m.pos);
      m.light.position.copy(m.pos); m.light.intensity = 350;
      m.acc += dt; m.sacc += dt;
      while (m.acc > .02){ m.acc -= .02; emit(flamePool, m.pos, (rand() - .5) * 3, (rand() - .5) * 3, (rand() - .5) * 3, .45, 3.5, 7); }
      while (m.sacc > .05){ m.sacc -= .05; emit(smokePool, m.pos, (rand() - .5), .6, (rand() - .5), 3.2, 2.5, 11, { op: .4 }); }
    }
    updateWrecks(dt); updateBurners(dt, now); updateParticles(dt);
  }

  // ------------------------------------------------------------ the last-minute car
  const G = 95;
  const STUNT = { TWIN: { lane: 4.3, rampSpeed: 19, approach: 10, arrive: 2.7 }, POOLE: { lane: 2.2, rampSpeed: 12, approach: 10, arrive: 2.5 } };
  function startStunt(b, now){
    if (b.stunt) cleanupStunt(b);
    const cfg = STUNT[b.key], car = makeCar(0xe0343f);
    car.position.set(-b.hinge + 2.1 - cfg.approach * cfg.arrive, b.deck, cfg.lane);
    b.group.add(car);
    b.stunt = { car, phase: "approach", t0: now, cfg, s: 0, vel: V(0, 0, 0), splash: null };
  }
  function cleanupStunt(b){
    const st = b.stunt; if (!st) return;
    st.car.parent && st.car.parent.remove(st.car);
    if (st.splash) scene.remove(st.splash.group);
    b.stunt = null;
  }
  const tmpA = V(0, 0, 0), tmpB = V(0, 0, 0);
  function leafWorldPoint(b, s, p){
    const ang = THREE.MathUtils.degToRad(b.maxDeg * p);
    return b.group.localToWorld(V(-b.hinge + s * Math.cos(ang), b.deck + s * Math.sin(ang), STUNT[b.key].lane));
  }
  function makeSplash(pos){
    const group = new THREE.Group(); group.position.set(pos.x, 0, pos.z); scene.add(group);
    const n = 70, geo = new THREE.BufferGeometry(), vel = [];
    for (let i = 0; i < n; i++){ const a = rand() * Math.PI * 2, sp = 2 + rand() * 5; vel.push(V(Math.cos(a) * sp * .6, 6 + rand() * 8, Math.sin(a) * sp * .6)); }
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const drops = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xeaf6ff, size: .35, transparent: true, opacity: 1 })); group.add(drops);
    addRipple(pos.x, pos.z, 1.2);                     // the ring and froth are drawn by the water shader
    const duck = new THREE.Group(), yellow = std(0xf2c230, { roughness: .4 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(.6, 16, 12), yellow); body.scale.set(1.3, .8, 1); duck.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.38, 16, 12), yellow); head.position.set(.5, .6, 0); duck.add(head);
    const beak = new THREE.Mesh(new THREE.ConeGeometry(.12, .35, 10), std(0xff7a3c)); beak.rotation.z = -Math.PI / 2; beak.position.set(.95, .55, 0); duck.add(beak);
    for (const z of [.17, -.17]){ const eye = new THREE.Mesh(new THREE.SphereGeometry(.05, 8, 6), std(0x111111)); eye.position.set(.78, .72, z); duck.add(eye); }
    duck.scale.setScalar(1.2); duck.visible = false; group.add(duck);
    return { group, drops, vel, duck, t: 0 };
  }
  function updateStunt(b, dt, now){
    const st = b.stunt; if (!st) return;
    const cfg = st.cfg;
    if (st.phase === "approach"){
      st.car.position.x = -b.hinge + 2.1 - cfg.approach * cfg.arrive + cfg.approach * (now - st.t0);
      if (now - st.t0 >= cfg.arrive){ b.leafL.add(st.car); st.car.position.set(2.1, 0, cfg.lane); st.tRamp = st.t0 + cfg.arrive; st.phase = "ramp"; }
    }
    if (st.phase === "ramp"){
      st.s = 2.1 + cfg.rampSpeed * (now - st.tRamp); st.car.position.x = Math.min(st.s, b.L + 2.1);
      if (st.s >= b.L + 2.1){
        const p = b.p, e = 1 / 60, pNext = progressAt(b, now + e);
        tmpA.copy(leafWorldPoint(b, b.L, p)); tmpB.copy(leafWorldPoint(b, b.L, pNext));
        const leafVel = tmpB.clone().sub(tmpA).divideScalar(e);
        const ang = THREE.MathUtils.degToRad(b.maxDeg * p);
        const along = V(Math.cos(ang), Math.sin(ang), 0).transformDirection(b.group.matrixWorld).multiplyScalar(cfg.rampSpeed);
        scene.attach(st.car);
        st.vel.copy(along).add(leafVel); st.phase = "fly";
      }
    } else if (st.phase === "fly"){
      for (let left = dt; left > 0 && st.car.position.y > .4; left -= 1 / 120){ const h = Math.min(1 / 120, left); st.vel.y -= G * h; st.car.position.addScaledVector(st.vel, h); }
      const hdir = V(st.vel.x, 0, st.vel.z);
      st.car.rotation.set(0, Math.atan2(-hdir.z, hdir.x), Math.atan2(st.vel.y, hdir.length()), "YXZ");
      if (st.car.position.y <= .4){ st.phase = "sink"; st.sinkT = 0; st.splash = makeSplash(st.car.position); }
    } else if (st.phase === "sink"){
      st.sinkT += dt; st.car.position.y -= 1.2 * dt; st.car.rotation.z -= .25 * dt;
      if (!st.faded){ st.faded = []; st.car.traverse(o => { if (o.material){ o.material = o.material.clone(); o.material.transparent = true; st.faded.push(o.material); } }); }
      st.faded.forEach(m => { m.opacity = clamp(1 - st.sinkT, 0, 1); });
      if (st.sinkT > 1.1){ st.car.parent && st.car.parent.remove(st.car); st.phase = "done"; }
    }
    if (st.splash){
      const sp = st.splash; sp.t += dt;
      const pos = sp.drops.geometry.attributes.position;
      for (let i = 0; i < sp.vel.length; i++){ const v = sp.vel[i], k = sp.t; pos.setXYZ(i, v.x * k, Math.max(0, v.y * k - 9.8 * 1.6 * k * k), v.z * k); }
      pos.needsUpdate = true; sp.drops.material.opacity = clamp(1.4 - sp.t, 0, 1);
      if (sp.t > 1.6){ sp.duck.visible = true; sp.duck.position.y = Math.min(.3, (sp.t - 1.6) * 1.2 - .5) + Math.sin(now * 2.4) * .08; sp.duck.rotation.z = Math.sin(now * 1.8) * .12; }
    }
  }

  // ------------------------------------------------------------ bridges: leaves, rams, barriers, lights
  function progressAt(b, t){
    const el = t - b.t0 - (b.target === 1 ? LIFT_DELAY : 0);
    return b.from + (b.target - b.from) * EASE(clamp(el / LIFT_TIME, 0, 1));
  }
  const wA = V(0, 0, 0), wB = V(0, 0, 0);
  function updateBridge(b, now, dt){
    b.p = progressAt(b, now);
    const a = THREE.MathUtils.degToRad(b.maxDeg * b.p);
    b.leafL.rotation.z = a; b.leafR.rotation.z = -a;
    // hydraulic rams follow the leaf: barrel fixed at the pier, piston runs to the leaf
    for (const r of b.rams){
      if (r.barrel.userData.broken || r.piston.userData.broken) continue;
      r.leaf.updateMatrix();
      wB.copy(r.attach).applyMatrix4(r.leaf.matrix);          // attachment point in bridge coordinates
      const d = wB.clone().sub(r.anchor), len = d.length(), barrelLen = Math.min(3.2, len * .6);
      wA.copy(r.anchor).addScaledVector(d.normalize(), barrelLen);
      orientBetween(r.barrel, r.anchor, wA); orientBetween(r.piston, wA.clone().addScaledVector(d, -.4), wB);
    }
    // barriers: down before the lift, up again once the bridge is back down
    const wantDown = b.target === 1 || b.p > .001;
    b.barrierP = clamp(b.barrierP + (wantDown ? 1 : -1) * dt / 1.1, 0, 1);
    const k = b.barrierP * b.barrierP * (3 - 2 * b.barrierP);
    const blink = Math.floor(now * 2.4) % 2 === 0;
    for (const bar of b.barriers){
      if (bar.userData.broken) continue;
      bar.rotation.x = bar.userData.up * (1 - k);
      bar.userData.lights.forEach((l, i) => { l.material.emissive.setHex(b.barrierP > .02 && ((i % 2 === 0) === blink) ? 0xff2d40 : 0x3a1016); l.material.emissiveIntensity = b.barrierP > .02 ? 6 : 1; });
    }
    const flashing = b.target === 1 || b.p > .01, on = flashing && Math.floor(now * 2) % 2 === 0;
    b.warns.forEach((w, i) => { w.material.emissive.setHex(flashing ? ((i % 2 === 0) === on ? 0xff2d40 : 0x3a1016) : 0x3a1016); w.material.emissiveIntensity = flashing ? 6 : 1; });
  }

  // ------------------------------------------------------------ render loop
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(512, 128), .85, .55, .72);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  let running = false, raf = 0, last = 0, prevNow = 0, visible = true, active = false, camY = 12.5;
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  function resize(){
    const w = container.clientWidth || 800, h = container.clientHeight || 200;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    const pr = renderer.getPixelRatio(); mirrorRT.setSize(Math.max(64, Math.round(w * pr * .6)), Math.max(32, Math.round(h * pr * .6)));
    camera.aspect = w / h;
    const wide = camera.aspect > 2.2;
    camera.fov = wide ? 21 : 38;
    camY = wide ? 12.5 : 22;
    camera.position.set(0, camY, wide ? 100 : 135);
    camera.lookAt(0, 5, -8);
    camera.updateProjectionMatrix();
  }
  function frame(ms){
    raf = requestAnimationFrame(frame);
    const now = ms / 1000, dt = Math.min(.05, now - (last || now)); last = now;
    waterTime += dt; wu.time.value = waterTime;
    for (const r of ripples) if (r.w > 0){ r.z += dt; if (r.z > 9) r.w = 0; }
    if (!reduce){ camera.position.x = Math.sin(now * .05) * 5; camera.lookAt(0, 5, -8); }
    const realDt = Math.min(.25, now - (prevNow || now)); prevNow = now;
    for (const b of bridges){ updateBridge(b, now, dt); updateTraffic(b, realDt); updateStunt(b, realDt, now); }
    updateBoats(dt, now); updateExplosions(dt);
    if (alarm) updateAlarm(dt, now);
    camera.position.y = camY;
    if (shake > .01 && !reduce){ camera.position.x += (Math.random() - .5) * shake; camera.position.y += (Math.random() - .5) * shake * .6; shake *= Math.exp(-dt * 5); }
    composer.render();
  }
  function sync(){
    const should = active && visible;
    if (should && !running){ running = true; last = 0; raf = requestAnimationFrame(frame); }
    if (!should && running){ running = false; cancelAnimationFrame(raf); }
  }
  new ResizeObserver(resize).observe(container);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; sync(); }).observe(container);
  resize();
  for (const b of bridges) updateBridge(b, performance.now() / 1000, 0);

  return {
    setActive(on){ active = on; if (on) resize(); sync(); },
    // alarm mode (stays on until the page reloads): red sky, a meteor shower, the bridges wrecked
    setAlarm(){
      if (alarm) return;
      alarm = true; captureLook(); makePools();
      for (const b of bridges) collectPieces(b);
      nextMeteor = clockA + .6;
    },
    _alarm(){
      return { k: +alarmK.toFixed(2), meteors: meteors3.length, wrecks: wrecks.length, burners: burners.length, explosions: explosionCount,
               broken: bridges.map(b => `${(b.pieces || []).filter(o => o.userData.broken).length}/${(b.pieces || []).length}`) };
    },
    _stats(){
      const onRaised = bridges.map(b => b.p > .02 ? b.cars.filter(c => onSpan(b, c.userData.x)).length : 0);
      return { explosions: explosionCount, carsOnRaisedSpan: onRaised, p: bridges.map(b => +b.p.toFixed(3)) };
    },
    _water(){ return { boats: wu.boatN.value, obstacles: wu.obN.value, boatA: wu.boatA.value.slice(0, wu.boatN.value).map(v => v.toArray().map(x => +x.toFixed(2))), boatB: wu.boatB.value.slice(0, wu.boatN.value).map(v => v.toArray().map(x => +x.toFixed(2))), ripples: ripples.map(v => v.toArray().map(x => +x.toFixed(2))) }; },
    _testCollision(){                    // puts the RIB and the launch head-on in view; used by the automated test
      const [rib, launch] = boats;
      rib.userData.dead = launch.userData.dead = false;
      rib.userData.t = .5; launch.userData.t = .48;
      const mid = V(0, 0, 0).lerpVectors(rib.userData.from, rib.userData.to, .5);
      launch.userData.from = mid.clone().add(V(40, 0, 0)); launch.userData.to = mid.clone().add(V(-40, 0, 0)); launch.userData.t = .45;
    },
    // states: { POOLE: bool lifting, TWIN: bool lifting, color: 'BOTH'|'POOLE'|'TWIN'|'OTHER' }
    setStates(states){
      const now = performance.now() / 1000;
      for (const b of bridges){
        const want = states[b.key] ? 1 : 0;
        if (want !== b.target){
          b.from = progressAt(b, now); b.target = want; b.t0 = now;
          if (want === 1 && !b.broken) startStunt(b, now); else cleanupStunt(b);
        }
      }
      const c = STATE_COLORS[states.color] ?? STATE_COLORS.OTHER;
      twin.dmxMat.emissive.setHex(c); twin.edgeMat.emissive.setHex(c);
    },
  };
}
