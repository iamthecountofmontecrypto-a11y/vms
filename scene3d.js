// 3D bridge scene (optional; loaded only when the "3D" toggle is on).
// Twin Sails (left) and Poole Bridge (right) as real 3D models: reflective water, night
// lighting with bloom, traffic that queues for lifts, boats, and the last-minute stunt car.
// Units are metres; roads run along x, the channel runs along z, y is up.
import * as THREE from "three";
import { Water } from "three/addons/objects/Water.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

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

// ---------------------------------------------------------------- procedural textures
function waterNormals(){
  const N = 256, data = new Uint8Array(N * N * 4);
  const waves = [[3, 1, .8, .1], [1, 4, .6, 1.7], [5, 3, .35, 2.4], [7, -2, .25, .6], [2, 7, .3, 3.1], [11, 5, .12, 1.2], [4, -9, .14, .4]];
  const h = (x, y) => waves.reduce((s, [kx, ky, a, p]) => s + a * Math.sin(2 * Math.PI * (kx * x + ky * y) / N + p), 0);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++){
    const dx = h(x + 1, y) - h(x - 1, y), dy = h(x, y + 1) - h(x, y - 1);
    const nx = -dx * 2.2, ny = -dy * 2.2, nz = 1, l = Math.hypot(nx, ny, nz), i = (y * N + x) * 4;
    data[i] = (nx / l * .5 + .5) * 255; data[i + 1] = (ny / l * .5 + .5) * 255; data[i + 2] = (nz / l * .5 + .5) * 255; data[i + 3] = 255;
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.needsUpdate = true;
  return t;
}
function canvasTex(w, h, draw, srgb = true){
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
const skyTex = () => canvasTex(4, 256, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, 0, h);
  gr.addColorStop(0, "#02040c"); gr.addColorStop(.6, "#060b22"); gr.addColorStop(.85, "#0c1433"); gr.addColorStop(1, "#131c42");
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
});
const towerFacade = () => canvasTex(256, 320, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, w, 0);
  gr.addColorStop(0, "#5aa895"); gr.addColorStop(.5, "#7cc8b3"); gr.addColorStop(1, "#4d9584");
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
  g.globalAlpha = .18; g.fillStyle = "#123b33";                       // weathered patches
  for (let i = 0; i < 60; i++) g.fillRect(Math.random() * w, Math.random() * h, 6 + Math.random() * 30, 2 + Math.random() * 10);
  g.globalAlpha = .35; g.strokeStyle = "#1f4a41"; g.lineWidth = 2;   // cladding seams
  for (let x = 32; x < w; x += 40){ g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
  for (let y = 40; y < h; y += 56){ g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  g.globalAlpha = 1;
  g.fillStyle = "#2e5ea8"; g.fillRect(w * .62, h * .28, w * .2, h * .5);   // blue panel
  g.fillStyle = "#1d3f7a"; g.strokeStyle = "#e8c45a"; g.lineWidth = 4;     // borough coat of arms
  g.beginPath(); g.moveTo(w * .18, h * .3); g.lineTo(w * .44, h * .3); g.lineTo(w * .44, h * .45);
  g.quadraticCurveTo(w * .44, h * .56, w * .31, h * .6); g.quadraticCurveTo(w * .18, h * .56, w * .18, h * .45); g.closePath(); g.fill(); g.stroke();
  g.fillStyle = "#e8c45a"; g.fillRect(w * .22, h * .36, w * .18, 5); g.fillRect(w * .22, h * .42, w * .18, 5);
  g.fillStyle = "#3d6e62"; g.fillRect(w * .3, h * .82, w * .14, h * .18);   // door
});
const trussTex = () => canvasTex(512, 64, (g, w, h) => {
  g.clearRect(0, 0, w, h); g.strokeStyle = "#2fb3a3"; g.lineWidth = 7; g.lineJoin = "round";
  g.beginPath(); g.moveTo(0, 6); g.lineTo(w, 6); g.moveTo(0, h - 6); g.lineTo(w, h - 6);
  for (let i = 0; i <= 10; i++){ const x = i * w / 10; g.lineTo(x, i % 2 ? 6 : h - 6); }
  g.stroke();
});

// ---------------------------------------------------------------- small builders
const std = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: .7, metalness: .1, ...o });
const glow = (color, intensity = 3) => new THREE.MeshStandardMaterial({ color: 0x000000, emissive: color, emissiveIntensity: intensity });
function box(w, h, d, mat, x = 0, y = 0, z = 0){
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); return m;
}
function lamp(x, z, deckY, color = 0xffd98a, withLight = false){
  const g = new THREE.Group();
  g.add(box(.18, 4.2, .18, std(0x4a5263), x, deckY + 2.1, z));
  g.add(box(.9, .12, .18, std(0x4a5263), x + (z > 0 ? 0 : 0), deckY + 4.2, z));
  const head = new THREE.Mesh(new THREE.SphereGeometry(.28, 12, 8), glow(color, 6)); head.position.set(x, deckY + 4.05, z); g.add(head);
  if (withLight){ const l = new THREE.PointLight(color, 18, 16, 2); l.position.set(x, deckY + 3.8, z); g.add(l); }
  return g;
}
function makeCar(color){
  const g = new THREE.Group();
  g.add(box(4.2, 1.0, 1.8, std(color, { roughness: .35, metalness: .5 }), 0, .75, 0));
  g.add(box(2.2, .7, 1.6, std(0x1a2233, { roughness: .2, metalness: .6 }), -.2, 1.55, 0));
  g.add(box(.08, .25, .4, glow(0xfff6c8, 5), 2.12, .85, .6)); g.add(box(.08, .25, .4, glow(0xfff6c8, 5), 2.12, .85, -.6));
  g.add(box(.08, .22, .4, glow(0xff3040, 4), -2.12, .9, .6)); g.add(box(.08, .22, .4, glow(0xff3040, 4), -2.12, .9, -.6));
  for (const [x, z] of [[1.3, .92], [1.3, -.92], [-1.3, .92], [-1.3, -.92]]){
    const w = new THREE.Mesh(new THREE.CylinderGeometry(.38, .38, .25, 14), std(0x0b0f18)); w.rotation.x = Math.PI / 2; w.position.set(x, .38, z); g.add(w);
  }
  return g;
}
function makeBoat(kind){
  const g = new THREE.Group();
  const hull = (len, beam, h, color) => {
    const s = new THREE.Shape(); s.moveTo(-len / 2, -beam / 2); s.lineTo(len * .3, -beam / 2); s.quadraticCurveTo(len / 2, -beam * .3, len / 2 + .4, 0);
    s.quadraticCurveTo(len / 2, beam * .3, len * .3, beam / 2); s.lineTo(-len / 2, beam / 2); s.closePath();
    const geo = new THREE.ExtrudeGeometry(s, { depth: h, bevelEnabled: false }); geo.rotateX(-Math.PI / 2);
    return new THREE.Mesh(geo, std(color, { roughness: .4 }));
  };
  const nav = (x, z, c) => { const m = new THREE.Mesh(new THREE.SphereGeometry(.14, 8, 6), glow(c, 3.5)); m.position.set(x, 1.4, z); g.add(m); };
  if (kind === "yacht"){
    g.add(hull(9, 2.8, 1.1, 0xe9eef6)); g.add(box(3, .7, 1.8, std(0xcdd6e4), -.5, 1.45, 0));
    g.add(box(.12, 12, .12, std(0xdfe6ef), .2, 7, 0));
    const sail = new THREE.Shape(); sail.moveTo(0, 0); sail.lineTo(0, 10.5); sail.lineTo(-3.8, 0); sail.closePath();
    const sm = new THREE.Mesh(new THREE.ShapeGeometry(sail), std(0xf4f7fb, { side: THREE.DoubleSide, roughness: .9 })); sm.position.set(.1, 1.6, 0); g.add(sm);
    const top = new THREE.Mesh(new THREE.SphereGeometry(.16, 8, 6), glow(0xffffff, 8)); top.position.set(.2, 13, 0); g.add(top);
  } else if (kind === "superyacht"){
    g.add(hull(18, 4.2, 1.6, 0xf4f7fb)); g.add(box(11, 1.4, 3.4, std(0xe9eef6), -1, 2.3, 0)); g.add(box(6, 1.1, 2.8, std(0xdfe6ef), -1.5, 3.5, 0));
    g.add(box(11.2, .45, 3.5, glow(0x7fe3ff, 1.5), -1, 2.1, 0));
    const under = new THREE.PointLight(0x2de2ff, 30, 14, 2); under.position.set(0, -.5, 0); g.add(under);
  } else if (kind === "trawler"){
    g.add(hull(11, 3.4, 1.8, 0xa8322d)); g.add(box(3, 2.2, 2.6, std(0xe9eef6), -2.5, 2.9, 0)); g.add(box(3.1, .5, 2.7, glow(0xffe2b8, 1.3), -2.5, 3.1, 0));
    g.add(box(.14, 6, .14, std(0xc9d2e0), 1.5, 4.8, 0));
    const flood = new THREE.PointLight(0xfff6c8, 22, 14, 2); flood.position.set(1.5, 7, 0); g.add(flood);
  } else if (kind === "tug"){
    g.add(hull(9, 3.6, 1.7, 0x1a1d27)); g.add(box(3.2, 1.8, 2.6, std(0xe56a2c), -.5, 2.6, 0)); g.add(box(.9, 1.6, .9, std(0x1a1d27), -.5, 4.2, 0));
    g.add(box(3.3, .4, 2.7, glow(0xffe2b8, 1.3), -.5, 2.8, 0));
  } else if (kind === "rib"){
    g.add(hull(5.5, 2.2, .7, 0x2a2f3c)); g.add(box(1, .9, .9, std(0xcfd6e2), -.3, 1.1, 0));
  } else if (kind === "launch"){
    g.add(hull(6.5, 2.4, .9, 0xe9eef6)); g.add(box(2.6, 1, 1.8, std(0x2b5fb3), -.6, 1.3, 0)); g.add(box(2.7, .3, 1.9, glow(0xffe2b8, 1.1), -.6, 1.5, 0));
  }
  nav(0, 1.2, 0xff3040); nav(0, -1.2, 0x41ff8b);
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
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";

  const scene = new THREE.Scene();
  scene.background = skyTex();
  scene.fog = new THREE.FogExp2(0x080d26, 0.0075);
  const camera = new THREE.PerspectiveCamera(24, 4, .5, 2500);

  scene.add(new THREE.HemisphereLight(0x5566a8, 0x0a0c18, .9));
  const moonLight = new THREE.DirectionalLight(0xc6d4ff, 1.6); moonLight.position.set(-60, 90, 40); scene.add(moonLight);

  // moon, halo and stars
  const moon = new THREE.Mesh(new THREE.SphereGeometry(9, 32, 16), new THREE.MeshBasicMaterial({ color: 0xfff6dc, fog: false }));
  moon.position.set(60, 95, -420); scene.add(moon);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(128, 128, (g, w) => {
    const r = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    r.addColorStop(0, "rgba(255,246,220,.55)"); r.addColorStop(.3, "rgba(255,246,220,.12)"); r.addColorStop(1, "rgba(255,246,220,0)");
    g.fillStyle = r; g.fillRect(0, 0, w, w); }), transparent: true, depthWrite: false, fog: false }));
  halo.scale.set(120, 120, 1); halo.position.copy(moon.position); scene.add(halo);
  const starGeo = new THREE.BufferGeometry(), starPos = [];
  for (let i = 0; i < 700; i++){
    const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * .42 + .08, r = 1400;
    starPos.push(r * Math.cos(th) * Math.cos(ph), r * Math.sin(ph), r * Math.sin(th) * Math.cos(ph));
  }
  starGeo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 1.6, sizeAttenuation: false, fog: false, transparent: true, opacity: .8 })));

  // water (reflective, with a procedural normal map)
  const water = new Water(new THREE.PlaneGeometry(1200, 1200), {
    textureWidth: 512, textureHeight: 512, waterNormals: waterNormals(),
    sunDirection: new THREE.Vector3(-.4, .8, .45).normalize(), sunColor: 0x5d73b8, waterColor: 0x020915,
    distortionScale: 1.7, fog: true,
  });
  water.rotation.x = -Math.PI / 2; water.material.uniforms.size.value = 4.5; scene.add(water);

  // distant shoreline with town lights
  const shore = new THREE.Group();
  for (let i = 0; i < 28; i++){
    const w = 14 + Math.random() * 26, h = 4 + Math.random() * 16, x = -330 + i * 24 + Math.random() * 8, z = -150 - Math.random() * 40;
    shore.add(box(w, h, 10, std(0x0b1226), x, h / 2, z));
    for (let k = 0; k < 4; k++) if (Math.random() < .7) shore.add(box(.9, .6, .2, glow(0xffd98a, 2.2 + Math.random() * 2), x - w / 2 + 2 + Math.random() * (w - 4), 1.5 + Math.random() * (h - 2.5), z + 5.2));
  }
  shore.add(box(900, 3, 60, std(0x080d1c), 0, 1.4, -180));
  scene.add(shore);

  // ------------------------------------------------------------ Twin Sails Bridge
  const twin = { key: "TWIN", group: new THREE.Group(), p: 0, from: 0, target: 0, t0: -99, maxDeg: 88, hinge: 8.8, L: 17.6, W: 12, deck: 6, lanes: [3, -3], stops: [-17, 17] };
  {
    const g = twin.group, { L, W, deck } = twin;
    const concrete = std(0x9aa2b2, { roughness: .9 }), deckMat = std(0xe8edf5, { roughness: .45, metalness: .15 });
    for (const sx of [-1, 1]){
      g.add(box(4.4, deck + 1, W + 2, concrete, sx * (twin.hinge + 2.2), (deck - 1) / 2 - .5, 0));   // main piers
      g.add(box(34, 1, W, deckMat, sx * (twin.hinge + 4.4 + 17), deck - .5, 0));                        // approach spans
      for (const px of [22, 34]) g.add(box(2, deck, 2.4, concrete, sx * px, deck / 2 - 1, 0));
      g.add(box(34, 1.1, .12, std(0xaab4c4, { metalness: .6, roughness: .3 }), sx * (twin.hinge + 4.4 + 17), deck + .55, W / 2 - .06));
      g.add(box(34, 1.1, .12, std(0xaab4c4, { metalness: .6, roughness: .3 }), sx * (twin.hinge + 4.4 + 17), deck + .55, -W / 2 + .06));
      g.add(lamp(sx * 26, W / 2 - .4, deck, 0xdff3ff, true)); g.add(lamp(sx * 18, -W / 2 + .4, deck, 0xdff3ff, false));
      const wash = new THREE.PointLight(0xcfe0ff, 60, 30, 1.6); wash.position.set(sx * twin.hinge, deck + 4, W / 2 + 5); g.add(wash);
      const post = box(.7, 5.5, .7, std(0x6f7888), sx * 16, deck + 2.75, -W / 2 - .6); g.add(post);
      const warn = new THREE.Mesh(new THREE.SphereGeometry(.3, 10, 8), glow(0x3a1016, 1)); warn.position.set(sx * 16, deck + 5.2, -W / 2 - .6); g.add(warn);
      (twin.warns ||= []).push(warn);
      g.add(box(12, deck + 2, W + 20, std(0x0e1628), sx * 44, (deck + 2) / 2 - 1, 0));                   // quay
    }
    // balustrade lights (take the colour of the current sign message)
    const dmxGeo = new THREE.SphereGeometry(.13, 8, 6);
    twin.dmxMat = glow(STATE_COLORS.BOTH, 4);
    const dmx = new THREE.InstancedMesh(dmxGeo, twin.dmxMat, 40); let n = 0;
    const m4 = new THREE.Matrix4();
    for (const sx of [-1, 1]) for (let i = 0; i < 10; i++) for (const sz of [-1, 1]){
      m4.makeTranslation(sx * (twin.hinge + 5 + i * 3.3), deck + 1.15, sz * (W / 2 - .06)); dmx.setMatrixAt(n++, m4);
    }
    g.add(dmx);
    // leaves: triangles split diagonally across the road, hinged at each pier
    const leafShape = pts => { const s = new THREE.Shape(); s.moveTo(pts[0][0], pts[0][1]); pts.slice(1).forEach(p => s.lineTo(p[0], p[1])); s.closePath(); return s; };
    const leafMat = std(0xeef2f8, { roughness: .45, metalness: .35 }), mastMat = std(0xf2f5fa, { metalness: .6, roughness: .3 });
    twin.edgeMat = glow(STATE_COLORS.BOTH, 3);
    const makeLeaf = side => {
      const pivot = new THREE.Group(); pivot.position.set(side * twin.hinge, deck, 0); g.add(pivot);
      const pts = side < 0 ? [[0, -W / 2], [0, W / 2], [L, W / 2]] : [[0, -W / 2], [0, W / 2], [-L, -W / 2]];
      const geo = new THREE.ExtrudeGeometry(leafShape(pts), { depth: 1.1, bevelEnabled: false }); geo.rotateX(Math.PI / 2);
      pivot.add(new THREE.Mesh(geo, leafMat));
      // mast along the long edge, running past the tip, with a glowing white tip
      const zEdge = side < 0 ? W / 2 - .1 : -W / 2 + .1, dir = side < 0 ? 1 : -1, len = L * 1.12;
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(.14, .2, len, 10), mastMat); mast.rotation.z = Math.PI / 2; mast.position.set(dir * len / 2, .35, zEdge); pivot.add(mast);
      const tip = new THREE.Mesh(new THREE.SphereGeometry(.34, 12, 8), glow(0xffffff, 10)); tip.position.set(dir * len, .35, zEdge); pivot.add(tip);
      // neon strip along the diagonal edge
      const a = new THREE.Vector3(0, .05, side < 0 ? -W / 2 : W / 2), b = new THREE.Vector3(dir * L, .05, side < 0 ? W / 2 : -W / 2);
      const strip = new THREE.Mesh(new THREE.BoxGeometry(a.distanceTo(b), .08, .08), twin.edgeMat);
      strip.position.copy(a.clone().add(b).multiplyScalar(.5)); strip.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x); pivot.add(strip);
      return pivot;
    };
    twin.leafL = makeLeaf(-1); twin.leafR = makeLeaf(1);
  }
  twin.group.position.set(-35, 0, -6); twin.group.rotation.y = .62;
  scene.add(twin.group);

  // ------------------------------------------------------------ Poole Bridge
  const poole = { key: "POOLE", group: new THREE.Group(), p: 0, from: 0, target: 0, t0: -99, maxDeg: 74, hinge: 12.3, L: 12.3, W: 10, deck: 4, lanes: [2.2, -2.2], stops: [-24, 24] };
  {
    const g = poole.group, { L, W, deck } = poole;
    const facade = towerFacade(), green = std(0x5fae9b, { roughness: .8 });
    const towerMats = [green, green, std(0x5aa593), green, std(0xffffff, { map: facade, roughness: .8 }), std(0xffffff, { map: facade, roughness: .8 })];
    poole.warns = [];
    for (const sx of [-1, 1]){
      const cx = sx * (poole.hinge + 4.1);
      for (const sz of [-1, 1]){                                   // a tower each side of the road
        const t = new THREE.Mesh(new THREE.BoxGeometry(8.2, 14, 3.6), towerMats); t.position.set(cx, 7 - 1, sz * (W / 2 + 1.8)); g.add(t);
        g.add(box(8.8, .6, 4.2, std(0x5aa593), cx, 13.3, sz * (W / 2 + 1.8)));
        g.add(box(3.2, .8, 2, std(0x4d9b89), cx, 14, sz * (W / 2 + 1.8)));
        for (const wx of [-2.4, 1.6]) g.add(box(.6, 1, .05, glow(0xffd98a, 3.2), cx + wx, 10.8, sz * (W / 2 + 1.8) + sz * 1.83));
        const sig = box(.9, 1.8, .5, std(0x141a26), cx - sx * 3.2, 15.2, sz * (W / 2 + 1.8)); g.add(sig);
        const w1 = new THREE.Mesh(new THREE.SphereGeometry(.26, 10, 8), glow(0x3a1016, 1)); w1.position.set(cx - sx * 3.2, 15.6, sz * (W / 2 + 1.8) + sz * .28); g.add(w1);
        poole.warns.push(w1);
        const up = new THREE.SpotLight(0xfff3c4, 90, 22, .55, .6, 1.6); up.position.set(cx, .5, sz * (W / 2 + 1.8) + sz * 6);
        up.target.position.set(cx, 10, sz * (W / 2 + 1.8)); g.add(up, up.target);
      }
      // lattice portal over the road between each pair of towers
      const portal = new THREE.Mesh(new THREE.PlaneGeometry(W + 0.2, 2.2), new THREE.MeshStandardMaterial({ map: trussTex(), transparent: true, alphaTest: .4, side: THREE.DoubleSide, emissive: 0x0d3f39, roughness: .6 }));
      portal.rotation.y = Math.PI / 2; portal.position.set(cx, 12, 0); g.add(portal);
      g.add(box(44, .9, W, std(0x3b4150, { roughness: .9 }), sx * (poole.hinge + 8.2 + 22), deck - .45, 0));   // approaches
      g.add(lamp(sx * 32, W / 2 - .4, deck, 0xffd98a, true)); g.add(lamp(sx * 44, -W / 2 + .4, deck, 0xffd98a, false));
      for (const px of [-2, 2]) g.add(box(.6, deck + 1, .6, std(0x2b2a2e), sx * (poole.hinge + 1) + px, (deck - 1) / 2, W / 2 + 4));
      g.add(box(14, deck + 2, W + 18, std(0x0e1628), sx * 58, (deck + 2) / 2 - 1, 0));                          // quay
      for (let i = 0; i < 3; i++) g.add(box(10, 6 + i * 3, 8, std(0x101a33), sx * (50 + i * 11), (6 + i * 3) / 2, -26 - i * 6));   // sheds
    }
    const truss = trussTex(); truss.wrapS = THREE.RepeatWrapping;
    const trussMat = new THREE.MeshStandardMaterial({ map: truss, transparent: true, alphaTest: .4, side: THREE.DoubleSide, emissive: 0x0d3f39, roughness: .6, metalness: .3 });
    const makeLeaf = side => {
      const pivot = new THREE.Group(); pivot.position.set(side * poole.hinge, deck, 0); g.add(pivot);
      const dir = side < 0 ? 1 : -1;
      pivot.add(box(L, .9, W, std(0x2c55b0, { roughness: .5, metalness: .4 }), dir * L / 2, -.45, 0));
      pivot.add(box(L, .9, W - .6, std(0x17307a, { roughness: .6 }), dir * L / 2, -1.3, 0));
      for (const sz of [-1, 1]){
        const tp = new THREE.Mesh(new THREE.PlaneGeometry(L, 1.7), trussMat); tp.position.set(dir * L / 2, .85, sz * (W / 2 - .05)); pivot.add(tp);
        pivot.add(box(L, .08, .08, std(0xc7d3e0, { metalness: .6 }), dir * L / 2, 1.9, sz * (W / 2 - .05)));
      }
      pivot.add(box(.3, 1.6, W, std(0xf2c230), dir * (L - .15), -.8, 0));
      return pivot;
    };
    poole.leafL = makeLeaf(-1); poole.leafR = makeLeaf(1);
  }
  poole.group.position.set(37, 0, 0); poole.group.rotation.y = -.42;
  scene.add(poole.group);
  const bridges = [twin, poole];

  // ------------------------------------------------------------ traffic (queues at the stop lines during a lift)
  for (const b of bridges){
    b.cars = [];
    const colors = [0x20283a, 0x39425a, 0x6b7a90, 0x8a1f2a, 0x2e3a55, 0xd9dee8];
    for (let i = 0; i < 6; i++){
      const lane = i % 2, car = makeCar(colors[i]);
      car.userData = { lane, dir: lane === 0 ? 1 : -1, x: -60 + i * 21, v: 11 + Math.random() * 3 };
      car.position.set(car.userData.x, b.deck, b.lanes[lane]); car.rotation.y = lane === 0 ? 0 : Math.PI;
      b.group.add(car); b.cars.push(car);
    }
  }
  function updateTraffic(b, dt){
    const closed = b.target === 1 || b.p > .002;
    for (const lane of [0, 1]){
      const cars = b.cars.filter(c => c.userData.lane === lane).sort((a, c) => (c.userData.x - a.userData.x) * (lane === 0 ? 1 : -1));
      cars.forEach((car, i) => {
        const u = car.userData, dir = u.dir;
        let limit = Infinity;
        const stop = lane === 0 ? b.stops[0] : b.stops[1];
        if (closed && dir * (u.x - stop) < 0) limit = Math.min(limit, dir * (stop - u.x));
        const ahead = cars[i - 1];
        if (ahead && dir * (ahead.userData.x - u.x) > 0) limit = Math.min(limit, dir * (ahead.userData.x - u.x) - 6.5);
        const step = Math.max(0, Math.min(u.v * dt, limit));
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
  // small craft cruising in the foreground whatever the bridges are doing
  addBoat("rib", null, { from: new THREE.Vector3(-130, 0, 42), to: new THREE.Vector3(130, 0, 30), speed: 14, always: true });
  addBoat("launch", null, { from: new THREE.Vector3(140, 0, 52), to: new THREE.Vector3(-140, 0, 58), speed: 6, always: true, phase: .4 });
  // tall boats through each channel once it's open
  const channel = (b, dz, lane, kind, speed, phase) => {
    const a = new THREE.Vector3(lane, 0, 80 * dz), c = new THREE.Vector3(lane, 0, -80 * dz);
    b.group.localToWorld(a); b.group.localToWorld(c);
    return addBoat(kind, b, { from: a, to: c, speed, phase });
  };
  twin.group.updateMatrixWorld(); poole.group.updateMatrixWorld();
  channel(twin, 1, -2, "yacht", 6, 0); channel(twin, -1, 2.5, "superyacht", 4.5, .5);
  channel(poole, 1, -1.8, "trawler", 5, 0); channel(poole, -1, 2, "tug", 4.2, .45); channel(poole, 1, .5, "yacht", 5.5, .7);
  function updateBoats(dt, t){
    for (const m of boats){
      const u = m.userData, len = u.from.distanceTo(u.to);
      if (u.always){ u.t = (u.t + u.speed * dt / len) % 1; }
      else {
        const open = u.b.p > .85, crossing = u.t > 0;
        if (!crossing && open){ u.wait -= dt; if (u.wait <= 0) u.t = 1e-4; }
        if (u.t > 0){ u.t += u.speed * dt / len; if (u.t >= 1){ u.t = 0; u.wait = 6 + Math.random() * 6; } }
      }
      m.visible = u.always || u.t > 0;
      m.position.lerpVectors(u.from, u.to, u.t);
      m.position.y = .05 + Math.sin(t * 1.7 + u.t * 40) * .08;
      m.lookAt(u.to.x, m.position.y, u.to.z); m.rotateY(-Math.PI / 2);
      m.rotation.z += Math.sin(t * 1.3 + len) * .02;
    }
  }

  // ------------------------------------------------------------ the last-minute car
  const G = 95;                        // cartoon gravity (same timing as the 2D version)
  const STUNT = { TWIN: { lane: 4.3, rampSpeed: 19, approach: 10, arrive: 2.7 }, POOLE: { lane: 2.2, rampSpeed: 12, approach: 10, arrive: 2.5 } };
  function startStunt(b, now){
    if (b.stunt) cleanupStunt(b);
    const cfg = STUNT[b.key], car = makeCar(0xe0343f);
    const hingeX = -b.hinge;
    car.position.set(hingeX + 2.1 - cfg.approach * cfg.arrive, b.deck, cfg.lane);
    b.group.add(car);
    b.stunt = { car, phase: "approach", t0: now, cfg, s: 0, vel: new THREE.Vector3(), splash: null };
  }
  function cleanupStunt(b){
    const st = b.stunt; if (!st) return;
    st.car.parent && st.car.parent.remove(st.car);
    if (st.splash){ scene.remove(st.splash.group); }
    b.stunt = null;
  }
  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();
  function leafWorldPoint(b, s, p){          // world position of a point s metres along the left leaf's deck at progress p
    const ang = THREE.MathUtils.degToRad(b.maxDeg * p);
    const local = new THREE.Vector3(-b.hinge + s * Math.cos(ang), b.deck + s * Math.sin(ang), STUNT[b.key].lane);
    return b.group.localToWorld(local);
  }
  function makeSplash(pos){
    const group = new THREE.Group(); group.position.set(pos.x, 0, pos.z); scene.add(group);
    const n = 60, geo = new THREE.BufferGeometry(), arr = new Float32Array(n * 3), vel = [];
    for (let i = 0; i < n; i++){ const a = Math.random() * Math.PI * 2, sp = 2 + Math.random() * 5; vel.push(new THREE.Vector3(Math.cos(a) * sp * .6, 6 + Math.random() * 8, Math.sin(a) * sp * .6)); }
    geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const drops = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xeaf6ff, size: .35, transparent: true, opacity: 1 })); group.add(drops);
    const ring = new THREE.Mesh(new THREE.RingGeometry(.8, 1.1, 48), new THREE.MeshBasicMaterial({ color: 0xdff3ff, transparent: true, opacity: .9, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = .06; group.add(ring);
    const duck = new THREE.Group();
    const yellow = std(0xf2c230, { roughness: .4 });
    const body = new THREE.Mesh(new THREE.SphereGeometry(.6, 16, 12), yellow); body.scale.set(1.3, .8, 1); duck.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.38, 16, 12), yellow); head.position.set(.5, .6, 0); duck.add(head);
    const beak = new THREE.Mesh(new THREE.ConeGeometry(.12, .35, 10), std(0xff7a3c)); beak.rotation.z = -Math.PI / 2; beak.position.set(.95, .55, 0); duck.add(beak);
    duck.scale.setScalar(1.2); duck.visible = false; group.add(duck);
    return { group, drops, vel, ring, duck, t: 0 };
  }
  function updateStunt(b, dt, now){
    const st = b.stunt; if (!st) return;
    const cfg = st.cfg;
    if (st.phase === "approach"){
      st.car.position.x += cfg.approach * dt;
      if (st.car.position.x >= -b.hinge + 2.1){                     // front wheels at the hinge: onto the leaf
        (b.leafL).add(st.car); st.car.position.set(2.1, 0, cfg.lane); st.s = 2.1; st.phase = "ramp";
      }
    } else if (st.phase === "ramp"){
      st.s += cfg.rampSpeed * dt; st.car.position.x = st.s;
      if (st.s >= b.L + 2.1){                                       // rear wheels leave the tip: airborne
        const p = b.p, e = 1 / 60, pNext = progressAt(b, now + e);
        tmpA.copy(leafWorldPoint(b, b.L, p)); tmpB.copy(leafWorldPoint(b, b.L, pNext));
        const leafVel = tmpB.clone().sub(tmpA).divideScalar(e);
        const ang = THREE.MathUtils.degToRad(b.maxDeg * p);
        const along = new THREE.Vector3(Math.cos(ang), Math.sin(ang), 0).transformDirection(b.group.matrixWorld).multiplyScalar(cfg.rampSpeed);
        scene.attach(st.car);
        st.vel.copy(along).add(leafVel); st.phase = "fly";
      }
    } else if (st.phase === "fly"){
      st.vel.y -= G * dt; st.car.position.addScaledVector(st.vel, dt);
      const hdir = new THREE.Vector3(st.vel.x, 0, st.vel.z);
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
      for (let i = 0; i < sp.vel.length; i++){
        const v = sp.vel[i], k = sp.t;
        pos.setXYZ(i, v.x * k, Math.max(0, v.y * k - 9.8 * 1.6 * k * k), v.z * k);
      }
      pos.needsUpdate = true; sp.drops.material.opacity = clamp(1.4 - sp.t, 0, 1);
      sp.ring.scale.setScalar(1 + sp.t * 6); sp.ring.material.opacity = clamp(.9 - sp.t * .6, 0, .9);
      if (sp.t > 1.6){ sp.duck.visible = true; sp.duck.position.y = Math.min(.3, (sp.t - 1.6) * 1.2 - .5) + Math.sin(now * 2.4) * .08; sp.duck.rotation.z = Math.sin(now * 1.8) * .12; }
    }
  }

  // ------------------------------------------------------------ bridges: leaf motion, lights
  function progressAt(b, t){
    const el = t - b.t0 - (b.target === 1 ? LIFT_DELAY : 0);
    return b.from + (b.target - b.from) * EASE(clamp(el / LIFT_TIME, 0, 1));
  }
  function updateBridge(b, now){
    b.p = progressAt(b, now);
    const a = THREE.MathUtils.degToRad(b.maxDeg * b.p);
    b.leafL.rotation.z = a; b.leafR.rotation.z = -a;
    const flashing = b.target === 1 || b.p > .01, on = flashing && Math.floor(now * 2) % 2 === 0;
    b.warns.forEach((w, i) => { w.material.emissive.setHex(flashing ? ((i % 2 === 0) === on ? 0xff2d40 : 0x3a1016) : 0x3a1016); w.material.emissiveIntensity = flashing ? 6 : 1; });
  }

  // ------------------------------------------------------------ render loop
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(512, 128), .85, .55, .72);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  let running = false, raf = 0, last = 0, visible = true, active = false;
  const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  function resize(){
    const w = container.clientWidth || 800, h = container.clientHeight || 200;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    camera.aspect = w / h;
    const wide = camera.aspect > 2.2;
    camera.fov = wide ? 21 : 38;
    camera.position.set(0, wide ? 12.5 : 22, wide ? 100 : 135);
    camera.lookAt(0, 6.5, -8);
    camera.updateProjectionMatrix();
  }
  function frame(ms){
    raf = requestAnimationFrame(frame);
    const now = ms / 1000, dt = Math.min(.05, now - (last || now)); last = now;
    water.material.uniforms.time.value += dt * .6;
    if (!reduce){ camera.position.x = Math.sin(now * .05) * 5; camera.lookAt(0, 6.5, -8); }
    for (const b of bridges){ updateBridge(b, now); updateTraffic(b, dt); updateStunt(b, dt, now); }
    updateBoats(dt, now);
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

  return {
    setActive(on){ active = on; if (on) resize(); sync(); },
    // states: { POOLE: bool lifting, TWIN: bool lifting, color: 'BOTH'|'POOLE'|'TWIN'|'OTHER' }
    setStates(states){
      const now = performance.now() / 1000;
      for (const b of bridges){
        const want = states[b.key] ? 1 : 0;
        if (want !== b.target){
          b.from = progressAt(b, now); b.target = want; b.t0 = now;
          if (want === 1) startStunt(b, now); else cleanupStunt(b);
        }
      }
      const c = STATE_COLORS[states.color] ?? STATE_COLORS.OTHER;
      twin.dmxMat.emissive.setHex(c); twin.edgeMat.emissive.setHex(c);
    },
  };
}
