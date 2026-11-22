function _define_property(obj,key,value){if(key in obj){Object.defineProperty(obj,key,{value:value,enumerable:true,configurable:true,writable:true});}else{obj[key]=value;}return obj;}
import*as THREE from"three";import{EffectComposer}from"three/examples/jsm/postprocessing/EffectComposer.js";import{RenderPass}from"three/examples/jsm/postprocessing/RenderPass.js";import{UnrealBloomPass}from"three/examples/jsm/postprocessing/UnrealBloomPass.js";
/* Flower engine from "TDF InteractiveFlower" (The Design Futurist), the Framer/React wrapper removed and three.js self-hosted. */
export const DEFAULTS={petalColor:"#3a64ff",innerGlowColor:"#ff82f3",stamenColor:"#ffcc44",stemColor:"#0a2a0a",backgroundColor:"#08080c",bloomStrength:50,petalCount:8,flowerScale:100,petalLength:50,stemLength:100};
export function mountFlower(container,props){const e=new FlowerEngine(container,{...DEFAULTS,...props});e.start(false);return()=>e.dispose()}
// ============================================================================
// ENGINE
// ============================================================================
class FlowerEngine{_onResize(){const w=this.container.clientWidth;const h=this.container.clientHeight;if(w===0||h===0)return;this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.renderer.setSize(w,h);this.composer.setSize(w,h);// Force a re-render on resize if in static mode (Canvas)
if(this.isStatic){this.renderFrame(this.clock.getElapsedTime(),new THREE.Vector3(0,0,0));}}_onMouseMove(e){const rect=this.container.getBoundingClientRect();this.mouse.x=(e.clientX-rect.left)/rect.width*2-1;this.mouse.y=-((e.clientY-rect.top)/rect.height)*2+1;}update(params,isStatic){this.isStatic=isStatic;this.bloomPass.strength=(params.bloomStrength??50)/200;this.renderer.setClearColor(params.backgroundColor,1);this.flower.updateParams(params);this.flower.setScale((params.flowerScale||100)/100);if(this.isStatic){this.renderFrame(0,new THREE.Vector3(0,0,0));}}renderFrame(t,hit){this.flower.animate(t,hit);const orbitRadius=8;this.camera.position.x=Math.sin(t*.12)*orbitRadius*.15;this.camera.position.z=orbitRadius;this.camera.position.y=2+Math.sin(t*.08)*.3;this.camera.lookAt(0,.5,0);this.composer.render();}start(isStatic){this.isStatic=isStatic;if(this.isStatic){this.renderFrame(0,new THREE.Vector3(0,0,0));return;}const tick=()=>{const t=this.clock.getElapsedTime();this.raycaster.setFromCamera(this.mouse,this.camera);const plane=new THREE.Plane(new THREE.Vector3(0,0,1),0);const hit=new THREE.Vector3;this.raycaster.ray.intersectPlane(plane,hit);this.renderFrame(t,hit);this.rafId=requestAnimationFrame(tick);};this.rafId=requestAnimationFrame(tick);}dispose(){cancelAnimationFrame(this.rafId);this.resizeObserver.disconnect();this.container.removeEventListener("mousemove",this._onMouseMove);this.renderer.dispose();this.flower.dispose();if(this.renderer.domElement.parentNode){this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);}}constructor(container,params){_define_property(this,"scene",void 0);_define_property(this,"camera",void 0);_define_property(this,"renderer",void 0);_define_property(this,"composer",void 0);_define_property(this,"bloomPass",void 0);_define_property(this,"flower",void 0);_define_property(this,"container",void 0);_define_property(this,"clock",void 0);_define_property(this,"rafId",0);_define_property(this,"mouse",new THREE.Vector2(-99,-99));_define_property(this,"raycaster",new THREE.Raycaster);_define_property(this,"resizeObserver",void 0);_define_property(this,"isStatic",false);this.container=container;this.clock=new THREE.Clock;this.scene=new THREE.Scene;this.camera=new THREE.PerspectiveCamera(40,container.clientWidth/container.clientHeight,.1,100);this.camera.position.set(0,2,8);this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});this.renderer.outputColorSpace=THREE.SRGBColorSpace;THREE.ColorManagement.enabled=true;this.renderer.setSize(container.clientWidth,container.clientHeight);this.renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.3;container.appendChild(this.renderer.domElement);this.composer=new EffectComposer(this.renderer);this.composer.addPass(new RenderPass(this.scene,this.camera));this.bloomPass=new UnrealBloomPass(new THREE.Vector2(container.clientWidth,container.clientHeight),(params.bloomStrength??50)/200,1,.1);this.composer.addPass(this.bloomPass);this.flower=new FlowerGroup(this.scene,params);this._onResize=this._onResize.bind(this);this._onMouseMove=this._onMouseMove.bind(this);this.resizeObserver=new ResizeObserver(()=>this._onResize());this.resizeObserver.observe(container);container.addEventListener("mousemove",this._onMouseMove);}}// ============================================================================
// FLOWER GROUP
// ============================================================================
class FlowerGroup{makeStem(color){const geo=new THREE.TubeGeometry(this.stemCurve,128,.06,20,false);const mat=new THREE.ShaderMaterial({uniforms:{uColor:{value:new THREE.Color(color)},uTime:{value:0}},vertexShader:STEM_VERT,fragmentShader:STEM_FRAG,transparent:true,side:THREE.DoubleSide,depthWrite:false});return new THREE.Mesh(geo,mat);}buildPetals(params){this.petals.forEach(p=>p.dispose());this.petals=[];const layers=4;const perLayer=params.petalCount||8;for(let layer=0;layer<layers;layer++){const countInLayer=perLayer-layer;const angleOffset=layer*Math.PI/perLayer;const tiltDeg=25+layer*18;const tiltRad=tiltDeg*Math.PI/180;const petalScale=.7+layer*.25;for(let i=0;i<countInLayer;i++){const yawAngle=i/countInLayer*Math.PI*2+angleOffset;const petal=new PetalUnit(this.headGroup,yawAngle,tiltRad,petalScale,layer,params);this.petals.push(petal);}}}animate(time,mouseWorld){this.petals.forEach(p=>p.animate(time,mouseWorld));this.stamen.animate(time);// Pass time to stem shader for GPU animation
if(this.stemMesh.material instanceof THREE.ShaderMaterial){this.stemMesh.material.uniforms.uTime.value=time;}// Gentle sway of the whole flower head
this.root.rotation.z=Math.sin(time*.4)*.025;this.root.rotation.x=Math.cos(time*.3)*.015;}updateParams(params){const newCount=params.petalCount||8;if(newCount!==this.currentPetalCount){this.currentPetalCount=newCount;this.buildPetals(params);}this.stamen.updateColor(params.stamenColor||params.innerGlowColor);this.petals.forEach(p=>p.updateProps(params));// Stem length logic
const sLen=(params.stemLength??100)/100;this.stemMesh.scale.y=sLen;// Anchor at bottom (-6.5). New Tip Y = bottom + (total height * scale)
const totalHeight=6.7// -6.5 to 0.2
;const newTipY=-6.5+totalHeight*sLen;this.headGroup.position.set(0,newTipY,0);if(this.stemMesh.material instanceof THREE.ShaderMaterial){const sc=params.stemColor||"#0a2a0a";this.stemMesh.material.uniforms.uColor.value.set(sc);}}setScale(s){this.root.scale.set(s,s,s);}dispose(){this.petals.forEach(p=>p.dispose());this.stamen.dispose();this.scene.remove(this.root);}constructor(scene,params){_define_property(this,"root",void 0);_define_property(this,"petals",[]);_define_property(this,"stamen",void 0);_define_property(this,"stemMesh",void 0);_define_property(this,"stemCurvePoints",void 0);_define_property(this,"stemCurve",void 0);_define_property(this,"scene",void 0);_define_property(this,"headGroup",void 0);_define_property(this,"currentPetalCount",8);this.scene=scene;this.currentPetalCount=params.petalCount||8;this.root=new THREE.Group;scene.add(this.root);// Head group for petals and stamen
this.headGroup=new THREE.Group;this.root.add(this.headGroup);// Stem with organic curve
const stemColor=params.stemColor||"#0a2a0a";this.stemCurvePoints=[new THREE.Vector3(0,-6.5,0),new THREE.Vector3(.1,-5,.05),new THREE.Vector3(-.05,-3.5,-.03),new THREE.Vector3(.08,-1.5,.02),new THREE.Vector3(0,.2,0)];this.stemCurve=new THREE.CatmullRomCurve3(this.stemCurvePoints);this.stemMesh=this.makeStem(stemColor);this.root.add(this.stemMesh);// Stamen (now in headGroup)
this.stamen=new StamenCluster(this.headGroup,params.stamenColor||params.innerGlowColor);// Build petals (now in headGroup)
this.buildPetals(params);// Initial scale setup
this.updateParams(params);}}// ============================================================================
// PETAL UNIT — Pivot-based architecture
//
// Structure:
//   pivotGroup (position = flower center = 0,0,0)
//     └─ mesh (geometry starts at y=0, extends upward to y=petalLength)
//
// The pivotGroup's rotation controls:
//   rotation.y = which compass direction the petal faces (yaw)
//   rotation.x = how far back the petal tilts (tilt/opening)
//
// This means ALL petals originate from one point,
// and their visual spread comes ONLY from their length + tilt.
// ============================================================================
class PetalUnit{animate(time,mouseWorld){this.material.uniforms.uTime.value=time;// --- 1. INDIVIDUAL TIP WORLD POSITION ---
// Transform petal tip from local → world to get each petal's unique position
const tipWorld=this.tipLocal.clone();this.mesh.localToWorld(tipWorld);// --- 2. DISTANCE & DIRECTION TO MOUSE ---
const dx=tipWorld.x-mouseWorld.x;const dy=tipWorld.y-mouseWorld.y;const dz=tipWorld.z-mouseWorld.z;const dist=Math.sqrt(dx*dx+dy*dy+dz*dz);// Gaussian influence: strong when close, fades with distance
const influence=Math.exp(-(dist*dist)/6);// Direction from mouse to petal tip (for directional push)
const pushX=dist>.01?dx/dist:0;const pushY=dist>.01?dy/dist:0;// --- 3. SPRING PHYSICS (Hooke's Law) on two axes ---
// Target displacement: push AWAY from mouse
const maxPush=.4// 눈에 확 띄게 강화
;const targetTilt=influence*pushY*maxPush;const targetTwist=influence*pushX*maxPush;const k=.12// Snappier spring
;const damp=.85// Slightly less damping for bounce
;this.velTilt+=(targetTilt-this.springTilt)*k;this.velTilt*=damp;this.springTilt+=this.velTilt;this.velTwist+=(targetTwist-this.springTwist)*k;this.velTwist*=damp;this.springTwist+=this.velTwist;// --- 4. APPLY AS ROTATION (breeze-like) ---
const windSway=Math.sin(time*.7+this.phase)*.03;this.pivot.rotation.x=-this.baseTilt+windSway+this.springTilt;this.pivot.rotation.z=this.springTwist// Gentle sideways lean
;// Pass a subtle value to shader for tip softening
this.material.uniforms.uSpring.value=influence*.3;}updateProps(params){this.material.uniforms.uColor.value.set(params.petalColor);this.material.uniforms.uInnerColor.value.set(params.innerGlowColor);const val=params.petalLength??50;const L=.5+val/100;this.mesh.scale.y=L;this.tipLocal.set(0,this.petalHeight*L,0);}dispose(){this.mesh.geometry.dispose();this.material.dispose();this.pivot.parent?.remove(this.pivot);}constructor(parent,yawAngle,tiltRad,scale,layer,params){_define_property(this,"mesh",void 0);_define_property(this,"pivot",void 0);_define_property(this,"material",void 0);_define_property(this,"phase",void 0);_define_property(this,"baseTilt",void 0);_define_property(this,"baseYaw",void 0);_define_property(this,"petalHeight",void 0);// Spring physics — two axes for organic feel
_define_property(this,"springTilt",0);_define_property(this,"springTwist",0);_define_property(this,"velTilt",0);_define_property(this,"velTwist",0);// Cached tip offset (local space direction of petal tip)
_define_property(this,"tipLocal",void 0);this.phase=Math.random()*Math.PI*2;this.baseTilt=tiltRad;this.baseYaw=yawAngle;// --- PIVOT at flower center ---
this.pivot=new THREE.Group;this.pivot.position.set(0,0,0);this.pivot.rotation.order="YXZ";this.pivot.rotation.y=yawAngle;this.pivot.rotation.x=-tiltRad;parent.add(this.pivot);// --- PETAL GEOMETRY ---
const petalWidth=1*scale;this.petalHeight=3.5*scale;const geo=new THREE.PlaneGeometry(petalWidth,this.petalHeight,24,48);geo.translate(0,this.petalHeight/2,0);// Tip position in local petal space (top center of the petal)
this.tipLocal=new THREE.Vector3(0,this.petalHeight,0);this.material=new THREE.ShaderMaterial({uniforms:{uColor:{value:new THREE.Color(params.petalColor)},uInnerColor:{value:new THREE.Color(params.innerGlowColor)},uTime:{value:0},uLayer:{value:layer},uSpring:{value:0}},vertexShader:PETAL_VERT,fragmentShader:PETAL_FRAG,transparent:true,side:THREE.DoubleSide,blending:THREE.AdditiveBlending,depthWrite:false});this.mesh=new THREE.Mesh(geo,this.material);this.pivot.add(this.mesh);}}// ============================================================================
// GLSL SHADERS
// ============================================================================
const PETAL_VERT=`
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewDir;
    uniform float uTime;
    uniform float uLayer;
    uniform float uSpring;

    void main() {
        vUv = uv;
        vec3 pos = position;

        // --- Horizontal curvature: wrap petal around Y axis ---
        // Creates a concave shape that hugs the stamen
        float curvature = 1.2 - float(uLayer) * 0.15;
        pos.z += pos.x * pos.x * curvature;

        // --- Vertical curvature: petal tip curls slightly outward ---
        pos.z += pow(vUv.y, 3.0) * 0.4;

        // --- Spring interaction: pushes the top of the petal outward ---
        pos.z += uSpring * smoothstep(0.1, 0.9, vUv.y);

        // --- Organic micro-sway ---
        pos.x += sin(uTime * 1.2 + pos.y * 2.0) * 0.02 * vUv.y;

        vec4 worldPos = modelMatrix * vec4(pos, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
`;const PETAL_FRAG=`
    uniform vec3 uColor;
    uniform vec3 uInnerColor;
    uniform float uTime;
    uniform float uLayer;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewDir;

    // SDF for petal silhouette clipping
    float petalSDF(vec2 p) {
        p.x = abs(p.x);
        float d1 = length(p - vec2(0.55, 0.45)) - 0.72;
        float d2 = length(p - vec2(-0.55, 0.45)) - 0.72;
        return max(d1, d2);
    }

    // Value noise
    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }
    float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(
            mix(hash(i), hash(i + vec2(1,0)), u.x),
            mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
            u.y
        );
    }
    float fbm(vec2 p) {
        float v = 0.0, a = 0.5;
        for (int i = 0; i < 4; i++) {
            v += a * noise(p);
            p *= 2.0;
            a *= 0.5;
        }
        return v;
    }

    void main() {
        // SDF clipping
        float d = petalSDF(vUv - vec2(0.5, 0.0));
        if (d > 0.0) discard;

        // Domain-warped FBM veins (radial from petal base)
        vec2 fromBase = vUv - vec2(0.5, 0.0);
        float r = length(fromBase);
        float theta = atan(fromBase.y, fromBase.x);
        vec2 polar = vec2(theta * 8.0, r * 6.0);
        float vein = fbm(polar + 0.5 * fbm(polar * 1.5));
        vein = pow(vein, 2.0) * 0.6;

        // Fresnel X-ray glow
        float dotNV = dot(vNormal, vViewDir);
        float fresnel = pow(1.0 - abs(dotNV), 2.5);
        float xray = (1.0 - abs(dotNV)) * 1.2;

        // Gradient: warm at base, cool at tip
        vec3 col = mix(uInnerColor * 2.5, uColor, pow(vUv.y, 0.65));
        col += vein * uInnerColor * 0.4;
        col *= (fresnel * 3.0 + xray * 0.6 + 0.25);

        // Alpha: edge glow + SDF soft edge + base fade
        float alpha = (fresnel * 1.6 + xray * 0.4);
        alpha *= smoothstep(0.0, -0.04, d); // SDF soft edge
        alpha *= smoothstep(0.0, 0.12, vUv.y); // Base transparency

        gl_FragColor = vec4(col, alpha);
    }
`;const STEM_VERT=`
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewDir;
    varying float vY;
    uniform float uTime;

    void main() {
        vUv = uv;
        vec3 pos = position;
        
        // Organic swaying in vertex shader (non-rebuilding)
        float sway = sin(uTime * 0.8 + pos.y * 0.5) * 0.15;
        float swayZ = cos(uTime * 0.6 + pos.y * 0.4) * 0.1;
        
        // Influence increases at the top, zero at the very base
        float influence = smoothstep(-6.5, 0.0, pos.y);
        pos.x += sway * influence;
        pos.z += swayZ * influence;

        vY = pos.y;
        vec4 worldPos = modelMatrix * vec4(pos, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
`;const STEM_FRAG=`
    uniform vec3 uColor;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewDir;
    varying float vY;

    void main() {
        // Fresnel for soft edges
        float dotNV = dot(vNormal, vViewDir);
        float fresnel = pow(1.0 - abs(dotNV), 2.0);
        
        // Cylinder lighting
        float light = dot(vNormal, normalize(vec3(1.0, 1.0, 1.0))) * 0.5 + 0.5;
        
        // Fade at top
        float fade = smoothstep(0.2, -0.2, vY);
        
        vec3 col = uColor * (light * 1.5 + fresnel * 0.8);
        
        // Soft alpha at edges helps reduce pixelation
        float alpha = (0.4 + fresnel * 0.6) * fade;
        
        gl_FragColor = vec4(col, alpha);
    }
`;// ============================================================================
// STAMEN CLUSTER
// ============================================================================
class StamenCluster{animate(time){this.fibers.forEach((f,i)=>{f.scale.y=1+Math.sin(time*4+i*.08)*.35;});}updateColor(color){this.fibers.forEach(f=>f.material.color.set(color));}dispose(){if(this.fibers.length>0){this.fibers[0].geometry.dispose();this.fibers.forEach(f=>f.material.dispose());}this.group.parent?.remove(this.group);}constructor(parent,color){_define_property(this,"group",void 0);_define_property(this,"fibers",[]);this.group=new THREE.Group;this.group.position.y=.1;parent.add(this.group);const count=200;const geo=new THREE.CapsuleGeometry(.006,.9,4,8);for(let i=0;i<count;i++){const mat=new THREE.MeshBasicMaterial({color:color,transparent:true,opacity:1,blending:THREE.AdditiveBlending});const m=new THREE.Mesh(geo,mat);const a=i*2.39996// golden angle
;const r=Math.sqrt(i/count)*.35;m.position.set(Math.cos(a)*r,Math.random()*.25,Math.sin(a)*r);m.rotation.set((Math.random()-.5)*.3,a,(Math.random()-.5)*.3);this.fibers.push(m);this.group.add(m);}}}// ============================================================================
// FRAMER CONTROLS
// ============================================================================
