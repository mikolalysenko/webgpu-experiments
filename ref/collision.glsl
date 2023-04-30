// Contact detection code by Mikola Lysenko
//
//
// Rendering code based on "RayMarching starting point" https://www.shadertoy.com/view/WtGXDD
//      Martijn Steinrucken aka The Art of Code/BigWings - 2020


#define CONTACT_ITERS 10

#define MAX_STEPS 100
#define MAX_DIST 100.
#define SURF_DIST .001
#define TAU 6.283185
#define PI 3.141592

// animation state

bool hit = false;    
vec3 qA = vec3(0);
vec3 qB = vec3(0);
vec3 minHit = vec3(0.);

// body a state
vec3 posA = vec3(0., 0., 0.);     // position
vec4 rotA = vec4(0., 0., 0., 1.); // quaternion
float mInvA = 0.;                 // inverse mass
mat3 rInvA = mat3(                // inverse inertia tensor
    0., 0., 0.,
    0., 0., 0.,
    0., 0., 0.);

// a position shift
vec3 linA = vec3(0., 0., 0.);     // relative translation update
vec4 torA = vec4(0., 0., 0., 1.); // relative rotation update (multiplied)


// body b state
vec3 posB = vec3(0., 0., 0.);      // position
vec4 rotB = vec4(0., 0., 0., 1.);  // quaternion
float mInvB = 1.;                  // inverse mass
mat3 rInvB = mat3(                 // inverse inertia tensor
    1., 0., 0.,
    0., 1., 0.,
    0., 0., 1.);

// b position shift
vec3 linB = vec3(0., 0., 0.);
vec4 torB = vec4(0., 0., 0., 1.);


// quaternion subroutines
vec4 quatMult(vec4 q1, vec4 q2) {
    vec3 crossProduct = cross(q1.xyz, q2.xyz);
    float dotProduct = dot(q1.xyz, q2.xyz);
    return vec4(crossProduct + q1.w * q2.xyz + q2.w * q1.xyz, q1.w * q2.w - dotProduct);
}
vec4 quatConjugate(vec4 q) {
    return vec4(-q.xyz, q.w);
}
vec3 quatTransformVec(vec4 quat, vec3 vec) {
    return quatMult(quatMult(quat, vec4(vec, 0.0)), quatConjugate(quat)).xyz;
}
vec4 axisAngleToQuat(vec3 axis, float angle) {
    float halfAngle = angle * 0.5;
    float sinHalfAngle = sin(halfAngle);
    vec3 axisNormalized = normalize(axis);
    return vec4(axisNormalized * sinHalfAngle, cos(halfAngle));
}

// position transformations
vec3 transformPosBoost(vec3 x, vec3 lin, vec4 tor, vec3 pos, vec4 rot) {
    return quatTransformVec(quatMult(tor, rot), x - lin - pos);
}
vec3 invTransformPosBoost(vec3 x, vec3 lin, vec4 tor, vec3 pos, vec4 rot) {
    return quatTransformVec(quatConjugate(quatMult(tor, rot)), x) + lin + pos;
}
vec3 transformPos (vec3 x, vec3 pos, vec4 rot) {
    return quatTransformVec(rot, x - pos);
}
vec3 transformGrad (vec3 v, vec4 tor, vec4 rot) {
    return quatTransformVec(quatConjugate(quatMult(tor, rot)), v);
}
vec3 transformA (vec3 x)      { return transformPos(x, posA, rotA); }
vec3 transformABoost (vec3 x) { return transformPosBoost(x, linA, torA, posA, rotA); }
vec3 invTransformABoost (vec3 x) { return invTransformPosBoost(x, linA, torA, posA, rotA); }
vec3 transformAGrad (vec3 x)  { return transformGrad(x, torA, rotA); }
vec3 transformB (vec3 x)      { return transformPos(x, posB, rotB); }
vec3 transformBBoost (vec3 x) { return transformPosBoost(x, linB, torB, posB, rotB); }
vec3 invTransformBBoost (vec3 x) { return invTransformPosBoost(x, linB, torB, posB, rotB); }
vec3 transformBGrad (vec3 v) { return transformGrad(v, torB, rotB); }


// standard sdf stuff
float sdSphere( vec3 p, vec3 c, float s ) {
  return length(p - c)-s;
}
float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}
float sdPlane (vec3 p) {
    return p.y;
}
float sdBox( vec3 p, vec3 b ) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// shape signed distance functions
float sdA (vec3 p) {
    // return sdBox(p, vec3(1.));
    return sdTorus(p, vec2(1.2, 0.35));
    // return sdPlane(p + 0.5);
}
float sdB (vec3 p) {
    // return sdBox(p, vec3(1.));
    return sdTorus(p, vec2(1.2, 0.35));
    // return sdSphere(p, vec3(0.), 0.3);
}

// boilerplate: gradients for shape sdfs
vec3 gradA (vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = sdA(p) - 
        vec3(sdA(p-e.xyy), sdA(p-e.yxy),sdA(p-e.yyx));    
    return normalize(n);
}
vec3 gradB (vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = sdB(p) - 
        vec3(sdB(p-e.xyy), sdB(p-e.yxy),sdB(p-e.yyx));    
    return normalize(n);
}

//
// begin actually interesting code
//
vec3 projIntersect (vec3 p, float depth) {
    const vec2 e = vec2(.001, 0);
    for (int i = 0; i < 6; ++i) {
        vec3 pa = transformABoost(p);
        vec3 pb = transformBBoost(p);        
        float fa = sdA(pa);
        float fb = sdB(pb);
        if (max(fa, fb) < -depth) {
            break;
        }
        if (fa > fb) {
            vec3 n = fa - vec3(sdA(pa-e.xyy), sdA(pa-e.yxy), sdA(pa-e.yyx));    
            p = invTransformABoost(pa - (fa + depth) * normalize(n));
        } else {
            vec3 n = fb - vec3(sdB(pb-e.xyy), sdB(pb-e.yxy), sdB(pb-e.yyx));
            p = invTransformBBoost(pb - (fb + depth) * normalize(n));
        }
    }
    return p;
}
vec3 findMin(vec3 p) {
    p = projIntersect(p, 0.);
    vec3 minPos = p;
    float minDepth = 0.;
    float lo = 0.;
    float hi = 1.;
    for (int i = 0; i < 8; ++i) {
        float testDepth = 0.5 * (lo + hi);
        p = projIntersect(p, testDepth);
        float d = max(sdA(transformABoost(p)), sdB(transformBBoost(p)));
        if (d < minDepth) {
            minDepth = d;
            minPos = p;
            lo = testDepth;
        } else {
            hi = testDepth;
        }
    }
    return minPos;
}
float updateContact (vec3 hit, vec3 norm, float d) {
    vec4 qa = quatMult(torA, rotA);
    vec4 qb = quatMult(torB, rotB);
    
    vec3 drotA = cross(hit - posA, norm);
    vec3 drotB = cross(hit - posB, norm);
    
    vec3 wrotA = rInvA * drotA;
    vec3 wrotB = rInvB * drotB;
   
    // calculate effective mass and impulse
    float w1 = mInvA + dot(drotA, wrotA);
    float w2 = mInvB + dot(drotB, wrotB);
    float impulse = -2. * d / max(0.0001, w1 + w2);
    
    // update linear velocity
    linA += impulse * mInvA * norm;
    linB -= impulse * mInvB * norm;
    
    // update rotational velocity
    qa += quatMult(vec4(0.5 * impulse * wrotA, 0.), qa);
    torA = normalize(quatMult(qa, quatConjugate(rotA)));    
    qb += quatMult(vec4(0.5 * impulse * wrotB, 0.), qb);
    torB = normalize(quatMult(qb, quatConjugate(rotB)));
    
    return impulse;
}
float solveContact () {
    float lambda = 0.;
    for (int i = 0; i < 5; ++i) {
        vec3 hitPos = 0.5 * (posA + linA + posB + linB);
        float dlambda = 0.;
        for (int outerIter = 0; outerIter < 5; ++outerIter) {
            hitPos = findMin(hitPos);
            
            vec3 ta = transformABoost(hitPos);
            vec3 tb = transformBBoost(hitPos);
            float fa = sdA(ta);
            float fb = sdB(tb);
            float rad = max(fa, fb);
            if (rad > 0.) {
                break;
            }
            hit = true;
            
            // calculate hit normal
            vec3 da = transformAGrad(gradA(ta));
            vec3 db = transformBGrad(gradB(tb));
            vec3 hitNorm = normalize(fa * da - fb * db);
            
            dlambda += updateContact(hitPos, hitNorm, rad);
            if (abs(dlambda) < 0.01) {
                break;
            }
       }
       lambda += dlambda;
       if (abs(dlambda) < 0.01) {
           break;
       }
   }
   
   minHit = findMin(0.5 * (posA + posB + linA + linB));
   qA = minHit + linA;
   qB = minHit + linB;
   
   return lambda;
}
//
// end actually interesting code
//


// --- scene stuff ----
// debug lines
float GetDistSolid(vec3 p) {
    float d = 10000.;
    // if (hit) {
        d = min(d, sdSphere(p, qA, 0.05));
        d = min(d, sdCapsule(p, qA, minHit, 0.005));
        d = min(d, sdCapsule(p, qB, minHit, 0.005));
        d = min(d, sdSphere(p, minHit, 0.07));
    // }
    return d;
}

float GetDistAlphaC(vec3 p) {
    return max(sdA(transformA(p)), sdB(transformB(p)));
}

float GetDistAlpha(vec3 p) {
    return min(sdA(transformABoost(p)), sdB(transformBBoost(p)));
}
// --- end scene stuff ----


// ray marching boilerplate
float RayMarchSolid(vec3 ro, vec3 rd) {
	float dO=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDistSolid(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}
float RayMarchAlphaC(vec3 ro, vec3 rd) {
	float dO=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDistAlphaC(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}
float RayMarchAlpha(vec3 ro, vec3 rd) {
	float dO=0.;
    for(int i=0; i<MAX_STEPS; i++) {
    	vec3 p = ro + rd*dO;
        float dS = GetDistAlpha(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST) break;
    }
    return dO;
}

// surface normal boilerplate
vec3 GetNormalSolid(vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = GetDistSolid(p) - 
        vec3(GetDistSolid(p-e.xyy), GetDistSolid(p-e.yxy),GetDistSolid(p-e.yyx));    
    return normalize(n);
}
vec3 GetNormalAlphaC(vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = GetDistAlphaC(p) - 
        vec3(GetDistAlphaC(p-e.xyy), GetDistAlphaC(p-e.yxy),GetDistAlphaC(p-e.yyx));
    
    return normalize(n);
}
vec3 GetNormalAlpha(vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = GetDistAlpha(p) - 
        vec3(GetDistAlpha(p-e.xyy), GetDistAlpha(p-e.yxy),GetDistAlpha(p-e.yyx));
    
    return normalize(n);
}

// camera
vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 
        f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u;
    return normalize(i);
}
mat2 Rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // update animation
    float tt = 0.1 * iTime - 0.02;
    posB = vec3(3. * cos(tt), 0., 0.);
    
    rotB = axisAngleToQuat(vec3(1, 0, 0), 3. * tt);
    // rotA = axisAngleToQuat(vec3(0, 1, 0), 10. * tt);
    /*
    rotB = 
        quatMult(
            axisAngleToQuat(vec3(1., 0., 0.), 0.5 * tt),
            axisAngleToQuat(vec3(0., 1., 0.), 0.13 * tt));
     */
     
    // solve for intersections
    solveContact();
    
    // set up view direction
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
	vec2 m = iMouse.xy/iResolution.xy;
    vec3 ro = vec3(0, 3, -3);
    ro.yz *= Rot(-m.y*PI+1.);
    ro.xz *= Rot(-m.x*TAU);    
    vec3 rd = GetRayDir(uv, ro, vec3(0,0.,0), 1.);
    
    // ray trace
    vec3 col = vec3(0);
    
    // trace solids
    float dSolid = RayMarchSolid(ro, rd);
    if (dSolid < MAX_DIST) {
        col = vec3(2., 0., 0.);
    } else {
        dSolid = MAX_DIST;
    }
    
    // trace alpha layer 0
    float dAlpha0 = RayMarchAlphaC(ro, rd);
    if(dAlpha0<dSolid) {
        vec3 p = ro + rd * dAlpha0;
        vec3 n = GetNormalAlphaC(p);
        vec3 r = reflect(rd, n);
        float dif = dot(n, normalize(vec3(1,2,3)))*.5+.5;
        if (hit) {
            col = mix(col, dif * vec3(1., 0.5, 0.2), 0.4);
        } else {
            col = mix(col, dif * vec3(0., 1., 1.), 0.3);
        }
    }
    
    // trace alpha layer 1
    float dAlpha1 = RayMarchAlpha(ro, rd);
    if(dAlpha1<dSolid) {
        vec3 p = ro + rd * dAlpha1;
        vec3 n = GetNormalAlpha(p);
        vec3 r = reflect(rd, n);
        float dif = dot(n, normalize(vec3(1,2,3)))*.5+.5;
        col = mix(col, vec3(dif), 0.2);
    }
    
    // gamma correct and output
    col = pow(col, vec3(.4545));
    fragColor = vec4(col,1.0);
}