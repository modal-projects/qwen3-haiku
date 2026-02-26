(function () {
    const treeCanvas = document.getElementById('tree-canvas');
    if (!treeCanvas) {
        return;
    }
    const treeCtx = treeCanvas.getContext('2d');
    const branchTips = [];
    const TREE_SCALE_INITIAL = 1.75;
    const TREE_SCALE_SHRUNK = 1;
    let treeRenderScale = TREE_SCALE_INITIAL;

    const COS30 = Math.cos(Math.PI / 6);
    const SIN30 = Math.sin(Math.PI / 6);

    function isoProject(x, y, z) {
        return {
            sx: (x - y) * COS30,
            sy: (x + y) * SIN30 - z,
        };
    }

    const rotationState = {
        yaw: 0,
        pitch: 0.35,
        yawVelocity: 0,
        pitchVelocity: 0,
        dragging: false,
        activePointerId: null,
        lastX: 0,
        lastY: 0,
        lastT: 0,
        lastFrameTs: 0,
        autoRotateEnabled: false,
        boostUntil: 0,
        frictionPer60: 0.93,
        maxPitch: 1.0,
        pixelsToRadians: 0.0065,
        spinBoost: 0.03,
        cruiseYawSpeed: 0.0026,
    };

    function triggerGenerateSpin() {
        const now = performance.now();
        rotationState.autoRotateEnabled = true;
        rotationState.boostUntil = now + 1800;
        rotationState.yawVelocity = Math.max(rotationState.yawVelocity, rotationState.spinBoost);
    }

    function rotateZ(point, angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return {
            x: point.x * c - point.y * s,
            y: point.x * s + point.y * c,
            z: point.z,
        };
    }

    function rotateX(point, angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return {
            x: point.x,
            y: point.y * c - point.z * s,
            z: point.y * s + point.z * c,
        };
    }

    function worldToView(point) {
        const zRot = rotateZ(point, rotationState.yaw);
        return rotateX(zRot, rotationState.pitch);
    }

    function projectWorld(point) {
        const v = worldToView(point);
        return isoProject(v.x, v.y, v.z);
    }

    const BLOSSOM_COLORS = ['#a6dfbf', '#bfe6c2', '#d9e4ba', '#f2d3c6', '#ffbfd8'];
    const TRUNK_BASE = ['#4a6633', '#3d5529', '#2e4020'];
    const TIP_CLR = ['#90b87a', '#7da066', '#6a8855'];

    let treeBranches = [];
    let treeNodes = [];
    let treeGenerated = false;
    let growthStart = 0;
    const MAX_DEPTH = 7;

    function lerpHex(a, b, t) {
        const ah = parseInt(a.slice(1), 16);
        const bh = parseInt(b.slice(1), 16);
        const r = Math.round(((ah >> 16) & 0xff) + (((bh >> 16) & 0xff) - ((ah >> 16) & 0xff)) * t);
        const g = Math.round(((ah >> 8) & 0xff) + (((bh >> 8) & 0xff) - ((ah >> 8) & 0xff)) * t);
        const bl = Math.round((ah & 0xff) + ((bh & 0xff) - (ah & 0xff)) * t);
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
    }

    function adjustHexHsl(hex, hueShiftDeg, satScale, lightDelta) {
        const value = parseInt(hex.slice(1), 16);
        let r = ((value >> 16) & 255) / 255;
        let g = ((value >> 8) & 255) / 255;
        let b = (value & 255) / 255;

        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const l = (max + min) * 0.5;
        const d = max - min;

        let h = 0;
        let s = 0;
        if (d > 1e-6) {
            s = d / (1 - Math.abs(2 * l - 1));
            if (max === r) {
                h = ((g - b) / d) % 6;
            } else if (max === g) {
                h = (b - r) / d + 2;
            } else {
                h = (r - g) / d + 4;
            }
            h *= 60;
            if (h < 0) {
                h += 360;
            }
        }

        h = (h + hueShiftDeg + 360) % 360;
        s = Math.max(0, Math.min(1, s * satScale));
        const nl = Math.max(0, Math.min(1, l + lightDelta));

        const c = (1 - Math.abs(2 * nl - 1)) * s;
        const hp = h / 60;
        const x = c * (1 - Math.abs((hp % 2) - 1));
        let rr = 0;
        let gg = 0;
        let bb = 0;
        if (hp < 1) {
            rr = c;
            gg = x;
        } else if (hp < 2) {
            rr = x;
            gg = c;
        } else if (hp < 3) {
            gg = c;
            bb = x;
        } else if (hp < 4) {
            gg = x;
            bb = c;
        } else if (hp < 5) {
            rr = x;
            bb = c;
        } else {
            rr = c;
            bb = x;
        }

        const m = nl - c * 0.5;
        r = Math.round((rr + m) * 255);
        g = Math.round((gg + m) * 255);
        b = Math.round((bb + m) * 255);
        return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
    }

    function faceColors(depth) {
        const t = Math.min(1, depth / MAX_DEPTH);
        return [
            lerpHex(TRUNK_BASE[0], TIP_CLR[0], t),
            lerpHex(TRUNK_BASE[1], TIP_CLR[1], t),
            lerpHex(TRUNK_BASE[2], TIP_CLR[2], t),
        ];
    }

    function thickness(depth) {
        if (depth === 0) return 20;
        if (depth === 1) return 14;
        if (depth === 2) return 9.5;
        return Math.max(1.2, 7.2 * Math.pow(0.68, depth - 2));
    }

    function segLength(depth) {
        if (depth === 0) return 160;
        if (depth === 1) return 110;
        if (depth === 2) return 90;
        return Math.max(20, 62 - depth * 5);
    }

    function makeDir(yaw, tilt) {
        const h = Math.sin(tilt);
        const v = Math.cos(tilt);
        return { x: Math.cos(yaw) * h, y: Math.sin(yaw) * h, z: v };
    }

    function addSeg(depth, start, dir, len, thick) {
        const jit = 1 + (Math.random() - 0.5) * 0.12;
        const L = len * jit;
        const end = { x: start.x + dir.x * L, y: start.y + dir.y * L, z: start.z + dir.z * L };
        treeBranches.push({
            start: { ...start },
            end: { ...end },
            depth,
            thickness: thick,
            faceColors: faceColors(depth),
            swayPhase: Math.random() * Math.PI * 2,
            swaySpeed: 0.25 + Math.random() * 0.35,
        });
        treeNodes.push({ pos: { ...end }, depth, glowPhase: Math.random() * Math.PI * 2 });
        return end;
    }

    function growCanopy(depth, pos, yaw, maxD, vigor) {
        if (depth > maxD) {
            branchTips.push({ ...pos });
            return;
        }
        const lenScale = 0.7 + vigor * 0.6 + (Math.random() - 0.5) * 0.3;
        const len = segLength(depth) * lenScale;
        const thick = thickness(depth) * (0.8 + vigor * 0.4);
        const tilt = 0.42 + depth * 0.09 + (Math.random() - 0.5) * 0.25;
        const dir = makeDir(yaw + (Math.random() - 0.5) * 0.15, tilt);
        const end = addSeg(depth, pos, dir, len, thick);

        if (depth >= maxD) {
            branchTips.push({ ...end });
            return;
        }

        if (Math.random() < 0.08 && depth > 3) {
            branchTips.push({ ...end });
            return;
        }

        const nKids = vigor > 0.6 ? 3 : (Math.random() < 0.45 ? 3 : 2);
        const spread = 0.95 + depth * 0.16 + (Math.random() - 0.5) * 0.24;
        for (let i = 0; i < nKids; i++) {
            const frac = nKids === 1 ? 0 : i / (nKids - 1) - 0.5;
            const kidYaw = yaw + frac * spread + (Math.random() - 0.5) * 0.4;
            const kidVigor = vigor * (0.55 + Math.random() * 0.45);
            growCanopy(depth + 1, end, kidYaw, maxD, kidVigor);
        }
    }

    function buildTree() {
        treeBranches = [];
        treeNodes = [];
        branchTips.length = 0;

        const p0 = { x: 0, y: 0, z: 0 };
        const trunkLean = { x: 0.03, y: -0.02, z: 0.999 };
        const p1 = addSeg(0, p0, { x: 0, y: 0, z: 1 }, segLength(0), thickness(0));
        const p2 = addSeg(1, p1, trunkLean, segLength(1), thickness(1));

        const mains = [
            { yaw: 0.4, tilt: 0.66, vigor: 1.0, lenMul: 1.12 },
            { yaw: 1.8, tilt: 0.58, vigor: 0.85, lenMul: 1.02 },
            { yaw: 3.3, tilt: 0.7, vigor: 0.7, lenMul: 0.96 },
            { yaw: 5.0, tilt: 0.78, vigor: 0.6, lenMul: 0.88 },
        ];
        for (const m of mains) {
            const yaw = m.yaw + (Math.random() - 0.5) * 0.5;
            const tilt = m.tilt + (Math.random() - 0.5) * 0.15;
            const dir = makeDir(yaw, tilt);
            const len = segLength(2) * m.lenMul * (0.85 + Math.random() * 0.3);
            const bs = addSeg(2, p2, dir, len, thickness(2) * (0.7 + m.vigor * 0.3));
            growCanopy(3, bs, yaw, MAX_DEPTH, m.vigor);
        }

        const topYaw = 2.0 + (Math.random() - 0.5) * 0.6;
        const topDir = makeDir(topYaw, 0.3 + Math.random() * 0.12);
        const topS = addSeg(2, p2, topDir, segLength(2) * 0.9, thickness(2) * 0.9);
        growCanopy(3, topS, topYaw, MAX_DEPTH, 0.9);

        const lowYaw = 4.2 + (Math.random() - 0.5) * 0.8;
        const lowDir = makeDir(lowYaw, 0.7 + Math.random() * 0.2);
        const lowS = addSeg(2, p1, lowDir, segLength(2) * 0.6, thickness(2) * 0.7);
        growCanopy(3, lowS, lowYaw, MAX_DEPTH - 1, 0.5);

        const low2Yaw = 1.0 + (Math.random() - 0.5) * 0.6;
        const low2Dir = makeDir(low2Yaw, 0.55 + Math.random() * 0.2);
        const low2S = addSeg(2, p1, low2Dir, segLength(2) * 0.5, thickness(2) * 0.65);
        growCanopy(3, low2S, low2Yaw, MAX_DEPTH - 2, 0.35);

        treeBranches.sort((a, b) => {
            const dA = (a.start.x + a.start.y + a.end.x + a.end.y) * 0.25 - (a.start.z + a.end.z) * 0.4;
            const dB = (b.start.x + b.start.y + b.end.x + b.end.y) * 0.25 - (b.start.z + b.end.z) * 0.4;
            return dA - dB;
        });
        treeGenerated = true;
        growthStart = performance.now();
    }

    function getAnimatedSegment(branch, t, swayOffsets) {
        const growT = Math.min(1, Math.max(0, (t - branch.depth * 280) / 500));
        if (growT <= 0) return null;

        const start = branch.start;
        let end = {
            x: start.x + (branch.end.x - start.x) * growT,
            y: start.y + (branch.end.y - start.y) * growT,
            z: start.z + (branch.end.z - start.z) * growT,
        };

        if (branch.depth >= MAX_DEPTH - 3 && swayOffsets) {
            const sway = Math.sin(swayOffsets.time * branch.swaySpeed + branch.swayPhase) * 0.025 * (branch.depth - (MAX_DEPTH - 4));
            end = { x: end.x + sway * 8, y: end.y + sway * 6, z: end.z };
        }

        return { start, end, growT };
    }

    function rotatedDepthKey(segment) {
        const mid = {
            x: (segment.start.x + segment.end.x) * 0.5,
            y: (segment.start.y + segment.end.y) * 0.5,
            z: (segment.start.z + segment.end.z) * 0.5,
        };
        const v = worldToView(mid);
        return (v.x + v.y) * 0.5 - v.z * 0.8;
    }

    function drawPrism(ctx, branch, segment) {
        const start = segment.start;
        const end = segment.end;
        const wBot = branch.thickness * 0.5;
        const taper = branch.depth <= 1 ? 0.9 : (branch.depth === 2 ? 0.82 : 0.72);
        const wTop = wBot * taper;

        const bOff = [{ x: -wBot, y: -wBot }, { x: wBot, y: -wBot }, { x: wBot, y: wBot }, { x: -wBot, y: wBot }];
        const tOff = [{ x: -wTop, y: -wTop }, { x: wTop, y: -wTop }, { x: wTop, y: wTop }, { x: -wTop, y: wTop }];

        const bV = bOff.map(o => projectWorld({ x: start.x + o.x, y: start.y + o.y, z: start.z }));
        const tV = tOff.map(o => projectWorld({ x: end.x + o.x, y: end.y + o.y, z: end.z }));

        ctx.beginPath();
        ctx.moveTo(tV[0].sx, tV[0].sy);
        for (let i = 1; i < 4; i++) ctx.lineTo(tV[i].sx, tV[i].sy);
        ctx.closePath();
        ctx.fillStyle = branch.faceColors[0];
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(tV[1].sx, tV[1].sy);
        ctx.lineTo(tV[2].sx, tV[2].sy);
        ctx.lineTo(bV[2].sx, bV[2].sy);
        ctx.lineTo(bV[1].sx, bV[1].sy);
        ctx.closePath();
        ctx.fillStyle = branch.faceColors[1];
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(tV[0].sx, tV[0].sy);
        ctx.lineTo(tV[3].sx, tV[3].sy);
        ctx.lineTo(bV[3].sx, bV[3].sy);
        ctx.lineTo(bV[0].sx, bV[0].sy);
        ctx.closePath();
        ctx.fillStyle = branch.faceColors[2];
        ctx.fill();
    }

    function drawLeaf(ctx, pos, bloom, nowMs) {
        const p = projectWorld(pos);
        const seed = Math.abs(Math.floor(pos.x * 1.7 + pos.y * 2.3 + pos.z * 3.1));
        const petalColor = BLOSSOM_COLORS[seed % BLOSSOM_COLORS.length];
        const blossomHueShift = (((seed >> 5) % 9) - 4) * 3.5;
        const blossomSatScale = 0.78 + (seed % 11) * 0.038;
        const blossomLightDelta = (((seed >> 2) % 11) - 5) * 0.024;
        const blossomBase = adjustHexHsl(petalColor, blossomHueShift, blossomSatScale, blossomLightDelta);
        const baseR = 3.9 + (seed % 5) * 0.32;
        const phase = (seed % 628) / 100;
        const swayX = Math.sin(nowMs * 0.0022 + phase) * 0.8 * bloom;
        const swayY = Math.cos(nowMs * 0.0017 + phase * 1.21) * 0.45 * bloom;
        const pulse = 1 + 0.08 * Math.sin(nowMs * 0.003 + phase * 1.6) * bloom;
        const scale = (0.15 + 0.85 * bloom) * pulse;
        const toneLight = adjustHexHsl(blossomBase, -2, 1.02, 0.14);
        const toneLighter = adjustHexHsl(blossomBase, 0, 1.06, 0.23);
        const toneWarm = adjustHexHsl(blossomBase, -8, 1.0, 0.08);
        const toneDark = adjustHexHsl(blossomBase, 4, 0.9, -0.2);
        ctx.save();
        ctx.translate(p.sx + swayX, p.sy + swayY);
        ctx.scale(scale, scale);
        const blossomAlpha = 0.32 + 0.68 * bloom;
        ctx.globalAlpha = blossomAlpha;
        const petalCount = seed % 3 === 0 ? 6 : (seed % 3 === 1 ? 5 : 4);
        const orbitStretchX = 0.9 + ((seed >> 1) % 5) * 0.08;
        const orbitStretchY = 0.78 + ((seed >> 3) % 5) * 0.06;
        const petalOrbitX = baseR * orbitStretchX;
        const petalOrbitY = baseR * orbitStretchY;
        const blossomRotation = phase * 0.35 + Math.sin(nowMs * 0.0007 + phase) * 0.06;
        for (let i = 0; i < petalCount; i++) {
            const a = blossomRotation + i * (Math.PI * 2 / petalCount);
            const cosA = Math.cos(a);
            const sinA = Math.sin(a);
            const px = cosA * petalOrbitX;
            const py = sinA * petalOrbitY;
            const petalSize = baseR * (0.88 + ((seed + i) % 4) * 0.07);
            const tipMul = 1.18 + ((seed + i) % 3) * 0.06;
            const tipX = px * tipMul;
            const tipY = py * tipMul;
            const nx = -sinA;
            const ny = cosA;

            const petalSatScale = 0.86 + ((seed + i * 3) % 7) * 0.038;
            const petalLightDelta = (((seed + i * 5) % 9) - 4) * 0.024;
            const petalHueShift = (((seed + i * 7) % 7) - 3) * 3.0;
            const petalBaseTone = adjustHexHsl(blossomBase, petalHueShift, petalSatScale, petalLightDelta);
            const petalTone = lerpHex(petalBaseTone, i % 2 ? toneLight : toneWarm, 0.28);
            const lobeToneA = lerpHex(petalTone, toneLighter, 0.18 + ((seed + i) % 3) * 0.08);
            const lobeToneB = lerpHex(petalTone, toneDark, 0.12 + ((seed + i + 1) % 2) * 0.1);
            const opaqueBoost = (seed + i * 11) % 6 === 0 ? 1.15 : 0;
            const petalOpacity = Math.min(1, blossomAlpha * (0.74 + ((seed + i * 2) % 6) * 0.06 + opaqueBoost));

            ctx.globalAlpha = petalOpacity;
            ctx.fillStyle = petalTone;
            ctx.beginPath();
            ctx.arc(px, py, petalSize, 0, Math.PI * 2);
            ctx.fill();

            const lobeOffset = petalSize * (0.2 + ((seed + i) % 2) * 0.05);
            ctx.globalAlpha = Math.min(1, petalOpacity * 0.94);
            ctx.fillStyle = lobeToneA;
            ctx.beginPath();
            ctx.arc(tipX + nx * lobeOffset, tipY + ny * lobeOffset, petalSize * 0.58, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = Math.min(1, petalOpacity * 0.9);
            ctx.fillStyle = lobeToneB;
            ctx.beginPath();
            ctx.arc(tipX - nx * lobeOffset, tipY - ny * lobeOffset, petalSize * 0.54, 0, Math.PI * 2);
            ctx.fill();

            ctx.globalAlpha = Math.min(1, petalOpacity * 0.55);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.16)';
            ctx.beginPath();
            ctx.arc(
                px - cosA * petalSize * 0.28 - nx * petalSize * 0.08,
                py - sinA * petalSize * 0.28 - ny * petalSize * 0.08,
                petalSize * 0.32,
                0,
                Math.PI * 2
            );
            ctx.fill();

            ctx.globalAlpha = Math.min(1, petalOpacity * 0.68);
            ctx.fillStyle = lerpHex(lobeToneB, toneDark, 0.45);
            ctx.beginPath();
            ctx.arc(px * 0.48, py * 0.48, petalSize * 0.19, 0, Math.PI * 2);
            ctx.fill();
        }

        const centerR = 1.6 + (seed % 3) * 0.22;
        ctx.globalAlpha = Math.min(1, blossomAlpha * 0.95);
        ctx.beginPath();
        ctx.arc(0, 0, centerR, 0, Math.PI * 2);
        ctx.fillStyle = '#fff5b8';
        ctx.fill();
        ctx.globalAlpha = Math.min(1, blossomAlpha * 0.85);
        ctx.beginPath();
        ctx.arc(0, 0, centerR * 0.55, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 232, 160, 0.9)';
        ctx.fill();
        ctx.globalAlpha = Math.min(1, blossomAlpha * 0.8);
        ctx.fillStyle = 'rgba(150, 115, 40, 0.55)';
        for (let i = 0; i < 6; i++) {
            const a = i * (Math.PI * 2 / 6) + phase * 0.4;
            ctx.beginPath();
            ctx.arc(Math.cos(a) * 1.9, Math.sin(a) * 1.7, 0.34, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.restore();
    }

    function drawNode(ctx, node, t) {
        const growT = Math.min(1, Math.max(0, (t - node.depth * 300) / 600));
        if (growT < 1) return;

        const glow = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(performance.now() * 0.002 + node.glowPhase));
        const p = projectWorld(node.pos);
        const r = node.depth >= MAX_DEPTH - 1 ? 2 : 3;

        ctx.beginPath();
        ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 199, 221, ${glow * 0.6})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(p.sx, p.sy, r + 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 199, 221, ${glow * 0.15})`;
        ctx.fill();
    }

    let energyDots = [];
    let lastEnergySpawn = 0;
    function updateEnergyDots(t, dt60) {
        if (t > 2200 && t - lastEnergySpawn > 4000 && treeBranches.length > 0) {
            lastEnergySpawn = t;
            const pathLen = 3 + Math.floor(Math.random() * (MAX_DEPTH - 2));
            const branches = [];
            const current = treeBranches.filter(b => b.depth === 0);
            if (current.length === 0) return;
            let picked = current[(Math.random() * current.length) | 0];
            branches.push(picked);
            for (let d = 1; d < pathLen; d++) {
                const children = treeBranches.filter(
                    b =>
                        b.depth === d &&
                        Math.abs(b.start.x - picked.end.x) < 1 &&
                        Math.abs(b.start.y - picked.end.y) < 1 &&
                        Math.abs(b.start.z - picked.end.z) < 1
                );
                if (children.length === 0) break;
                picked = children[(Math.random() * children.length) | 0];
                branches.push(picked);
            }
            energyDots.push({ branches, progress: 0, speed: 0.0015 + Math.random() * 0.001 });
        }

        energyDots = energyDots.filter(dot => {
            dot.progress += dot.speed * 16 * dt60;
            return dot.progress < dot.branches.length;
        });
    }

    function drawEnergyDots(ctx) {
        for (const dot of energyDots) {
            const idx = Math.floor(dot.progress);
            if (idx >= dot.branches.length) continue;
            const frac = dot.progress - idx;
            const b = dot.branches[idx];
            const pos = {
                x: b.start.x + (b.end.x - b.start.x) * frac,
                y: b.start.y + (b.end.y - b.start.y) * frac,
                z: b.start.z + (b.end.z - b.start.z) * frac,
            };
            const p = projectWorld(pos);

            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = '#ffe8f3';
            ctx.fill();

            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 6, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 182, 213, 0.25)';
            ctx.fill();
        }
    }

    function resizeTreeCanvas() {
        const rect = treeCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        treeCanvas.width = rect.width * dpr;
        treeCanvas.height = rect.height * dpr;
        treeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function setupTreeInteraction() {
        function releasePointer(e) {
            if (rotationState.activePointerId !== e.pointerId) return;
            rotationState.dragging = false;
            if (treeCanvas.hasPointerCapture(e.pointerId)) {
                treeCanvas.releasePointerCapture(e.pointerId);
            }
            rotationState.activePointerId = null;
            treeCanvas.style.cursor = 'grab';
        }

        treeCanvas.addEventListener('pointerdown', e => {
            if (rotationState.activePointerId !== null) return;
            rotationState.activePointerId = e.pointerId;
            rotationState.dragging = true;
            rotationState.lastX = e.clientX;
            rotationState.lastY = e.clientY;
            rotationState.lastT = performance.now();
            treeCanvas.setPointerCapture(e.pointerId);
            treeCanvas.style.cursor = 'grabbing';
        });

        treeCanvas.addEventListener('pointermove', e => {
            if (!rotationState.dragging || rotationState.activePointerId !== e.pointerId) return;
            const now = performance.now();
            const dtMs = Math.max(1, now - rotationState.lastT);
            const dt60 = dtMs / 16.6667;
            const dx = e.clientX - rotationState.lastX;
            const dy = e.clientY - rotationState.lastY;

            const yawDelta = dx * rotationState.pixelsToRadians;
            const pitchDelta = dy * rotationState.pixelsToRadians;
            rotationState.yaw += yawDelta;
            rotationState.pitch += pitchDelta;
            rotationState.pitch = Math.max(-rotationState.maxPitch, Math.min(rotationState.maxPitch, rotationState.pitch));
            rotationState.yawVelocity = yawDelta / dt60;
            rotationState.pitchVelocity = pitchDelta / dt60;
            rotationState.lastX = e.clientX;
            rotationState.lastY = e.clientY;
            rotationState.lastT = now;
        });

        treeCanvas.addEventListener('pointerup', releasePointer);
        treeCanvas.addEventListener('pointercancel', releasePointer);
        treeCanvas.addEventListener('lostpointercapture', releasePointer);
    }

    function animateTree(now = performance.now()) {
        if (!treeGenerated) {
            requestAnimationFrame(animateTree);
            return;
        }

        const rect = treeCanvas.getBoundingClientRect();
        if (treeCanvas.width !== rect.width * (window.devicePixelRatio || 1)) {
            resizeTreeCanvas();
        }

        if (!rotationState.lastFrameTs) rotationState.lastFrameTs = now;
        const dtMs = Math.max(1, now - rotationState.lastFrameTs);
        const dt60 = Math.min(4, dtMs / 16.6667);
        rotationState.lastFrameTs = now;
        const targetScale = treeCanvas.classList.contains('shrunk') ? TREE_SCALE_SHRUNK : TREE_SCALE_INITIAL;
        treeRenderScale += (targetScale - treeRenderScale) * Math.min(1, 0.12 * dt60);

        if (!rotationState.dragging) {
            rotationState.yaw += rotationState.yawVelocity * dt60;
            rotationState.pitch += rotationState.pitchVelocity * dt60;

            if (rotationState.autoRotateEnabled) {
                if (now < rotationState.boostUntil) {
                    rotationState.yawVelocity += 0.0013 * dt60;
                } else if (Math.abs(rotationState.yawVelocity) < 0.0009) {
                    rotationState.yaw += rotationState.cruiseYawSpeed * dt60;
                }
            }

            const damp = Math.pow(rotationState.frictionPer60, dt60);
            rotationState.yawVelocity *= damp;
            rotationState.pitchVelocity *= damp;
            rotationState.pitch = Math.max(-rotationState.maxPitch, Math.min(rotationState.maxPitch, rotationState.pitch));
        }

        const w = rect.width;
        const h = rect.height;
        treeCtx.clearRect(0, 0, w, h);
        treeCtx.save();
        treeCtx.translate(w / 2, h - 20);
        treeCtx.scale(treeRenderScale, treeRenderScale);

        const t = now - growthStart;
        const swayOffsets = { time: now * 0.001 };

        const drawQueue = [];
        for (const branch of treeBranches) {
            const segment = getAnimatedSegment(branch, t, swayOffsets);
            if (!segment) continue;
            drawQueue.push({ branch, segment, depthKey: rotatedDepthKey(segment) });
        }
        drawQueue.sort((a, b) => a.depthKey - b.depthKey);

        for (const item of drawQueue) {
            drawPrism(treeCtx, item.branch, item.segment);
        }

        const bloomBase = t - MAX_DEPTH * 270;
        for (const tip of branchTips) {
            const seed = Math.abs(Math.floor(tip.x * 1.7 + tip.y * 2.3 + tip.z * 3.1));
            const delayMs = seed % 420;
            const raw = (bloomBase - delayMs) / 760;
            if (raw <= 0) continue;
            const clamped = Math.min(1, raw);
            const bloom = 1 - Math.pow(1 - clamped, 3);
            drawLeaf(treeCtx, tip, bloom, now);
        }

        for (const node of treeNodes) {
            drawNode(treeCtx, node, t);
        }

        updateEnergyDots(t, dt60);
        drawEnergyDots(treeCtx);

        treeCtx.restore();
        requestAnimationFrame(animateTree);
    }

    const PETAL_PATHS = [
        'M8 0C8 0 10 6 8 10C6 14 2 16 0 16C0 16 2 10 4 6C6 2 8 0 8 0Z',
        'M6 0C6 0 9 4 9 8C9 12 6 16 3 16C0 16 0 12 2 8C4 4 6 0 6 0Z',
        'M5 0C5 0 10 3 10 8C10 13 6 16 3 16C0 16 1 11 3 7C5 3 5 0 5 0Z',
    ];
    const BLOSSOM_PETALS = ['#a6dfbf', '#bfe6c2', '#d9e4ba', '#f2d3c6', '#ffbfd8', '#ffacd2'];

    function tipToScreen(tip) {
        const canvasRect = treeCanvas.getBoundingClientRect();
        const p = projectWorld(tip);
        const originX = canvasRect.left + canvasRect.width / 2;
        const originY = canvasRect.top + canvasRect.height;
        return {
            x: originX + p.sx * treeRenderScale,
            y: originY + (p.sy - 20) * treeRenderScale,
        };
    }

    function spawnPetal() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 16 16');
        svg.classList.add('petal');

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', PETAL_PATHS[(Math.random() * PETAL_PATHS.length) | 0]);
        path.setAttribute('fill', BLOSSOM_PETALS[(Math.random() * BLOSSOM_PETALS.length) | 0]);
        svg.appendChild(path);

        const size = 22 + Math.random() * 20;
        svg.style.width = size + 'px';
        svg.style.height = size + 'px';

        const fromTip = branchTips.length > 0 && Math.random() < 0.6;
        let sideBias = 0;
        if (fromTip) {
            const tip = branchTips[(Math.random() * branchTips.length) | 0];
            const screen = tipToScreen(tip);
            svg.style.left = screen.x + 'px';
            svg.style.top = screen.y + 'px';
            sideBias = (screen.x - window.innerWidth * 0.5) / Math.max(1, window.innerWidth * 0.5);
        } else {
            svg.style.left = Math.random() * 100 + 'vw';
            svg.style.top = Math.random() * 100 + 'vh';
        }

        const dx = fromTip ? sideBias * (90 + Math.random() * 160) + (Math.random() - 0.5) * 70 : (Math.random() - 0.5) * 300;
        const dy = fromTip ? -170 - Math.random() * 260 : -150 - Math.random() * 250;
        const rot = (Math.random() - 0.5) * 720;
        const dur = 6 + Math.random() * 8;

        svg.style.setProperty('--dx', dx + 'px');
        svg.style.setProperty('--dy', dy + 'px');
        svg.style.setProperty('--rot', rot + 'deg');
        svg.style.animationDuration = dur + 's';

        document.body.appendChild(svg);
        svg.addEventListener('animationend', () => svg.remove());
    }

    buildTree();
    resizeTreeCanvas();
    setupTreeInteraction();
    window.addEventListener('resize', resizeTreeCanvas);
    requestAnimationFrame(animateTree);
    setInterval(spawnPetal, 350);
    for (let i = 0; i < 20; i++) setTimeout(spawnPetal, i * 150);

    window.haikuTree = {
        triggerGenerateSpin,
    };
})();
