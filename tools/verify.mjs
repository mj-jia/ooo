#!/usr/bin/env node
// 校验仓库自洽性：模型/运行时文件是否与 manifest 一致，页面引用的资源是否真实存在。
// 用法：node tools/verify.mjs   （或 npm run verify）
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

let failures = 0;
const ok = (msg) => console.log(`  \x1b[32m✓\x1b[0m ${msg}`);
const bad = (msg) => {
  failures++;
  console.log(`  \x1b[31m✗\x1b[0m ${msg}`);
};

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

/* ---------- 1. manifest 与实际文件 ---------- */
const manifestPath = path.join(ROOT, 'models/manifest.json');
if (!fs.existsSync(manifestPath)) {
  console.error('找不到 models/manifest.json');
  process.exit(1);
}
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

console.log('\n模型与运行时文件');

const entries = [
  ...manifest.models.map((m) => ({ label: m.file, file: path.join('models', m.file), bytes: m.bytes, sha256: m.sha256 })),
  ...Object.entries(manifest.runtime.files).map(([name, f]) => ({
    label: name,
    file: path.join('vendor', name),
    bytes: f.bytes,
    sha256: f.sha256,
  })),
];

function human(bytes) {
  return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

for (const e of entries) {
  const abs = path.join(ROOT, e.file);
  const show = e.file.split(path.sep).join('/');
  if (!fs.existsSync(abs)) {
    bad(`${show} 不存在`);
    continue;
  }
  const size = fs.statSync(abs).size;
  if (Number.isFinite(e.bytes) && size !== e.bytes) {
    bad(`${show} 大小不符：${size} ≠ ${e.bytes}`);
    continue;
  }
  const digest = sha256(abs);
  if (e.sha256 && digest !== e.sha256) {
    bad(`${show} sha256 不符：${digest.slice(0, 16)}… ≠ ${e.sha256.slice(0, 16)}…`);
    continue;
  }
  ok(`${show}  ${human(size)}  ${digest.slice(0, 12)}…`);
}

/* ---------- 2. 页面引用的本地资源 ---------- */
function collectRefs(source, pattern) {
  const refs = new Set();
  for (const match of source.matchAll(pattern)) refs.add(match[1]);
  return refs;
}

const checks = [
  { from: 'index.html', pattern: /(?:href|src)="\.\/([^"]+)"/g },
  { from: 'app.js', pattern: /from\s+'\.\.?\/([^']+)'/g },
  { from: 'app.js', pattern: /url:\s*'\.\/([^']+)'/g },
];

console.log('\n页面引用的资源');

for (const { from, pattern } of checks) {
  const source = fs.readFileSync(path.join(ROOT, from), 'utf8');
  const base = path.dirname(path.join(ROOT, from));
  for (const ref of collectRefs(source, pattern)) {
    const target = ref.startsWith('..') ? path.join(base, ref) : path.join(ROOT, ref);
    if (fs.existsSync(target)) {
      ok(`${from} → ${ref}`);
    } else {
      bad(`${from} 引用了不存在的文件：${ref}`);
    }
  }
}

/* ---------- 3. 前端配置与 manifest 是否对得上 ---------- */
console.log('\n前端配置');

const appSource = fs.readFileSync(path.join(ROOT, 'app.js'), 'utf8');
const urlToModel = new Map(
  manifest.models.map((m) => [`./models/${m.file}`, m])
);

for (const [ref, model] of urlToModel) {
  const re = new RegExp(`url:\\s*'${ref.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}'[\\s\\S]{0,400}?input:\\s*'([^']+)'`);
  const hit = appSource.match(re);
  if (!hit) {
    bad(`app.js 里没有引用 ${ref}`);
    continue;
  }
  if (hit[1] === model.input.name) {
    ok(`${model.file} 输入节点名一致（${hit[1]}）`);
  } else {
    bad(`${model.file} 输入节点名不一致：app.js 写 ${hit[1]}，manifest 写 ${model.input.name}`);
  }
}

/* ---------- 结果 ---------- */
if (failures) {
  console.error(`\n${failures} 项未通过\n`);
  process.exit(1);
}
console.log('\n全部通过\n');
