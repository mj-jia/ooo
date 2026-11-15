// 端到端冒烟测试：用本机 Edge 打开页面，上传一张小图，跑一次真实推理，检查输出。
// 用法：node tools/e2e.mjs [图片路径]
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const URL_BASE = 'http://127.0.0.1:8123/';
const IMG = process.argv[2] || 'tools/out/small128.png';
const MODEL = process.argv[3] || 'fast';
const OUT = 'tools/out';
const TIMEOUT = 420_000;

fs.mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  protocolTimeout: 900_000,
  args: ['--no-sandbox', '--disable-gpu', '--enable-unsafe-webgpu', '--window-size=1280,1100'],
  defaultViewport: { width: 1280, height: 1100 },
});

const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
page.on('requestfailed', (r) => logs.push(`[reqfail] ${r.url()} ${r.failure()?.errorText}`));

console.log('→ 打开页面');
await page.goto(URL_BASE, { waitUntil: 'domcontentloaded', timeout: 60_000 });

// 检查首屏隐藏状态
const workHidden = await page.$eval('#work', (el) => el.hidden);
console.log('   work 区域初始隐藏:', workHidden);

console.log('→ 上传图片', path.resolve(IMG));
const input = await page.$('#file');
await input.uploadFile(path.resolve(IMG));
await page.waitForFunction(() => !document.getElementById('work').hidden, { timeout: 20_000 });

const before = await page.evaluate(() => ({
  name: document.getElementById('mName').textContent,
  size: document.getElementById('mSize').textContent,
  out: document.getElementById('mOut').textContent,
  beforeW: document.getElementById('cvBefore').width,
  beforeH: document.getElementById('cvBefore').height,
}));
console.log('   元信息:', JSON.stringify(before));

console.log('→ 开始推理（模型:', MODEL, '）');
if (MODEL !== 'fast') {
  await page.select('#model', MODEL);
  await new Promise((r) => setTimeout(r, 300));
}
const t0 = Date.now();
await page.click('#run');

await page.waitForFunction(
  () => {
    const dl = document.getElementById('download');
    const err = document.getElementById('err');
    return (!dl.disabled && dl.textContent.includes('下载')) || !err.hidden;
  },
  { timeout: TIMEOUT, polling: 2000 }
);

const result = await page.evaluate(() => ({
  time: document.getElementById('mTime').textContent,
  dev: document.getElementById('mDev').textContent,
  outSize: document.getElementById('mOut').textContent,
  err: document.getElementById('err').hidden ? null : document.getElementById('err').textContent,
  afterW: document.getElementById('cvAfter').width,
  afterH: document.getElementById('cvAfter').height,
  progress: document.getElementById('ptext').textContent,
}));

console.log('   结果:', JSON.stringify(result, null, 2));
console.log('   壁钟耗时:', ((Date.now() - t0) / 1000).toFixed(1), 's');

if (result.err) {
  console.error('✗ 页面报错:', result.err);
} else {
  const dataUrl = await page.evaluate(() =>
    document.getElementById('cvAfter').toDataURL('image/png')
  );
  fs.writeFileSync(path.join(OUT, 'browser_output.png'), Buffer.from(dataUrl.split(',')[1], 'base64'));
  console.log('   已保存 tools/out/browser_output.png');
}

await page.screenshot({ path: path.join(OUT, 'ui_result.png'), fullPage: true });

if (logs.length) {
  console.log('\n控制台输出:');
  for (const l of logs.slice(0, 40)) console.log('   ', l);
}

await browser.close();
console.log('完成');
