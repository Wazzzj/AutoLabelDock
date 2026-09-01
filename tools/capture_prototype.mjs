import { chromium } from "playwright";
import fs from "fs";

const OUTPUT_DIR = "ui_reference/prototype";

fs.mkdirSync(OUTPUT_DIR, {
  recursive: true,
});

const pages = [
  {
    name: "train",
    url: "http://127.0.0.1:9001/a-train.html",
  },
  {
    name: "models",
    url: "http://127.0.0.1:9001/a-models.html",
  },
];

const browser = await chromium.launch();

const page = await browser.newPage({
  viewport: {
    width: 1440,
    height: 900,
  },
});

for (const item of pages) {
  console.log(`正在截图：${item.url}`);

  await page.goto(item.url, {
    waitUntil: "networkidle",
  });

  await page.screenshot({
    path: `${OUTPUT_DIR}/${item.name}.png`,
    fullPage: false,
  });

  console.log(
    `已保存：${OUTPUT_DIR}/${item.name}.png`
  );
}

await browser.close();

console.log("原型截图完成");