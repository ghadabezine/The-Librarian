// End-to-end UI check: open the app, ask a question, wait for the cited answer,
// click the first citation and save screenshots to docs/screenshots/.
// Requires the backend on :8000 and `npx vite --port 5173` in frontend/.
//   node scripts/ui_smoke_test.mjs "ما هو السن الأدنى للزواج في تونس؟"
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const question = process.argv[2] || "ما هو السن الأدنى للزواج في تونس؟";
const name = process.argv[3] || "chat";
mkdirSync("docs/screenshots", { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1360, height: 860 } });
await page.goto("http://localhost:5173/");
await page.screenshot({ path: "docs/screenshots/home.png" });
await page.fill("textarea", question);
await page.keyboard.press("Enter");
await page.waitForSelector("text=/S1|S2|S3|لم أجد|could not|pas trouvé/", { timeout: 300000 });
const cite = page.locator("button[aria-label^='source']").first();
if (await cite.count()) await cite.click();
await page.waitForTimeout(500);
await page.screenshot({ path: `docs/screenshots/${name}.png` });
const answer = await page.locator("div.rounded-2xl.border.bg-white").last().innerText();
console.log(answer);
await browser.close();
