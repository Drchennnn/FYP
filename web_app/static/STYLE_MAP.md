# STYLE_MAP.md

本文件用于集中记录「组件说明 / 用途 / 代码位置」，避免把 HTML 结构说明散落在 CSS 注释里。

> 约定：本项目 UI 组件统一使用命名空间前缀 **`uiw-`**（ui web），或置于父容器 `.uiw` 下，防止与 Bootstrap/现有页面类名（如 `.container`, `.card`, `.content`）冲突。

---

## 1) Weather Card（天气卡片）
- **用途**：展示当日/未来天气摘要；hover 时可展示更详细信息（地点、日期、温度、图标/文字）。
- **HTML**：`static/html/1.html`（注释：`天气组件`）
- **CSS**：`static/css/1.css`（原选择器：`.card`, `.card-header`, `.temp`, `.temp-scale`, `.cloud`, `.sun` 等）
- **建议落地命名**：
  - 容器：`.uiw-weather-card`
  - 内部：`.uiw-weather-card__header`, `.uiw-weather-card__temp`, `.uiw-weather-card__scale`

## 2) Error Alert（错误提示组件）
- **用途**：用于展示 API 拉取失败、离线产物缺失等错误。
- **HTML**：`static/html/1.html`（注释：`错误提示组件`）
- **CSS**：该组件大量依赖 Tailwind 风格类名（如 `flex`, `gap-2`），如果项目未引入 Tailwind，则需要改写为本地 CSS。

## 3) Day/Night Toggle（日夜切换）
- **用途**：切换主题（可扩展为中英文切换按钮）。
- **HTML**：`static/html/1.html`（注释：`日间夜间切换组件`）
- **CSS**：`static/css/1.css`（原选择器：`.main-circle`, `.phone`, `.toggle`, `.names`, `.light`, `.dark` 等）
- **建议落地命名**：`.uiw-theme-toggle`（避免 `.content` 等通用名冲突）

## 4) Spinner（统一加载动画）
- **用途**：数据拉取时覆盖层加载动画。
- **HTML**：`static/html/1.html`（注释：`统一的加载动画`）
- **CSS**：`static/css/1.css`（原选择器：`.spinner`, `.spinner1`, `@keyframes spinning82341`）
- **建议落地命名**：`.uiw-spinner` / `.uiw-spinner__inner`

## 5) Background Pattern（背景纹理）
- **用途**：页面背景 pattern。
- **HTML**：`static/html/1.html`（注释：`背景（pattern）`）
- **CSS**：`static/css/1.css`（原选择器：`.container` —— 强烈建议改名，避免污染 Bootstrap）
- **建议落地命名**：`.uiw-bg-pattern`

## 6) Gradient Button（重要操作按钮）
- **用途**：关键操作按钮（如“刷新/重新读取离线结果”）。
- **HTML**：`static/html/1.html`（注释：`按钮（gradient）`）
- **CSS**：`static/css/1.css`（原选择器：`.btn`, `#container-stars`, `#stars`, `#glow` 等）
- **建议落地命名**：`.uiw-btn-primary`（并将 id 改为 class，避免同页多按钮时 id 冲突）

## 7) Curve Toggle Checkbox（曲线显示/隐藏）
- **用途**：图表中某条曲线显示/隐藏。
- **HTML**：`static/html/1.html`（注释：`按钮，用于在图表中显示/隐藏数据或者曲线`）
- **CSS**：`static/css/1.css`（原选择器：`.checkBox`, `.transition` 等）
- **建议落地命名**：`.uiw-curve-toggle`

## 8) Thermometer / Threshold Lab（温度计/阈值评测）
- **用途**：阈值/风险评测的实验性组件（建议放到子页面 /definitions 或 /lab）。
- **HTML**：`static/html/1.html`（注释：`温度计/阈值评测`）
- **CSS**：`static/css/1.css`（原选择器包含 `body/html/svg` 的全局样式 —— 必须命名空间化）
- **JS**：`static/js/main.js`（尾部 GSAP/TweenMax/Draggable 逻辑）
- **建议落地命名**：父容器 `.uiw-thermo`，所有样式前缀 `.uiw-thermo ...`，避免影响全站。

