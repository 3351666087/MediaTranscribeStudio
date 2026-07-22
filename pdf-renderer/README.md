# MediaTranscribeStudio Java PDF Renderer

唯一生产 PDF 链：**Java 17 + OpenHTMLtoPDF + Apache PDFBox**。

## 核心保证

- 任意正整数说话人数 `N`，无固定五人分支；回归覆盖 `N=1/2/5/8/13`。
- 说话人视觉样式的唯一入口：

  ```java
  new SpeakerPalette().style(speaker.order, speaker.colorToken)
  ```

- 内置开源静态 TrueType 中文字体 `LXGW WenKai v1.522`，运行时不扫描系统字体、不访问网络、不加载远程资源。
- 中文文本可搜索；PDFBox 检查嵌入字体、正文、段号、时间戳和完整说话人集合。
- 13 个不可补偿 hard gates 与 14 个 `AESTHETIC-*` facets；facet 权重精确合计 `1.0`，最低分 `85`。
- 最多五轮确定性版式修复；修复只能调整模板、CSS、字体尺寸、边距和分页，前后正文哈希必须一致。
- 每页生成 PNG，另产出 contact sheet、inspection、quality report、repair queue 和 artifact manifest。
- CLI stdout 严格只输出一个 JSON；所有日志进入 stderr；未知字段、尾随 JSON、解析错误和质量失败均非零退出。

## 构建

```powershell
mvn "-Dmaven.repo.local=target/m2" clean test
mvn "-Dmaven.repo.local=target/m2" package
```

独立 shaded JAR：

```text
target/pdf-renderer.jar
```

## CLI

```powershell
java -jar target/pdf-renderer.jar --request D:\jobs\request.json
```

请求使用 `schemas/pdf-render-request.schema.json`，并通过 `reportDocumentPath` 指向版本化 `ReportDocument`。
`reportDocumentPath` 必须位于 `outputDirectory` 内；该限制保证 manifest 使用安全相对路径并阻止输出目录逃逸。

成功输出结构：

```text
input/report-document.json
render/report.xhtml
render/report.pdf
artifacts/screens/page-001.png
artifacts/contact-sheet.png
artifacts/pdf-inspection.json
artifacts/pdf-extracted-text.txt
artifacts/quality-report.json
artifacts/repair-queue.json
artifacts/manifest.json
artifacts/render-result.json
qa/round-01/...
```

## Design Pack 内化

模块采用 `frontend-design-pack-global` 的适用硬离线、层级、排版、色彩、密度、节制、真实内容压力、
字体失败、图像失败和脚本失败检查。外部模板包本身仍声明 `implementationReady=false`，因此本模块只声称
项目级 PDF QA 契约已内化，不声称外部模板包达到 release-ready。
