# 5×5 便携式光场视频数据集：当前阶段最终指南（MD版）

## 0. 一句话结论

这项工作 **值得继续做** ，而且不必为了“全量无损 PNG”把自己拖进存储黑洞。当前最稳妥的路线是：

* **原始层**保留 `4K + D-Log M 10-bit + MP4`；
* **主 benchmark 发布层**做成 `1080p + 校正后 + 标准化颜色 + 8-bit JPEG 帧序列`；
* **高保真子集**再单独保留少量 `PNG`；
* **处理流程**坚持“从原始 MP4 直接流式处理到最终帧”，不要先转一个中间 CRF 视频再抽帧；
* **投稿策略**优先考虑  **ACCV 2026** ，有余力再冲  **BMVC 2026** 。BMVC 截止更早；ACCV 给你更多时间补 baseline、统计和 release。BMVC 2026 的摘要/正文截止分别是 5 月 22 日和 5 月 29 日；ACCV 2026 的注册/投稿截止分别是 7 月 3 日和 7 月 5 日。([BMVC 2026](https://bmvc2026.bmva.org/calls/call-for-papers/?utm_source=chatgpt.com "Call for Papers - BMVC 2026"))

---

## 1. 项目定位：这到底是一篇什么样的论文

你这项工作不只是“采了一个数据集”，更像是一个 **复合型资源论文** ：

* 一个  **便携式 5×5 消费级运动相机阵列** ；
* 一套  **后期软件同步、标定、校正、裁剪、标准化发布流程** ；
* 一个  **真实动态光场视频资源** ，可支持后续的视角补全、视角合成、时空超分等任务。

这意味着它既可以往 **dataset / benchmark** 写，也可以往 **resource / pipeline** 写。
如果时间更紧，文章就更应该偏  **benchmark-oriented** ；如果时间更宽裕，就可以把“为什么便携、为什么软件同步、这套 trade-off 有什么价值”讲得更完整。

---

## 2. 为什么你现在的 PNG 会这么大

你现在看到的“单张 PNG 20MB+”， **并不奇怪** ，主要原因有两个：

### 2.1 4K 实拍内容本来就很难被 PNG 压小

PNG 是 **无损格式** 。对高分辨率、纹理丰富、噪声较多的实拍视频帧，它往往压不动。PNG 规范允许的采样深度范围是 1–16 bit，而对多通道真彩图像，常用的是  **8-bit 和 16-bit** 。([w3.org](https://www.w3.org/TR/png-3/?utm_source=chatgpt.com "Portable Network Graphics (PNG) Specification (Third ..."))

### 2.2 你的“10-bit PNG”很可能实际上落到了高位深 PNG 档

PNG 对多样本像素并不是“天然 10-bit 一档”，而是 **8-bit / 16-bit** 更常见；所以从 10-bit 视频导高位深 PNG 时，很可能会变成更大的高位深 PNG。你自己也已经观察到：同一帧导成 8-bit 图像后，体积会显著下降。这个现象和 PNG 规范是吻合的。([w3.org](https://www.w3.org/TR/png-3/?utm_source=chatgpt.com "Portable Network Graphics (PNG) Specification (Third ..."))

---

## 3. `ffprobe` 显示 `bt709`，是不是已经是 Rec.709 了？

**不能直接这么下结论。**

你看到：

```text
pix_fmt=yuv420p10le
color_space=bt709
color_transfer=bt709
color_primaries=bt709
```

这更像是在说明 **编码流里的颜色相关属性/标签** ，不等于你已经完成了“D-Log M → Rec.709 的显示映射”。FFmpeg 文档本身就把这类颜色项当作流/帧属性来处理；它们可以被设置，但“被标成 bt709”不等于画面已经经过了完整的显示风格转换。([DJI Download Center](https://dl.djicdn.com/downloads/DJI_Osmo_Action_5_Pro/UM/20240919/DJI_Osmo_Action_5_Pro_User_Manual_v1.0_en.pdf?utm_source=chatgpt.com "User Manual"))

另外，DJI 官方手册明确写了：Action 5 Pro 支持  **Normal 8-bit、Normal 10-bit、HLG 10-bit、D-Log M 10-bit** ；其中  **D-Log M 是为后期专业调色设计的** 。DJI 还单独提供了 **Action 5 Pro D-Log M to Rec.709 LUT** 下载。这恰恰说明：D-Log M 本身不是“天然最终显示版”，而是一个保留后期空间的拍摄 profile。([DJI Download Center](https://dl.djicdn.com/downloads/DJI_Osmo_Action_5_Pro/UM/20240919/DJI_Osmo_Action_5_Pro_User_Manual_v1.0_ja.pdf?utm_source=chatgpt.com "ユーザーマニュアル"))

### 结论

* 你的素材 **不是 RAW** ，而是带有 D-Log M 颜色模式的视频文件；
* `bt709` 标签 **不等于** “已经做好了最终 Rec.709 标准化”；
* 对数据集发布来说，仍然建议你把 **颜色标准化** 当成 benchmark 生成流程的一步，而不是假设原素材天然就“已经好用了”。([DJI Download Center](https://dl.djicdn.com/downloads/DJI_Osmo_Action_5_Pro/UM/20240919/DJI_Osmo_Action_5_Pro_User_Manual_v1.0_ja.pdf?utm_source=chatgpt.com "ユーザーマニュアル"))

---

## 4. 数据集格式：最终定稿方案

## 4.1 第一层：Raw Archive（原始层）

**保留原始 MP4，不展开成全量 PNG。**

建议内容：

* `25` 个视点原始视频；
* `4K`；
* `D-Log M 10-bit`；
* 每个场景一个文件夹；
* 附带：
  * 相机编号 / 阵列布局；
  * 同步偏移；
  * 内参 / 外参 / 畸变参数；
  * rectification 参数；
  * crop 参数；
  * README；
  * split 文件；
  * 颜色说明（原始为 D-Log M，benchmark 层如何标准化）。

**为什么这样做：**
原始层的职责是“ **完整、可复现** ”，不是“人人拿来就能训练”。D-Log M 10-bit 对这层是合理的，因为 DJI 官方明确支持并定义了这个模式。([DJI Download Center](https://dl.djicdn.com/downloads/DJI_Osmo_Action_5_Pro/UM/20240919/DJI_Osmo_Action_5_Pro_User_Manual_v1.0_ja.pdf?utm_source=chatgpt.com "ユーザーマニュアル"))

---

## 4.2 第二层：Main Benchmark Release（主 benchmark 发布层）

**主发布层不要只给 MP4。**

最终建议定为：

* `1080p`；
* **同步后、校正后、裁剪后** ；
* **标准化颜色后的 8-bit** ；
* **JPEG 帧序列** ；
* 可选质量：`Q=95–98`。

### 为什么主发布层更推荐 JPEG，而不是只给 MP4

因为你的目标不是单纯“存档”，而是做一个 **研究者能直接用的 benchmark** 。
如果只给 MP4，使用者通常还要自己：

* 解码；
* 抽帧；
* 处理 GOP 依赖；
* 自己组织 dataloader。

而 JPEG 帧序列对视觉训练/评测会更直接。

### 为什么不是全量 8-bit PNG

因为即便你把处理后图像降到 8-bit，PNG 体积依旧可能很大；你已经测到单帧仍然可能有几 MB。全量场景一乘，整体还是会非常重。
所以主 benchmark 层追求的是： **可用 + 体积可接受 + 处理方便** 。在这个层面，JPEG 比全量 PNG 更平衡。

---

## 4.3 第三层：Fidelity Subset（高保真子集）

这层仍然保留，而且我建议你做。

建议：

* 选 `5–8` 个代表性场景；
* 导出少量  **PNG 帧序列** ；
* 用于：
  * 高保真定性展示；
  * reviewer 追问时补充；
  * 后续像素级分析；
  * 未来期刊扩展。

### 这一层为什么保留

因为你毕竟是低层视觉 / 光场视频场景，完全没有高保真样本会让人觉得你“只剩压缩版”。
但是这层 **只做小规模** ，不要做全量。

---

## 4.4 第四层：Preview MP4（可选预览层）

这层只是可选，不是主 benchmark。

建议：

* 给一份轻量版 `MP4`；
* 用于网页预览、快速浏览、下载前查看。

### 注意

这层 **不是** 原始 MP4 的替代品，也 **不是** 高保真子集的替代品。
它只是为了方便展示。

---

## 5. Rec.709 到底应该怎么做

这里的 “Rec.709” 不是文件容器，而是 **benchmark 层的标准显示目标** 。

### 最稳妥的做法

在导出 benchmark 帧的时候，使用 **DJI 官方 Action 5 Pro D-Log M → Rec.709 LUT** 做标准化。DJI 官方下载中心确实提供了这套 LUT。([DJI Official](https://www.dji.com/downloads/softwares/dji-osmo-action-5-pro-d-log-m-to-rec-709-vivid-lut?utm_source=chatgpt.com "DJI OSMO Action 5 pro D-Log M to Rec.709 vivid LUT"))

### 推荐处理顺序

从原始 MP4 出发，一次完成：

1. 解码；
2. 软件同步；
3. 标定 / rectification；
4. crop / resize；
5. 颜色标准化（LUT）；
6. 直接输出最终 `JPEG` 帧。

### 不推荐的顺序

不建议：

`原始 MP4 → 套 LUT → 再编码成中间 MP4（CRF）→ 再抽帧`

因为这样会多引入一次视频有损压缩，增加对下游任务的潜在影响。FFmpeg 的 `lut3d` 是一个对输入视频流逐帧应用 3D LUT 的滤镜，本质上可以直接放在处理链里，而不是必须先出中间视频。([DJI Download Center](https://dl.djicdn.com/downloads/DJI_Osmo_Action_5_Pro/UM/20240919/DJI_Osmo_Action_5_Pro_User_Manual_v1.0_en.pdf?utm_source=chatgpt.com "User Manual"))

---

## 6. 最终推荐的制作工作流

## 6.1 你自己本地的“生产工作流”

建议采用：

`Raw MP4`
→ 软件同步
→ 标定 / 校正 / 裁剪 / 降采样
→ 颜色标准化
→ **直接输出最终 JPEG 帧序列**

### 核心原则

* **不要**把整段视频先展开成海量中间 PNG；
* **不要**先做一个 CRF 中间视频再二次抽帧；
* **要**做 **流式处理** ：一帧进，一帧出。

这样你真正需要长期保存的只有：

* 原始 MP4；
* 最终 benchmark 帧；
* 参数文件；
* 少量高保真子集。

---

## 6.2 目录结构建议

```md
PortableLFV/
  raw/
    scene_001/
      view_00.mp4
      ...
      view_24.mp4
      sync_offsets.json
      calibration.yaml
      rectification.yaml
      crop.json
      readme.md

  benchmark/
    scene_001/
      view_00/
        000001.jpg
        000002.jpg
        ...
      ...
      metadata.json

  fidelity_subset/
    scene_003/
      view_00/
        000001.png
        000002.png
        ...
      ...

  preview/
    scene_001/
      view_00_preview.mp4
      ...
      scene_preview_grid.mp4

  splits/
    train.txt
    val.txt
    test.txt

  tools/
    export_benchmark_frames.py
    apply_rectification.py
    decode_video.sh
    readme.md
```

---

## 7. 为什么不建议只发布 MP4

单看“压缩和下载”，MP4 很好；
但单看“benchmark 易用性”， **只给 MP4 不够好** 。

你的数据集如果想被视觉研究者真正拿来训练和评测，主 benchmark 层应该让他们可以：

* 直接按帧读取；
* 直接组织输入/GT；
* 不需要自己额外抽帧预处理。

因此：

* **原始层** ：MP4 合理；
* **主 benchmark 层** ：JPEG 帧序列更合理；
* **预览层** ：再补 MP4。

---

## 8. 相关任务方向：你的数据集适合哪些 benchmark

你对领域分布的理解基本是对的：

* **动态光场视频**方向的方法相对少；
* 比较常见的主题集中在：
  * **编码 / 压缩 / streaming / processing** ；
  * **时空超分（space-time super-resolution）** ；
  * **视角补全 / 视角插值 / angular SR / view synthesis** 。

4DLFVD 这篇工作本身就把自己的定位写成支撑  **LF video coding, processing, and streaming** ；TIP 2023 的 **LFSTVSR** 明确是  **Space-Time Super-Resolution for Light Field Videos** 。同时，静态 LF 方向已经有比较成熟的 benchmark，尤其是 **Light Field Super-Resolution: A Benchmark** 这类工作，以及 Cambridge 的  **light field view interpolation benchmark** 。([ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3458305.3478450?utm_source=chatgpt.com "4DLFVD: A 4D Light Field Video Dataset"))

### 对你最现实的含义

你的第一篇论文 **不必强行只绑定“纯动态 LF video SOTA”** 。
可以把：

* **frame-wise 的 LF 方法** ；
* **动态视频方法** ；
* **简单几何 / warping 基线**

组合成一个足够说明问题的 benchmark。

---

## 9. baseline 为什么要做

baseline 不只是为了“证明你数据难”，它更重要的作用有三个：

1. **证明数据能跑**
   不是只有文件，而是能形成明确的 benchmark。
2. **定义任务协议**
   输入是什么，输出是什么，怎么评。
3. **解释难点**
   告诉别人：哪些场景最难，为什么难。

ACM MM 2026 的 dataset call 里就把 **baseline solution(s)** 明确列为提交内容之一。([ACM Multimedia 2026](https://2026.acmmm.org/site/call-datasets.html?utm_source=chatgpt.com "ACM Multimedia 2026 Conference — Call for Dataset Papers"))

### 你的第一优先级 baseline，不建议一上来只做 LFSTVSR

LFSTVSR 可以做，但它不应该是 **唯一主 baseline** 。
因为 LFSTVSR 对应的是时空超分，而你这个数据集最天然的卖点其实是：

* 25 个视点；
* 动态场景；
* 稠密角度结构；
* 软件同步和校正后的一致性。

### 更推荐的主 benchmark

我建议你的**第一主 benchmark**定成：

## Sparse-view → Dense-view Reconstruction / Angular Completion

例如：

* 输入：`5×5` 里的稀疏子集（如 `3×3`，或 cross/checkerboard 采样）；
* 输出：完整 `5×5`；
* GT：完整 25 视点；
* 指标：`PSNR / SSIM / LPIPS`。

### 基线建议（从轻到重）

1. **简单插值 / warping 非学习基线** ；
2. **一个 frame-wise 的 LF angular SR / view synthesis 方法** ；
3. **时间允许再加 LFSTVSR 作为附加 benchmark** 。

这样更符合你数据集的结构，也更容易先把论文立起来。

---

## 10. 场景统计要不要做

**建议做，但做精简版就够。**

VFI/视频数据集常会统计运动量、位移量；你这里虽然是光场视频，但其实同样有必要做一些 **难度画像** 。Cambridge 的 light-field interpolation benchmark 明确指出，**大视差**是主要挑战，**遮挡**和**非 Lambertian 表面**也会显著增加难度。([剑桥大学计算机实验室](https://www.cl.cam.ac.uk/research/rainbow/projects/lightfield-benchmark/lightfield_benchmark.pdf?utm_source=chatgpt.com "A BENCHMARK OF LIGHT FIELD VIEW INTERPOLATION ..."))

### 建议至少做这 4 类统计

1. **中心视点时间运动量**
   相邻帧光流幅值统计；
2. **跨视点视差范围**
   邻近视点之间的 disparity 幅值统计；
3. **遮挡强度**
   用前后向一致性或 warping mask 粗估；
4. **跨视点光度不一致**
   曝光 / 亮度差异统计。

### 为什么值得做

不是为了炫技，而是为了让 benchmark 结果更可解释：
为什么某些场景难、为什么某些方法掉得厉害、你的数据和已有资源相比到底“难”在哪里。

---

## 11. 4DLFVD 给你的启发是什么

4DLFVD 是一个 **很重要的先例** ，但不能完全照抄。

它发表在  **ACM MMSys 2021** ，摘要里明确说它是一个用于 **LF video coding, processing, and streaming** 的 4D light-field video dataset，使用  **10×10 相机矩阵、100 个摄像头、1920×1056 分辨率** ，强调的是“稀缺资源”和“系统价值”。([ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3458305.3478450?utm_source=chatgpt.com "4DLFVD: A 4D Light Field Video Dataset"))

### 你可以学它的地方

* 原始视频流 + 参数文件也是合理发布方式；
* 不一定非要全量 PNG 才算“正规”；
* “采集系统 + 资源”本身就是论文贡献。

### 你不该完全照抄的地方

* 它没有强 benchmark baseline，不代表你现在投 BMVC/ACCV 也可以不做；
* 现在视觉 reviewer 更习惯看到：
  * 明确任务定义；
  * train/val/test split；
  * baseline；
  * 难度统计。

---

## 12. 投稿策略：最终建议

## 12.1 当前最现实的选择

### 主攻：ACCV 2026

* Abstract registration: **2026-07-03**
* Paper submission: **2026-07-05**
* Topics 包括：
  * Computational Photography, Sensing, and Display
  * Datasets and Performance Analysis
  * Low-level Vision, Image Processing
  * 3D Computer Vision
  * Motion and Tracking
    ([ACCV 2026](https://accv2026.org/submissions/?utm_source=chatgpt.com "Submissions"))

**为什么主攻 ACCV：**

* 你还要找实习；
* baseline 还没做；
* 数据 release 和统计还需要时间；
* ACCV 给你更多缓冲。

---

### 冲刺：BMVC 2026

* Abstract deadline: **2026-05-22**
* Full paper deadline: **2026-05-29**
* Topics 包括：
  * Computational Photography and Photogrammetry
  * Datasets and Evaluation
  * Low-level and Physics-based Vision
  * Vision Applications, Systems, and Robotics
    ([BMVC 2026](https://bmvc2026.bmva.org/calls/call-for-papers/?utm_source=chatgpt.com "Call for Papers - BMVC 2026"))

**为什么 BMVC 是冲刺目标：**

* 题目匹配；
* 会比较有分量；
* 但时间明显更紧。

---

### 已经基本错过的口子

* **ACM MM 2026** main/dataset track：注册和投稿节点已经在 3 月 25 日和 4 月 1 日。([ACM Multimedia 2026](https://2026.acmmm.org/site/important-dates.html?utm_source=chatgpt.com "ACM Multimedia 2026 Conference — Important Dates"))
* **MMSys 2026 ODS track** ：官网已显示 submission closed。([ACM MMSys 2026](https://2026.acmmmsys.org/?utm_source=chatgpt.com "ACM MMSys 2026"))

---

## 12.2 关于 BMVC / EI / 毕业要求

这一点 **不要赌** 。

我目前查到的 BMVC 官方信息主要强调：

* BMVC 是 BMVA 的旗舰会议；
* 有 BMVA / BMVA Press 体系的 proceedings；
* 有在线 proceedings 页面。([bmva.org](https://www.bmva.org/bmvc?utm_source=chatgpt.com "The British Machine Vision Conference (BMVC)"))

**但我没有在官方页面上找到它明确承诺 “EI Compendex 检索” 的表述。**
所以如果你毕业要求严格写的是“EI 或 SCI”，那最稳妥的做法不是听传闻，而是：

* 让导师或学院秘书给出 **书面口径** ；
* 用本校既往毕业案例确认；
* 不要默认 BMVC 一定满足。

同理，ACCV 也建议按你们学院认定口径确认。

---

## 13. 论文写法建议：BMVC 和 ACCV 为什么看起来不一样

这不是因为两个会“规则完全不同”，而是因为 **你的时间窗口和最稳妥的卖点不同** 。

### BMVC 写法建议

更偏：

* dataset / benchmark；
* 数据是什么；
* 为什么难；
* 任务协议；
* baseline 和 split。

### ACCV 写法建议

更偏：

* portable acquisition；
* software synchronization / calibration pipeline；
* portability–accuracy trade-off；
* dataset/resource release。

### 不是互斥关系

BMVC 也可以讲 pipeline，ACCV 也要有 benchmark。
只是对你现在的完成度来说：

* **BMVC 适合“收紧版故事”** ；
* **ACCV 适合“展开版故事”** 。

---

## 14. 当前最终决策（可直接执行）

## 14.1 数据格式

* **Raw 层** ：`4K + D-Log M 10-bit + MP4 + 参数文件`
* **主 benchmark 层** ：`1080p + 校正后 + 标准化颜色 + 8-bit JPEG 帧序列`
* **高保真子集** ：少量 `PNG`
* **预览层** ：可选轻量 `MP4`

## 14.2 处理流程

* 从原始 MP4 出发；
* 一次完成同步 / 校正 / 裁剪 / resize / 颜色标准化；
* **直接输出最终 JPEG** ；
* 不做“中间 CRF 视频 → 再抽帧”的二次压缩流程。

## 14.3 benchmark

* **主任务** ：Sparse-view → Dense-view angular completion
* **首批 baseline** ：
* 简单插值 / warping
* 一个 frame-wise LF 角度补全 / angular SR 方法
* 时间足够再加 LFSTVSR

## 14.4 统计

* motion
* disparity
* occlusion
* photometric inconsistency

## 14.5 投稿

* **主投 ACCV 2026**
* **冲 BMVC 2026**
* BMVC/ACCV 是否满足毕业检索要求，**必须单独确认，不要默认**

---

## 15. 接下来最该做的 5 件事

1. **立刻定数据目录结构**
   不再摇摆全量 PNG / 只 MP4 这件事。
2. **抽 3–5 个场景做小实验**
   比较：
   * 8-bit PNG
   * JPEG(Q=95/97/98)
   * 是否套 DJI 官方 LUT
     看体积和视觉效果。
3. **先做主 benchmark 的 protocol**
   * 输入哪些视点；
   * 输出哪些视点；
   * train/val/test 怎么划。
4. **先做一个最小 baseline**
   不要一上来只盯 LFSTVSR。
5. **尽快确认毕业认定口径**
   尤其是 BMVC / ACCV 的 EI / 会议认定问题。

---

如果你愿意，下一步最适合做的是：
**我直接继续帮你写一版“数据集 README 模板 + benchmark protocol 草稿 + 论文摘要初稿”。**
