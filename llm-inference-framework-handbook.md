# 从一次请求到自己的推理框架

## DeepInfer、vLLM、SGLang 源码学习手册

**面向读者：**刚进入 LLM infra，希望不仅会部署，还能理解和开发推理引擎的人。懂基本 Python、数组和服务端概念即可；公式都配有解释。

**研究基线：2026-10-20，Asia/Shanghai。**本文是固定版本的文档与源码分析，附可在 CPU 上运行的教学实验；没有进行三个框架的 GPU/NPU 性能对测。项目能力与限制以本次快照为准，性能选型结论以“待验证的工程判断”呈现。

| 项目 | 本次版本 | 固定源码 SHA | 怎样理解这个版本 |
|---|---|---|---|
| vLLM | v0.32.0 | `5df63a7203cdb9328d824a27b682e77c7b9928be` | 2026-10-18 发布；本次核对时最新非预发布 release |
| SGLang | v0.5.22 | `baa9befa381511cd4d56f83980951e5a45fffc6d` | 2026-10-19 发布；本次核对时最新非预发布 release |
| DeepInfer | main 的固定快照 | `8b16149741a4935e1dec8cea27853f273b374c5b` | 不将 main 快照称作稳定发行版 |

版本证据：[V0](#src-v0)、[S0](#src-s0)、[D1](#src-d1)。后文源码链接固定到这些 SHA，避免学习过程中被 main 的变化误导。

> **新增负载均衡专题：**[直接阅读第 21 章](#ch21)：DeepSeek 官方部署说明、DualPath Decode 组内算法、阈值公式、逐步算例与跨集群路由启发。

## 怎样使用这份手册

你需要建立的能力有三层：**能解释一个 token 如何产生；能解释为什么并发请求会互相影响；能修改实现而不破坏正确性。**记住功能名不是目标。

- **先读一遍主线：**第 1～5 章。你会认识服务分层、模型计算、KV 内存和调度循环。
- **再对照项目：**第 6～8 章。带着“状态在哪里、谁修改它、什么时候释放”的问题看源码。
- **理解性能机制：**第 9～13 章。学习 kernel、并行、推测解码、P/D 和选型。
- **开始自己实现：**第 14～17 章。运行实验，按里程碑开发，再做正确性与性能验收。
- **查漏补缺：**第 18～20 章是练习、术语与证据索引。
- **负载均衡优先：**先读第 21 章，区分官方生产部署说明、DualPath 组内算法与跨集群路由，再回到第 4、10、12、13 章。

建议第一遍跳过较细的兼容限制，第二遍读源码时再回来核对。不要一开始尝试把所有优化组合打开。

## 目录

1. [先建立系统地图](#ch1)
2. [一个 token 怎样算出来](#ch2)
3. [KV cache：推理引擎的核心状态](#ch3)
4. [从单请求循环到连续批处理](#ch4)
5. [自己设计引擎时需要的对象与接口](#ch5)
6. [vLLM：从 token 预算看通用引擎](#ch6)
7. [SGLang：从请求、batch 与缓存树看运行时](#ch7)
8. [DeepInfer：从特定模型和硬件看协同优化](#ch8)
9. [内核、CUDA Graph 与异步执行](#ch9)
10. [多卡与 MoE：把一个模型拆开](#ch10)
11. [推测解码：一次验证多个未来 token](#ch11)
12. [P/D 分离：把两个阶段部署开](#ch12)
13. [三个项目的优劣与选型](#ch13)
14. [无需 GPU 的动手实验](#ch14)
15. [实现自己的框架：分阶段路线](#ch15)
16. [如何证明正确、如何证明更快](#ch16)
17. [按问题阅读源码](#ch17)
18. [自测题与参考答案](#ch18)
19. [术语速查与常见误解](#ch19)
20. [版本固定的资料索引](#ch20)
21. [负载均衡专题：官方部署说明与 DualPath 组内调度](#ch21)

<a id="ch1"></a>
## 1. 先建立系统地图

### 1.1 模型、引擎、服务平台分别是什么

一个模型主要给出数学计算和权重：给定前面的 token，下一 token 的概率分布是什么。推理引擎负责让这个计算在有限内存和设备上高效执行。推理服务还要负责 HTTP、排队、流式响应、取消、健康检查、弹性扩缩容等。

把它们画在一张图上：

```text
用户 / SDK
    │  对话、采样参数、超时
    ▼
API 与全局路由                  选模型、租户、区域/集群
    │
    ▼
集群内的实例 / P-D 路由         选哪个副本；必要时选 P 和 D 配对
    │
    ▼
Tokenizer / 请求管理           文本 → token IDs，跟踪请求生命周期
    │
    ▼
引擎 Scheduler                 本轮算哪些请求、各算多少 token
    │                  ↔ KV / 状态管理器
    ▼
Executor / Model Runner        组织张量、并行通信、调用模型
    │
    ▼
Attention / GEMM / MoE kernels  GPU/NPU 上的实际计算
    │
    ▼
Sampler → Detokenizer → 流式输出
    └──────── 更新状态，进入下一轮 ────────┘
```

这是一张教学分层图，不代表每个框架都用相同进程划分。模块可以同进程，也可以跨进程或跨机器。一个“模型实例”也可能由很多张卡共同组成。

### 1.2 四种“调度”不能混为一谈

| 决策 | 决策对象 | 典型信号 | 例子 |
|---|---|---|---|
| 跨集群分流 | 新请求去哪个集群 | SLO、可用容量、在途负载、区域约束 | 跨集群路由器 |
| 实例路由 | 去哪个服务副本 / P-D 配对 | 队列、缓存亲和、健康、目的端 KV 余量 | vLLM DP 路由、SGLang gateway 的相应功能 |
| 引擎迭代调度 | 本次 forward 放哪些 token | token 预算、KV、请求状态、图尺寸 | 三者各自的 scheduler |
| 专家布局与通信 | expert 放到哪张卡、token 发给谁 | 每 expert/rank 负载、通信拓扑 | EP / EPLB |

**上层路由器 跨集群路由与 DeepInfer 的 step 调度不是同一层。**“跨多机”也不自动等于“跨多个服务集群”：一个同步 EP 实例可能跨很多机器，却仍共同完成同一次模型计算。

vLLM 和 SGLang 的部署文档覆盖实例路由与分布式执行；DeepInfer 这次公开包明确未包含完整 P/D 副本负载均衡。不能把它们直接视作 上层路由器 的替代品。[V6](#src-v6)、[S6](#src-s6)、[D1](#src-d1)

### 1.3 三个项目为什么值得一起读

- **vLLM：**适合学习一个通用引擎如何定义请求、调度输出、KV 管理、执行器和扩展接口。
- **SGLang：**适合学习缓存复用、batch 生命周期、层级缓存与服务路径怎样联动。
- **DeepInfer：**适合学习当模型与硬件边界较明确时，调度、推测执行、内核和并行布局如何协同设计。

这是本文的学习顺序与阅读视角，不是能力的排他分类。vLLM 也有模型特化，SGLang 也有通用抽象，DeepInfer 也有请求与块管理。DeepInfer README 还明确说明其最初主要基于 vLLM，并受 SGLang、TensorRT-LLM 启发；它们不是三套毫无关联的理论。[V1](#src-v1)、[S1](#src-s1)、[D1](#src-d1)

<a id="ch2"></a>
## 2. 一个 token 怎样算出来

### 2.1 自回归不是一次把整段回答算出来

Tokenizer 把文本变成整数序列，例如 `[17, 82, 9]`。token 不必等于一个字或一个词。

模型输入这个前缀，输出词表中每个 token 的分数，称为 **logits**。Sampler 根据 greedy、temperature、top-k、top-p 等规则选出下一个 token。假设选出 `31`，接着用 `[17, 82, 9, 31]` 预测下一个。

```text
已知 prompt ──forward──> logits ──sample──> y1
prompt + y1 ──forward──> logits ──sample──> y2
prompt + y1 + y2 ──────> ...
```

因为 `y2` 依赖 `y1`，普通自回归不能直接独立并行生成任意未来 token。推测解码是绕过部分串行成本的方法，后面会讲，但仍需要主模型验证。

### 2.2 Transformer 中 Q、K、V 在做什么

只看一层、一个 attention head。每个位置的隐状态 `x` 乘上三个权重，得到：

```text
q = x Wq       本位置想查什么
k = x Wk       本位置提供怎样的匹配信息
v = x Wv       本位置提供什么内容
```

位置 `t` 的注意力计算可以写成：

```text
score[j] = dot(q[t], k[j]) / sqrt(head_dim),  j ≤ t
weight = softmax(score)
attention[t] = Σ weight[j] × v[j]
```

因果约束 `j ≤ t` 表示只能看自己和过去。真实模型还有多个 head、多层、位置编码、归一化、残差和 FFN/MoE。这里先保留与推理系统最相关的事实：**下一个位置需要过去位置的 K/V，但过去的 Q 通常不用留给未来位置。**

过去位置不能看未来，因此在模型权重、位置和上下文不变时，已经算出的历史 K/V 可以复用。这就是 KV cache 的基础。

### 2.3 Prefill 与 Decode

**Prefill：**处理已知 prompt，填充历史 K/V，并通常从最后一个 prompt 位置得到第一个输出 token。prompt 已知，所以多个位置的计算可以组织成大矩阵操作，因果 mask 保证不看未来。

**Decode：**每次把刚生成的 token 作为新输入，追加它的 K/V，得到下一 token。对于普通解码，每条活跃序列每轮通常只处理一个新位置。

一个容易错一位的例子：

```text
输入 a b c
Prefill(a,b,c) → 生成 d     KV 已包含 a,b,c；通常尚未包含 d
Decode(d)     → 生成 e     KV 已包含 a,b,c,d；通常尚未包含 e
```

所以“已输出 token 数”“已计算位置数”“已预留 KV 槽位数”不是同一个数。推测执行和异步调度会让它们更不一致。

### 2.4 为什么两个阶段的瓶颈不同

粗略看，长 prompt Prefill 的 attention 工作量随上下文长度平方增长；单个 Decode 新位置读取的历史 KV 随上下文长度线性增长。模型的线性投影和 FFN 又有不同的计算量。

长 Prefill 往往更容易形成较大的矩阵乘法，提高计算单元利用率；小 batch Decode 往往更受权重/KV 读取带宽和 kernel launch 成本影响。但不能把它们永久标签化为“Prefill 纯算力瓶颈、Decode 纯带宽瓶颈”：batch、MoE 通信、量化、长上下文、硬件都会改变瓶颈。

可以用一个分析下界理解，而不是当作精确预测器：

```text
step 时间 ≥ max(本轮 FLOPs / 有效算力, 本轮数据量 / 有效带宽)
```

实际还包括调度、同步、通信、数据搬运和无法隐藏的串行部分。这个下界说明：减少 FLOPs 不一定降低时延，减少带宽访问也不一定改善计算受限的 batch。

### 2.5 你必须会区分的时延

| 指标 | 含义 | 常见误读 |
|---|---|---|
| TTFT | 请求发出到收到第一个输出 token | 不只是某个 Prefill kernel 的时间 |
| ITL | 相邻输出 token 的间隔 | 网络 chunk 不一定一块一个 token |
| TPOT | 常见口径为 `(末 token 时间−首 token 时间)/(输出数−1)` | 平均值掩盖卡顿；输出数不足 2 时不适用 |
| E2E | 从发请求到请求结束 | 长输出自然更大，比较要控制长度分布 |
| 吞吐 | 单位时间完成请求或输出 token 数 | 不能省略质量、成功率和时延约束 |
| Goodput | 满足指定 SLO 的有效工作 / 时间 | 先定义什么算成功、哪些 SLO 同时满足 |

后文实验中的“轮次”不是毫秒。不同 batch 的每轮耗时不同。

<a id="ch3"></a>
## 3. KV cache：推理引擎的核心状态

### 3.1 先算一遍显存账

对常规 dense attention / GQA，忽略分页、对齐、共享、分片和临时空间：

```text
每 token KV 字节数 = 2 × 层数 × KV head 数 × head_dim × 元素字节数
总 KV 字节数 = 每 token KV 字节数 × 所有序列的已缓存 token 数
```

`2` 是 K 与 V。GQA 的 KV head 数可以小于 query head 数；别拿 query head 数代进去。

**教学例子，非任何项目的实测配置：**32 层、8 个 KV heads、head_dim=128、BF16 每元素 2 字节：

```text
2 × 32 × 8 × 128 × 2 = 131072 bytes = 128 KiB / token
一个 8192-token 的上下文约 1 GiB KV
32 条这样的序列约 32 GiB KV
```

这还没算权重、activation、通信 buffer、图捕获、draft 模型和碎片。权重约 `参数量×每参数字节数` 也只是起点；量化 scales、未量化层、临时张量会增加实际占用。

MLA、SWA、线性 attention / Mamba、跨层 KV 复用不能直接照套这条总公式。需要按各层实际缓存状态求和。并行后每卡占用也未必简单除以卡数，例如部分 KV heads 或状态可能复制。[V5](#src-v5)、[S4](#src-s4)、[D6](#src-d6)

### 3.2 为什么分页

朴素方案为每条请求分配“最大上下文长度”的连续内存。问题是输出长度事先不确定，预留会浪费，扩容可能搬运。

分页方案把内存切成固定大小的 block。逻辑上连续的 token，通过 block table 指向物理页：

```text
假设每页 4 token

请求 A： 逻辑页 0       逻辑页 1       逻辑页 2
            │             │             │
物理页：    #7            #2            #11
内容：     a b c d       e f g h       i _ _ _

token 6 → 逻辑页 floor(6/4)=1，页内偏移 2 → 物理页 #2 的偏移 2
```

这样新增 token 通常只需补页，不需要整条序列搬家。尾页仍可能有浪费；页太小则索引、分配和 kernel 访存管理成本增加。页大小是内存利用率和执行效率之间的取舍。

**Paged KV 管内存布局；Paged Attention kernel 负责按这个布局读取。**只有 Python 字典式 block table 并不会自动得到高性能 attention。

### 3.3 Prefix cache 与普通 KV cache 是两件事

- 请求内 KV cache：同一条请求的下一步不重复算历史。
- Prefix cache：另一条请求具有相同前缀时，复用已有的历史 KV，减少 Prefill。

例如两条 prompt 都以同一个长 system prompt 开头。第一条算过之后，第二条可能直接复用这些页。它主要减少重复输入计算，**不会使每个新的 Decode token 不再需要 attention**。

复用要求的不只是“文本看起来一样”。token 序列、模型/权重版本、位置语义、LoRA、多模态输入、缓存命名空间等都可能影响有效性。一个共享缓存系统必须明确缓存身份；不能让不同模型误用同一页。

### 3.4 vLLM 的 hash block 与 SGLang 的 radix tree

两者都能做 prefix caching，差异是组织方式。[V4](#src-v4)、[S3](#src-s3)

**vLLM 的主要路径：**每个满 block 用“父 block hash + 当前 block token + 额外身份信息”建立 key。查找时从前往后找可复用 block。父 hash 让相同 token 块在不同历史位置不会被错误视为相同上下文。

**SGLang 的 radix 路径：**把 token 前缀组织成压缩前缀树。共享前缀对应共享树路径；分叉处拆节点，沿路径查最长匹配前缀。节点持有对应缓存位置及锁定/淘汰信息。page size 仍约束可复用粒度，不能把“radix tree”理解成任意位置都可零成本复用。

```text
请求 1：A B C D X
请求 2：A B C D Y Z

radix：root ──[A B C D]──┬──[X]
                         └──[Y Z]

hash： h0 = hash(root, [A B C D], namespace)
       h1/h2 在共享父 h0 后分叉（示意，非实际 block 长度）
```

| 设计问题 | hash block 视角 | radix tree 视角 |
|---|---|---|
| 怎么识别前缀 | 链式 block 身份 | 共享树路径 |
| 怎么复用 | 匹配块并增加引用 | 匹配节点/位置并锁定路径 |
| 元数据复杂度来自哪里 | hash、block pool、请求映射 | 树分裂/合并、路径锁、节点状态 |
| 常见学习价值 | 内容寻址与物理页生命周期分离 | 前缀结构与淘汰/层级缓存联动 |

这些是结构差异，不足以单独推出速度排名。命中分布、CPU 开销、页大小、attention backend 和并发行为都要测。

### 3.5 共享、写时复制与回收

一个安全的最小设计需要以下约束：

1. 共享的已完成页不能被某个请求随意覆写。
2. 如果允许共享未满尾页，后续写入前要做 copy-on-write，或通过设计禁止这种共享。
3. 请求结束只释放它自己的引用；其他请求或缓存仍可能持有该页。
4. 可淘汰不等于已经清空。无活跃请求引用的缓存页可能保留内容直到被重用。
5. 异步 GPU 操作尚在读取时，即使 CPU 已收到 cancel，也不能立刻把页交给新请求。

注意实现细节：vLLM 的 block `ref_cnt` 表示活跃请求使用情况；零引用的缓存块仍可能存在于 cache map 和 free queue 中。教学实验采用更简单的“缓存本身也持一份所有权”模型，二者记账方式不同，不应逐字段对照。[V4](#src-v4)

### 3.6 层级缓存和混合状态

GPU KV 最快、也最贵。把不活跃前缀放到 CPU 内存甚至外部存储，可以换取更大容量；重新使用时要搬回来。收益条件是：**省下的重算时间，大于查找、传输、等待和争用增加的时间。**

SGLang HiCache 区分实例私有的 GPU L1、host L2 与可配置共享的 storage L3。L2 不因为容量大就变成全局共享缓存。vLLM 有 Offloading/第三方 KV connector 等路径，不能说它没有层级缓存。[S5](#src-s5)、[V7](#src-v7)

混合模型更难：full attention 需要整个历史；SWA 只保留窗口；Mamba 类状态有递推/检查点语义。统一缓存系统需要管理多种状态，而不是一个所有层通用的 `tokens × K × V` 数组。SGLang 此版本已有 `UnifiedRadixCache` 及 Full/SWA/Mamba 组件；vLLM 有 hybrid KV manager。阅读设计文档还要注意文档自己的历史版本说明，最终以固定源码为准。[S4](#src-s4)、[V5](#src-v5)

<a id="ch4"></a>
## 4. 从单请求循环到连续批处理

### 4.1 三种 batch

**静态 batch：**先凑一组请求，整组结束后再接下一组。短请求等长请求，设备也容易留下空位。

**连续批处理：**每轮都能移除完成请求、接入新请求，batch 成员随时间变化。请求之间的独立计算被放在同一轮执行，但每条序列自己的依赖仍需保持。

**Chunked Prefill：**长 prompt 分成多个 chunk，避免单次长 Prefill 长时间占住设备、拖慢已有 Decode。它改变调度粒度，不是截断输入；每个 chunk 仍然要正确读取前面已算的历史。

三者在这里是通用机制。当前三项目的具体实现不能用早年的“某项目只支持静态 batch”概括。[V2](#src-v2)、[S2](#src-s2)、[D2](#src-d2)

### 4.2 用同一组请求走四轮

设每轮最多处理 **6 个新输入位置**，先服务已有 Decode，再处理 Prefill，每条 Prefill 本轮至多 3 token。A、B 已在 Decode，C 新到达、prompt 长 6；下一轮 D 到达、prompt 长 3。忽略 KV 不足，假设 A/B 足够长。

| 轮次 | 安排 | 本轮 token 数 | 本轮结束的状态 |
|---|---|---:|---|
| 1 | A decode 1、B decode 1、C prefill 3 | 5 | C 已算 3/6，尚无首 token |
| 2 | A 1、B 1、C prefill 3、D prefill 1 | 6 | C 完成 Prefill，产生首 token；D 1/3 |
| 3 | A 1、B 1、C decode 1、D prefill 2 | 5 | D 完成 Prefill，产生首 token |
| 4 | A 1、B 1、C 1、D decode 1 | 4 | 四条持续生成 |

表中的 token 数是**本轮计算预算**，不是活跃请求总历史长度，也不是本轮输出数。Prefill 3 个位置通常不会向用户输出 3 个回答 token。

为什么第一轮不把 C 的剩余 1 个预算也用上？因为这个例子规定 Prefill 每请求 chunk 上限为 3。这说明 token budget、per-request chunk、max sequences 是不同约束，可能让预算有剩余。

这不是任何一个框架默认策略的精确复现，是帮助你推演状态的教学规则。

### 4.3 token budget 之外还有哪些约束

一个可执行的 batch 至少同时满足：

```text
本轮计算 token 总数 ≤ token budget
活跃序列数 ≤ sequence budget
需要的 KV 页 + 在途预留 ≤ 可用/可回收页
模型长度、grammar、LoRA、并行与 backend 等兼容条件成立
```

1 个 Prefill token 与 1 个长上下文 Decode token 耗时不相等，所以 token budget 是便于实现的控制量，不是精确成本模型。

KV 不够时可以延后接入、驱逐闲置前缀、回收并重算某些请求、offload，或拒绝过载请求。不同选择分别影响 TTFT、ITL、吞吐和网络/存储成本。所谓 preemption/retraction 通常不是免费暂停：恢复可能需要重算。

### 4.4 吞吐、公平性和尾延迟为什么冲突

更大的 batch 常能摊薄权重读取和 launch 开销，但单轮时间也可能变长。优先缓存命中的请求可能提高整体吞吐，却让冷请求等待更久；一直优先 Decode 可保护在途请求，但 Prefill 可能饥饿。

一个可落地的调度策略要声明：服务目标是什么、允许谁等待多久、过载时如何退化。不要把“GPU 一直忙”视作充分正确的目标。第 16 章会把这些变成可以测的验收条件。

<a id="ch5"></a>
## 5. 自己设计引擎时需要的对象与接口

以下是教学接口，并不是从某一项目复制的 API。

| 对象 | 最少需要的信息 | 容易犯的错 |
|---|---|---|
| Request | request_id、输入、采样参数、期限、状态 | 混淆请求终止和 GPU 工作完成 |
| Sequence | token IDs、已算位置、输出位置、块映射 | 把 emitted/computed/reserved 合成一个数 |
| Scheduler | waiting/running、预算、公平规则 | 先改状态再发现分配失败，却不回滚 |
| KV manager | 页池、引用、命中、预留、释放 | 取消时提前释放在途页 |
| Batch plan | 请求/位置/长度/块表/执行模式 | Runner 重排后结果映射错请求 |
| Runner | 模型、device buffers、kernel/graph、通信 | 输入元数据与实际 KV 布局不一致 |
| Sampler | logits 变换、随机数、grammar/stop 状态 | 推测回滚后 grammar 或 RNG 不一致 |
| Output processor | token→文本、增量输出、完成事件 | Unicode/stop 跨 token 边界处理错误 |

可以先写同步主循环：

```python
while has_work():
    ingest_requests_and_cancellations()
    plan = scheduler.plan(kv_manager)   # 选择、检查、预留；明确事务边界
    if plan.empty:
        wait_for_input_or_resources()
        continue
    result = runner.execute(plan)       # 使用计划指定的 token 与块表
    scheduler.commit(plan, result)     # 前进位置、处理输出/结束/回退
    release_resources_when_safe()
    emit_outputs()
```

逐个回答这几个问题，你就开始具备读引擎源码的能力：

- `plan` 创建后，请求可能被取消吗？谁识别这个变化？
- 执行失败，已经预留的页怎样撤销？
- CPU 提交的是第 n+1 轮，用户刚取消第 n 轮所属请求，哪一轮结果还允许输出？
- 一个 token 已经被采样但尚未写 KV，下一轮从哪个位置开始？
- batch 被重排、压缩或 padding 后，如何还原 request_id？

真实框架的难点往往就在这些状态边界，而不是 `model.forward()` 这一行。

<a id="ch6"></a>
## 6. vLLM：从 token 预算看通用引擎

### 6.1 先画出调用主线

下面省略了 API server 的众多入口和各硬件特化：

```text
请求入口 / AsyncLLM
        │
        ▼
EngineCore / EngineCoreProc
        │
        ├─ Scheduler.schedule()
        │      └─ KVCacheManager → BlockPool
        │
        ├─ model_executor.execute_model(SchedulerOutput)
        │      └─ Worker → GPUModelRunner → model / attention / sampler
        │
        └─ Scheduler.update_from_output()
               └─ 输出增量、完成/取消、回收
```

`EngineCore.step()` 源码可以直接看到 schedule、非阻塞 execute、处理 abort 和 update 的顺序；`step_with_batch_queue()` 则引入在途 batch 队列。不是所有模式都走完全相同的执行分支。[V2](#src-v2)

### 6.2 最值得学习的抽象：尚未计算的 token

固定版本的 scheduler 注释明确表达：核心调度不必用“Prefill 阶段 / Decode 阶段”二分来描述每条请求，而是追赶：

```text
num_computed_tokens
        → num_tokens_with_spec
```

后者包含 prompt、输出和 speculative token。实际实现还要考虑 output placeholders、lookahead、长 prompt chunk 阈值等，不能把上面两个量之差当作全部代码。

这个抽象的好处是统一解释：

- 新请求：很多 prompt 位置尚未计算。
- 普通 Decode：新生成的一个位置尚未计算。
- Prefix hit：前面的一部分已经有可复用计算。
- Speculation：还有多个候选位置待验证。

**“统一调度计数”不表示底层 kernel 不区分执行类型，也不表示每 token 成本相同。**执行层仍需要不同 attention 元数据、graph 与采样处理。[V3](#src-v3)

### 6.3 KV 管理器怎样与 scheduler 配合

`get_computed_blocks()` 查已算好的前缀；`allocate_slots()` 处理复用与新增槽位。调度器必须确保 batch 的计算预算和可分配的 KV 同时满足。分配失败可能触发延后或 preemption。

`BlockPool` 管物理块、引用和可回收队列。把“请求该算什么”与“物理块怎么管理”拆开后，scheduler 不必直接操作 GPU KV tensor。这是自己设计通用框架时很值得学习的边界。[V4](#src-v4)

### 6.4 V1 engine 与 Model Runner V2 不要混淆

本版本源码仍在 `vllm/v1/` 下，同时存在：

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/model_runner.py`，其中有 Model Runner V2 相关实现。

不要根据目录名字推断“V1 engine 已经整个被 V2 替换”，也不要只读旧路径就认为新版没有新执行路径。实际选择受版本配置与支持范围约束；阅读时先从运行入口追到实际实例化的 Runner。[V10](#src-v10)

### 6.5 优势与代价

**优势：**通用请求/执行边界、较广的模型和硬件接口、KV connector 等扩展位置，适合研究如何让一个引擎容纳多种模型和部署方式。[V1](#src-v1)、[V7](#src-v7)

**代价：**灵活性带来兼容矩阵。一个 connector 支持的 TP/PP、KV layout、hybrid state、speculation 组合不一定适用于另一个。抽象接口减少局部耦合，不会消除跨模块的正确性依赖。

**我的学习建议：**第一次只追同步的普通文本生成：`EngineCore → Scheduler → KVCacheManager → Runner → update`。先理解不带 connector、不带 speculation 的闭环，再加一个变量。

<a id="ch7"></a>
## 7. SGLang：从请求、batch 与缓存树看运行时

### 7.1 先区分 SGLang 的不同部分

SGLang 不只是一个用于编排生成的前端语言。本文重点是其服务运行时 SRT，以及与它配合的 gateway、缓存和模型执行。也不要把整个项目简单缩写为 RadixAttention；radix cache 是理解其运行时的一条入口。[S1](#src-s1)

常见服务路径可以这样看：

```text
HTTP / OpenAI-compatible frontend
          │
          ▼
TokenizerManager            输入处理、请求跟踪
          │
          ▼
Scheduler                   接入请求、选择 batch、处理结果
          ├─ Req / ScheduleBatch
          ├─ prefix cache / KV pool / HiCache
          └─ TP worker / ModelRunner → 模型与 kernels
          │
          ▼
DetokenizerManager          token 增量转文本
          │
          └─ 返回到请求管理与流式响应
```

实际进程、IPC 和并行拓扑依配置变化。上图画的是职责关系，不是对每一种模式的进程数量承诺。[S2](#src-s2)、[S10](#src-s10)

### 7.2 从 event loop 看设计

`event_loop_normal()` 的核心顺序很直接：

```text
ingest_requests
→ get_next_batch_to_run
→ run_batch
→ process_batch_result
```

`get_next_batch_to_run()` 会处理已有 batch、新 Prefill、缓存相关事件与同步要求。`ScheduleBatch` 明确承载执行阶段和序列状态；这与 vLLM scheduler 注释强调统一 token 进度的表达方式不同。

但不能把差异夸大为“vLLM 能混合调度，SGLang 只能把 P/D 完全隔开”。SGLang 有 chunked/mixed batch 等路径；应按所选配置追实际分支。[S2](#src-s2)

### 7.3 缓存树影响的不只是命中率

一个请求进入 Prefill 前，查到的可复用前缀会减少剩余输入工作，并影响是否能接入。活跃请求锁住的路径不能像闲置缓存一样随意淘汰。请求结束后，部分数据还能留给后续请求使用。

因此“排谁、能排多少、淘汰谁”会互相影响。你阅读 `match_prefix / insert / evict / inc_lock_ref / dec_lock_ref` 时，要同时观察 scheduler 何时调用它们。[S3](#src-s3)

最新代码还包含 `UnifiedRadixCache` 和 Full/SWA/Mamba 等组件。不要把网上早期教程里的单一 radix tree 当成所有模型当前的完整缓存实现。是否采用哪个 cache class，需要看模型和运行配置。[S4](#src-s4)

### 7.4 overlap event loop 的实际依赖

`event_loop_overlap()` 会先启动当前 batch 的设备工作，再处理上一 batch 的结果；当前 batch 的采样在必要的上一轮状态处理之后进行。源码有 result queue、batch copy，以及读写依赖的 barrier。

这展示了一个关键思想：**隐藏 CPU 工作需要管理多个不同时间点的状态，而不只是开一个线程。**grammar 等状态可能依赖上一轮输出，不能为了 overlap 让它读取错误版本。[S2](#src-s2)

### 7.5 优势与代价

**优势：**缓存复用、HiCache、P/D、实例路由的相关文档与实现联系较直接，适合学习这些功能怎样组成完整服务路径。其模型、硬件、LoRA、多模态覆盖也很广，不应视作只支持某种 DeepSeek 部署。[S1](#src-s1)、[S5](#src-s5)、[S6](#src-s6)

**代价：**缓存树、分层传输、混合状态和 speculation 发生组合时，状态同步很复杂。能单独开启某个功能，不代表该功能和所有其他优化都兼容。

**我的学习建议：**先跑通 `Req → ScheduleBatch → event_loop_normal`；再沿一个 prefix hit 追引用增减；最后才看 overlap 和 HiCache。不要从最复杂的 MoE + P/D + DSpark 组合起步。

<a id="ch8"></a>
## 8. DeepInfer：从特定模型和硬件看协同优化

### 8.1 公开包的范围是比较的前提

此快照 README 列出的支持是 DeepSeek-V4.1-Flash、NVIDIA SM100 与 Ascend 950。它开放了离线推理核心，包括模型、kernels、并行、推测验证、异步调度等。

README 同时列出完整推理服务仍需补充的 KV cache system、VE/Prefill/Decode disaggregation 和 P/D DP 副本负载均衡；引擎已经具备其中不少底层能力。**源码有本地 KV/block 管理，不能把这句话翻译成“DeepInfer 没有 KV cache”。**它欠缺的是完整服务层系统，而不是完全不存 K/V。[D1](#src-d1)、[D2](#src-d2)

### 8.2 调用主线

```text
examples/offline_inference.py
          │
          ▼
Python LLMEngine
          ├─ 请求/输出管理、step、abort、延迟释放
          ├─ C++ Scheduler / Sequence / BlockManager
          └─ Worker.execute_model_step
                  ├─ ModelRunner / 模型 / 定制 kernels
                  ├─ MTP/DSpark 与 sampler
                  └─ DP/CP/EP、EPLB
```

源码阅读锚点：`deepinfer/engine/llm_engine.py`、`csrc/engine/scheduler.hpp`、`block_manager.hpp`、`mtp_varlen.hpp`、`deepinfer/worker/worker.py`。[D2](#src-d2)、[D3](#src-d3)

C++ 实现调度可能减少部分 Python 对象和解释器成本，但“用了 C++”本身不是性能结论。若关键路径是大规模通信或设备计算，换语言未必解决主要瓶颈。

### 8.3 异步调度的具体顺序

Worker 中的主要顺序是：

```text
本轮 main model forward
→ 上一轮 post-process
→ 本轮 sample
→ 本轮 post-process 立即进行或延后到下一轮
```

配合 GPU 端长度修正、事件同步与延迟释放，把部分 CPU 工作放进设备计算期间。这和 SGLang overlap、vLLM 异步执行要解决的瓶颈有共性，正确性机制各有实现。

`zero-overhead` 是项目对该机制的命名，不代表所有负载的 CPU 成本实测为零。[D4](#src-d4)

### 8.4 三个最值得深入的机制

**置信度验证预算。**`MTPVarlen` 根据 confidence 估计“多验证一个候选 token 能带来多少有效输出”，再结合设备 step 时间选预算。它包含启动基线和在线耗时修正，不只是固定 nextn。第 11 章详细展开。[D3](#src-d3)

**逐 step 的专家布局。**EPLB 不是只追求长窗口总 token 平均分配；policy 利用逐 step 的负载向量，尝试减小同步执行中最忙 rank 导致的等待。热点 expert 可以有多个 physical 副本，代价是权重显存与重排搬运。[D5](#src-d5)

**SWA bounded replay。**对于兼容的 YoCo 模型配置，减少 decoder consumer 部分重复 Prefill 的范围；保留的后缀要覆盖滑窗与回退需求。示例推荐 140 token，不代表整个长 prompt 只需计算最后 140 token，也不能移植到任意 full-attention 模型。官方部署文章进一步说明 V4.1 的 SWA bounded replay 包含近似恢复，不能将其笼统描述为所有 attention 状态的精确重建；见第 21 章引用的官方来源。[D6](#src-d6)

### 8.5 优势与代价

**优势：**模型和硬件边界明确，容易看到“模型结构 → 缓存布局 → kernel → 图尺寸 → scheduler”的完整协同，而不是把每个优化孤立理解。

**代价：**适配范围较窄、完整服务栈需要补齐，里面某个高效实现可能依赖专用算子、checkpoint 或并行拓扑。你学到的应是约束与优化目标，不是把所有文件拷到通用引擎里。

**学习顺序：**等你已理解普通 KV、连续 batch 和 speculative verify 后，再读 `mtp_varlen.hpp` 与 `worker.py`，收获会明显更大。

<a id="ch9"></a>
## 9. 内核、CUDA Graph 与异步执行

### 9.1 为什么框架需要自定义 kernel

Python 调用一个算子只是提交工作。真正决定设备效率的因素包括：如何分块、怎样从 HBM 搬到片上存储、哪些中间结果可以不写回、不同计算能否融合，以及并行通信能否和计算重叠。

FlashAttention 类方法减少 attention 中间矩阵的显存读写；Paged Attention 支持分页 KV 访问。二者解决的问题不同，可以在同一个 attention backend 中结合。量化 GEMM、MoE dispatch/combine 和采样也都有各自优化空间。

三个框架都有执行后端与特化 kernels，不能只看 Python scheduler 给它们打分。DeepInfer 的 README 列出了 FlashMLA、DeepGEMM、DeepEP 等依赖；vLLM/SGLang 的后端也会随模型与硬件选择。[D1](#src-d1)、[V1](#src-v1)、[S1](#src-s1)

### 9.2 CUDA Graph：少做重复提交

Decode 每轮往往调用相似的一组算子。CUDA Graph 把可重复的设备执行结构捕获后再 replay，减少 CPU launch 开销。它不是编译一个“任何 batch 都能随意变化”的通用执行图。

动态 batch 通常需要若干预捕获尺寸与 padding：真实 11 条请求可能使用更大的图档位。相应代价包括：图准备时间、内存、额外 padding 工作和支持范围限制。

这使 scheduler 需要知道“选择更多 token 是否跨入更贵的图档位”。动态推测预算如果只看接受率、不看 graph 阶梯成本，可能作出错误选择。

CUDA Graph 是 NVIDIA 机制；不能直接用其名词描述 Ascend 的实现。跨硬件可以讨论减少 launch/sync 和固定执行结构的共同目标，但实现要分别验证。

### 9.3 异步不是越多越好

```text
同步（示意）：
CPU:  [计划 n]              [收尾 n][计划 n+1]
GPU:          [执行 n]                         [执行 n+1]

重叠（示意）：
CPU:  [计划 n] [准备后续工作/收尾上一轮] ...
GPU:          [执行 n                  ][执行 n+1 ...]
```

重点是减少设备等待 CPU 的空隙。必须同时处理：

- CPU 提前预测的序列长度与设备实际接受长度是否一致；
- block table 和 batch metadata 是否被下一轮覆写；
- sampler 的 grammar、RNG 与 stop 状态读哪一轮版本；
- 已取消请求的在途输出是否丢弃；
- GPU 读完前的资源是否仍被 pin 住。

因此写自己的框架时，**先做同步参考实现，再做 overlap，并用参考实现检查输出与资源生命周期**。吞吐提升不足以证明异步实现正确。[V2](#src-v2)、[S2](#src-s2)、[D4](#src-d4)

### 9.4 精度与确定性

低精度权重和 KV 可以省内存、减少带宽，但支持格式、缩放、校准和 kernel 不匹配会增加转换成本或损伤质量。不能只比较“都是 FP8”而忽略具体格式与量化方法。

确定性又是另一维度：不同 batch 组成、reduction 顺序、随机数消费、推测路径都可能改变输出。DeepInfer 将 batch invariance 与 deterministic sampling 分开，后者有固定 seed、关闭 MTP_VARLEN 等条件；SGLang 也有 backend/缓存路径限制。[D1](#src-d1)、[S9](#src-s9)

学习时把三件事分开验收：数学分布是否正确、给定参考条件下数值误差是否可接受、是否要求逐 token 可复现。

<a id="ch10"></a>
## 10. 多卡与 MoE：把一个模型拆开

### 10.1 先记住每种并行拆什么

| 名称 | 拆分对象 | 收益 | 主要代价 |
|---|---|---|---|
| TP，Tensor Parallel | 一层的矩阵/head 等 | 单层分摊显存和计算 | 层间/层内频繁通信 |
| PP，Pipeline Parallel | 不同层放不同设备 | 模型容量扩展 | pipeline bubble、阶段不均衡、调度复杂 |
| DP，Data Parallel | 数据/batch 或独立实例 | 提高总服务能力 | 路由、负载不均；MoE 下未必完全独立 |
| EP，Expert Parallel | 不同 expert 放不同设备 | 容纳和并行执行 MoE experts | all-to-all、热点、同步等待 |
| CP，Context Parallel | 同一序列的上下文计算 | 长上下文容量/计算扩展 | attention 所需信息交换 |

这是概念表。一个框架的 DP/CP flag 可能只影响某些模型阶段，不能仅根据名字推断所有 tensor 的布局。实际 TP、DP、EP 有时复用同一组设备，不应把所有数字随意相乘计算总卡数。[V6](#src-v6)、[V9](#src-v9)、[S8](#src-s8)、[D1](#src-d1)

### 10.2 最容易误解的 DP

**独立 dense replicas：**每个副本完整执行自己的请求，A 忙通常不会要求 B 做同一次 collective。

**DP attention + EP MoE：**attention 处理各自的请求，后面的 MoE 却要把 token 发给共同的 expert ranks。这里的 DP ranks 是耦合执行的。某个 rank 没有真实请求，也可能需要参与 dummy forward，保证通信顺序一致。

这解释了为什么“给每个 rank 随机选择一个不同大小的验证图”可能破坏分布式执行；也解释了为什么 replica routing 和 rank 内 workload balance 要分开考虑。vLLM 的 DP 文档明确讨论了这种耦合。[V6](#src-v6)

### 10.3 MoE 怎样产生慢 rank

MoE 的 router 为 token 选择少量 experts。假设四个 rank 本轮分别得到 `[100, 95, 90, 220]` 单位工作。若它们需要在阶段末同步，其他 rank 会等 220 的那一个。

EPLB 可以把热点逻辑 expert 复制成多个物理副本或重排布局，让未来工作更均衡。它不是把用户请求重新分配到另一个数据中心。

还要注意“平均均衡”可能骗人：两步负载 `[200,0]` 与 `[0,200]` 的长窗均值是 `[100,100]`，但每步都等一个忙 rank。与每步 `[100,100]` 相比，同步成本可能很不同。这个教学例子解释了 DeepInfer 为什么关注逐 step 最忙 rank 的目标。[D5](#src-d5)

### 10.4 三者有什么不同

vLLM 和 SGLang 都有 EP/EPLB、通信后端和 overlap 路径；DeepInfer 在固定模型与硬件上组合 Attention DP、balanced CP、EP/EPLB。不能用“谁支持 MoE”来区分这几个版本。[V9](#src-v9)、[S8](#src-s8)、[D1](#src-d1)

实际评估要看：完整实例拓扑、网络、token 分布、每 rank 峰值、冗余 expert 显存、迁移频率、通信与计算占比。单看总 GPU 利用率无法判断负载均衡是否有效。

<a id="ch11"></a>
## 11. 推测解码：一次验证多个未来 token

### 11.1 Draft → Verify → Commit

一个更便宜的 drafter 先猜未来若干 token，target model 对这些候选进行并行验证，接受其中一个前缀，拒绝后修正，然后继续。

```text
已知前缀 P
Draft:   [a, b, c, d]
Verify:   ✓  ✓  ✗
Commit:  [a, b, 修正 token]
丢弃：未被接受分支上的后续 KV / grammar / draft 状态
```

主模型一次处理多个已知候选位置，可能比逐轮单位置处理更高效；但 drafter、验证、采样和回滚都有成本。高 batch 已充分占满设备时，speculation 未必赚钱。

### 11.2 为什么不能只说“猜对就收，猜错就丢”

对 greedy decoding，比较主模型 argmax 可以给出简单直观的验证。但随机采样要保持 target 分布，需要更严格的接受/修正规则。

经典线性 draft rejection sampling 中，候选由 `q` 产生，target 分布为 `p`，位置接受概率是：

```text
min(1, p(x) / q(x))
```

拒绝时从归一化的 `max(p−q, 0)` 修正分布采样；全部接受时还能从相应 target 条件分布产生 bonus token。这个结论有前提：正确使用相应条件上下文、精确记录采样分布并执行完整修正流程。不同树式/特化实现不能只套一行公式就宣布正确。[F1](#src-f1)

temperature、top-p、grammar 等改变了实际分布，必须纳入算法定义。动态裁剪验证长度也不能窥视未来信息而改变应该保留的分布。工程上要一起检查 RNG、KV 和约束状态的回滚。

### 11.3 MTP、EAGLE、DSpark 与 speculative decoding 的关系

Speculative decoding 是一类执行方法；MTP、EAGLE、DSpark 等提供不同的 drafter 或配套结构。支持某种 MTP checkpoint 不代表自动拥有高接受率，也不代表支持任意动态验证。

对于初学者，先区分三条独立的轴：

1. **谁来猜？**独立 draft model、模型自带预测模块等。
2. **猜什么结构？**线性 token 链、树、块。
3. **算多少？**固定 K、按 batch-size 选 K、逐请求/逐位置动态预算。

只比较一个 `nextn` 参数会混淆这三件事。[V8](#src-v8)、[S7](#src-s7)、[D3](#src-d3)

### 11.4 用一组数理解动态预算

令 `s[i,k]` 表示请求 i 的前 k 个候选全部接受的概率，即 survival probability。它不是第 k 个 token 单独正确的概率。

忽略 EOS、截断等边界，在每请求有一个主/bonus 输出的简化模型中：

```text
预计输出数 ≈ 请求数 B + Σ_i Σ_(k≤K_i) s[i,k]
效率 ≈ 预计输出数 / 本轮总耗时
```

假设两条请求，最多各验证 3 个候选：

```text
A 的 survival = [0.9, 0.8, 0.7]
B 的 survival = [0.6, 0.2, 0.1]
```

若本轮只分配 3 个额外候选预算，优先给 A 的三个位置可得到预计 `2+0.9+0.8+0.7=4.4` 输出，而给 A 两个、B 一个是 `4.3`。这只是局部分配示例；公平性、上限、图约束都可能改变实际选择。

再假设有三种测得的执行成本（纯教学数字）：

| 额外验证预算 | 预计输出 | 本轮时间 | 预计 token/s |
|---:|---:|---:|---:|
| 0 | 2 | 4 ms | 500 |
| 3 | 4.4 | 6 ms | 733 |
| 6 | 5.3 | 10 ms | 530 |

最多验证不等于最高有效吞吐。这也是为什么 confidence 与成本模型要一起用。

### 11.5 三个版本的具体差异

| 路径 | 依据什么调整 | 调整对象 | 必须知道的边界 |
|---|---|---|---|
| DeepInfer MTPVarlen | confidence survival、基线图耗时、在线 forward/sampler 修正 | 总验证预算及各序列长度 | 模型/硬件路径特定；涉及异步历史与采样正确性 |
| vLLM Adaptive Verification | confidence 或在线接受率估计、启动时图成本 | 请求/位置的验证预算 | 默认关闭；full CUDA graph、backend 条件；无 PP/LoRA 等限制 |
| vLLM Dynamic SD | 用户配置的 batch-size → K 区间 | batch 对应的 draft 长度 | 与 Adaptive Verification 不同；此路径 DP>1 会回退静态 |
| SGLang EAGLE/EAGLE3 adaptive | 按 batch-size 桶的 accept-length EMA，滞回/更新周期 | 预捕获 step 档位 | 当前要求 topk=1；不是逐请求成本最优分配 |
| SGLang DSpark ragged verify | confidence survival + SPS 表/时间成本 | 总预算与逐请求 verify length | 非静态策略依赖 confidence；无有效 profile 时可能退化 verify-all |

依据：[D3](#src-d3)、[V8](#src-v8)、[S7](#src-s7)。表中限制属于各自路径，不能从一行搬到另一行。

vLLM 没有 confidence head 时存在从 draft logits 学习接受率的路径；这不意味着任意模型/后端都无条件可用。SGLang 的 DSpark 动态图协调也有具体 DP/PP/P-D 条件。最新三者已有交叉机制，不能继续使用“DeepInfer 独有自适应、另外两个都是固定 K”的旧比较。

### 11.6 哪些收益数字不能直接拿来用

正确的分子是实际有效输出，不是生成了多少 draft、验证了多少候选。DeepInfer 示例 benchmark 的原始 Decode 计数含全接受假设，绘图还可使用外部 acceptance 缩放；示例同时有强制负载均衡和计时过滤条件。它适合解释引擎能力边界，不能单独证明真实服务的 accepted-token 吞吐。[D7](#src-d7)

你自己的验收应同时记录：draft 数、verified 数、accepted 数、实际 emitted 数、各阶段耗时和资源。每个指标的“+1 主/bonus token”口径都要写清。

<a id="ch12"></a>
## 12. P/D 分离：把两个阶段部署开

### 12.1 为什么想把它们拆开

Prefill 与 Decode 的 batch 形态、瓶颈和时延目标不同。放在同一个实例上，长 Prefill 可能干扰已有请求的 ITL；分开后可以独立调整 P 与 D 的资源比例和执行配置。

但分开后 KV 必须从 P 到 D，且两边要协调请求状态。收益应覆盖网络、buffer、协议、排队和额外资源成本。**P/D 的价值常是更好地满足 TTFT/ITL 目标，不是自动增加同样资源下的原始吞吐。**vLLM 的 P/D 文档也明确提醒吞吐不一定提高。[V7](#src-v7)

### 12.2 跟踪一条请求的状态

下面是一条概念路径；不同 connector 的 push/pull、chunk 传输和首 token 归属会不同。

```text
ROUTED
  → P_QUEUED
  → P_RUNNING
  → KV_READY / TRANSFERRING
  → D_READY
  → D_RUNNING
  → FINISHED / ABORTED
```

每个转换都要有明确的所有权：

| 时刻 | 必须回答的问题 |
|---|---|
| 选中 P、D | D 不健康或容量不够，谁重试、谁取消已发给 P 的工作？ |
| P 计算中 | D 是否已经为尚未到达的 KV 预留了空间？ |
| 传输中 | P 的源页何时可释放，D 的目的页何时可被 attention 读取？ |
| 传输完成 | 数据、block metadata、位置、模型版本是否一致？ |
| 用户取消 | 两边是否都收到终止，重试是否会重复释放或重复输出？ |
| worker 故障 | 重算还是失败？旧传输完成事件能否误写新请求的页？ |

共享 KV 的格式不是只有一个 byte buffer：layer 顺序、TP 分片、KV 精度、页布局、模型状态都必须一致或有受支持的转换。

### 12.3 提前预留还是 Prefill 完成后预留

**提前预留 D 空间：**能尽早确认目的容量，可能重叠 Prefill 与 chunk transfer；但长 Prefill 尚未完成时，D 的 KV 空间就被占住。

**晚预留：**P 完成后再争取 D 空间，减轻 D 的提前占用；但可能让 P 更久保留结果，增加等待，损失部分传输 overlap。

SGLang v0.5.22 新增 opt-in `prefill_complete` decode allocation policy，正是这个取舍。它有很窄的组合边界：Mooncake/Python bootstrap、attention TP/CP=1、PP=1 等；与若干 staging、role switching、Decode HiCache 等模式不兼容。所有通信 worker 要同版本、同策略。不要把它当成所有 P/D 部署都能开的选项。[S6](#src-s6)

release 中该策略约 26.8% throughput、48.9% p95 TTFT 改善来自指定 GLM-5.2-NVFP4 的 Decode KV 压力场景，是同一路径下策略变化的实验，**不是 SGLang 对 vLLM 的跑分**。[S0](#src-s0)

### 12.4 三个项目的切入点

- vLLM：从 KV connector 合约和具体 backend 的兼容矩阵切入；本版本 P/D 文档仍标 experimental。
- SGLang：从 scheduler 的 P/D event loop、bootstrap、传输、gateway 与 HiCache 联动切入。
- DeepInfer：从引擎服务 hooks 和执行机制切入；公开包并未给出完整可直接部署的 P/D 服务系统。

依据：[V7](#src-v7)、[S6](#src-s6)、[D2](#src-d2)。它们均不等价于跨集群全局容量调度。

<a id="ch13"></a>
## 13. 三个项目的优劣与选型

### 13.1 一个有用的比较表

下面的“优势/代价”是基于前述架构的工程判断；没有用未运行的 benchmark 给性能排名。

| 维度 | vLLM v0.32.0 | SGLang v0.5.22 | DeepInfer 固定快照 |
|---|---|---|---|
| 公开范围 | 通用推理与服务框架 | 通用推理运行时及缓存/路由相关能力 | 特定模型/硬件的离线推理核心 |
| 读代码的抓手 | 请求 token 进度、SchedulerOutput、KV manager、Executor | Req/ScheduleBatch、event loop、radix/统一缓存 | LLMEngine、C++ scheduler、Worker 与专用执行路径 |
| Prefix/cache | hash block、hybrid 管理、connector | radix/组件化混合缓存、HiCache | 本地块/状态管理；完整 KV 服务需补齐 |
| 推测验证 | 多 drafter 与 adaptive verification，另有 Dynamic SD | EAGLE adaptive 与 DSpark ragged 等不同路径 | DSpark/confidence 与在线时间修正结合 |
| P/D | connector 扩展，逐项核对兼容性 | 较直接的 P/D、gateway、HiCache 集成路径 | 服务集成边界开放，完整系统未包含 |
| MoE/并行 | EP/EPLB、DP/TP/PP/CP 等依配置支持 | EP/EPLB、DP/TP/PP/CP 等依配置支持 | 明确的 Attention DP/balanced CP、MoE EP/EPLB |
| 最值得学 | 可扩展引擎边界与统一调度抽象 | 缓存、调度和服务生命周期的联动 | 模型、硬件与执行策略的协同 |
| 主要开发成本 | 扩展点多，组合兼容和通用抽象复杂 | 状态路径多，缓存/异步/推测组合复杂 | 适配范围与完整服务能力需要自己扩展 |

各行证据在第 6～12 章及资料索引。这个表不意味着某项目在其他列缺少能力。

### 13.2 按需求作决定

| 你的主要目标 | 优先阅读/验证 | 为什么 |
|---|---|---|
| 学会设计一个通用最小引擎 | vLLM 主循环，再对照 SGLang | 先建立请求、KV、计划与执行边界 |
| 多轮对话、大量公共前缀、长文档复用 | SGLang radix/HiCache 与 vLLM prefix/connectors 同测 | 缓存收益依赖分布、搬运成本和路由亲和 |
| 多模型/多硬件/多 LoRA 的平台 | 先建立 vLLM 通用基线，同时查 SGLang 所需组合 | 覆盖和兼容性比抽象吞吐排名更先决 |
| 固定大 MoE、追求具体拓扑下的效率 | 两个通用框架实测；DeepInfer 用于机制对照 | 最新版本均有大量 MoE 与模型特化优化 |
| 学习调度/内核/模型共同设计 | DeepInfer 的 Worker、MTPVarlen、EPLB | 优化目标和模型假设较集中可见 |
| 打造线上 P/D 服务 | 先跑通 vLLM 或 SGLang 受支持组合 | 服务状态机、失败恢复和兼容矩阵是主要工程量 |

我的建议是：**开发入门以一个正确的单卡同步引擎为主线；读 vLLM 学边界，读 SGLang 学缓存与生命周期，读 DeepInfer 学如何把性能优化做深。**不需要一开始决定“只站一个框架”。

### 13.3 什么时候值得自己做

可以先区分“教学框架”和“生产框架”：

- 为学习做自己的最小引擎很有价值，你会真正理解 KV、调度与采样。
- 为生产从零重写，应有明确无法通过现有扩展点解决的需求，例如特殊模型状态或硬件执行限制，以及维护投入。
- 如果只是某个模型缺一个 kernel，或者某种负载需要不同准入策略，先在现有框架中做有边界的改动，更容易隔离收益和验证正确性。

性能框架不是越小越先进，也不是功能越多就越适合你。要优化的对象必须先明确。

### 13.4 对现有 路由/容量系统的联系

在本文的分层中，跨集群路由器 在跨集群分流层。引擎变化会改变各集群的服务时间、缓存命中、batch 效率与可满足 SLO 的容量，从而改变上层输入；它不因此成为跨集群路由算法。

建议先固定请求分布与路由策略，比较引擎/配置；收益成立后，再按模型、硬件、并行拓扑、缓存状态和关键选项重新标定容量 profile。这里给的是实验方法，没有推断你们当前线上版本、参数或容量。

<a id="ch14"></a>
## 14. 无需 GPU 的动手实验

配套文件：**`inference_lab.py`**，与本手册放在同一目录。只用 Python 标准库，推荐 Python 3.11+。

```bash
python3 inference_lab.py --test
python3 inference_lab.py --demo
```

### 14.1 实验真正做了什么

它包含一个固定随机权重、未训练的单层单头 causal attention toy model：token/位置输入 → Q/K/V → attention → 残差/非线性 → logits → greedy token。

你会看到：

- 每次重算完整前缀，与增量 KV 计算的 logits 对比；
- 逻辑 token 位置如何映射到物理页；
- 满页前缀复用，尾页保持请求私有；
- 同一 scheduler 中交错推进多个请求；
- Prefill 最后一个位置产生首 token；
- 中途取消与页回收；
- 页不足时等待，以及无法继续时明确失败；
- 清掉闲置 prefix cache 后，所有物理页均归还。

toy model 没有训练，生成的整数不承载语言能力。调度器模拟 batch 选择，但数值计算逐 token 执行，不是向量化 GPU batch。它没有真实 GPU kernels、异步、采样随机性、LoRA、MoE、分布式通信、层级缓存或真实时间模拟。prefix 身份固定在一个模型实例内，也不实现生产多租户缓存策略。

### 14.2 为什么这个实验有学习价值

很多入门示例只写 `model.generate()`，把真正需要学习的内存与状态全部藏起来。这里刻意让块分配、引用、提交和终止可见，同时用独立的完整前缀参考计算检查增量路径。

源码函数按阅读顺序：

```text
TinyModel.full_logits / step
→ PagePool
→ PrefixCache
→ Request
→ Engine.tick
→ tests / demo
```

运行测试的实际结果记录在同目录 `inference_lab_test_result.txt`。它只能证明该教学实现测试范围内的行为，不是三个正式框架的正确性或性能报告。

### 14.3 建议做的四个修改

1. 改小 `page_size`，观察物理页数与尾页浪费的变化；不要从 Python 执行时间推断 GPU 最佳页大小。
2. 改小 `token_budget` 或 `prefill_chunk`，观察首 token 和完成轮次如何改变。
3. 去掉 B 的公共前缀，比较 B 实际重算的 token 数。
4. 让请求在 Prefill 中和 Decode 中分别取消，检查在无活跃请求且清缓存后页池是否全部归还。

**实验先不实现异步释放。**这是有意保留的下一阶段练习：给每批工作加完成事件，让 cancel 只做逻辑终止，等事件完成再回收页。不能只在现有同步代码外套一个线程就认为完成了。

<a id="ch15"></a>
## 15. 实现自己的框架：分阶段路线

以下按验收成果组织，而不是承诺几周就能写出生产引擎。

### 阶段 A：可作为裁判的正确参考实现

**目标：**单请求、同步、greedy、标准 causal Transformer。

先用一个小模型和成熟张量库实现完整前缀 forward，再实现增量 KV。先不写 CUDA，不做分页，不做服务接口。选一个明确 checkpoint，固定 tokenizer、chat template、权重和位置编码。

**验收：**同一前缀完整重算 vs KV 增量 logits 在容差内一致；短/长序列、位置边界、EOS、max_new_tokens 均通过。测试还要比较实际生成 token 与 teacher-forced 固定 token 轨迹，避免随机输出分叉掩盖数值问题。

### 阶段 B：单卡 batch 与分页 KV

**目标：**不同长度请求组成 batch，KV 不预留到最大输出长度。

先让 Runner 接收平铺 token、位置、seq lengths 与 block table。用现有可验证的 attention backend，先把布局接对；自写高性能分页 kernel 可以留到后面。

**验收：**batch 与单请求 logits 对齐；页边界前后正确；释放后页可重用；OOM 分配失败不破坏状态；反复循环请求无持续泄漏。

### 阶段 C：连续调度与 chunked prefill

**目标：**显式 waiting/running、每轮 token/sequence budget、请求完成后补入新请求。

设计“计划—预留—执行—提交”的状态边界。刚开始不做 preemption：内存不足就等待或清理闲置缓存，实在无法接入要清楚拒绝。之后再增加可恢复重算。

**验收：**混合长短请求能持续前进，冷请求不会无限饥饿，预算不超限；相同 teacher-forced 输入不因 batch 插队而错误；每个请求只终止一次。

### 阶段 D：API 与真实生命周期

**目标：**提供最小 streaming API，支持取消、超时、过载拒绝和健康检查。

Tokenizer、Detokenizer 与引擎核心分职责；明确 API 层认定断连后怎样传递取消。正确处理 stop string 跨 token、Unicode 增量和重复 cancel。

**验收：**客户端断连不会继续无限占用资源；队列有边界；请求已结束后不继续输出；worker 故障有明确错误语义。记录每个阶段的时间戳。

### 阶段 E：Prefix cache

**目标：**共享不可变满页，建立 identity、引用与淘汰策略。

从最简单的满块 hash 开始通常容易形成可测原型；如果你的目标是树式匹配/会话复用，再考虑 radix。不要因为某框架用了树，就在尚无需求时先实现最复杂的树。

**验收：**命中与不命中 logits 一致；不同模型/adapter/多模态输入不能误命中；活跃页不能被淘汰；缓存清理后引用正确。报告命中“节约的计算 token”，并非只报告命中请求百分比。

### 阶段 F：有 profiling 依据的性能优化

**目标：**只解决已测出的主要瓶颈。

CPU launch 明显时研究 CUDA Graph；小矩阵或多次中间落盘明显时研究融合；设备等待 CPU 时研究异步；KV 带宽明显时研究布局/精度。每次只改变少量可解释变量。

**验收：**同步参考与优化路径对齐；长时间运行无泄漏；profile 显示目标瓶颈确实减少；端到端 SLO goodput 提高，而非仅局部 kernel 更快。

### 阶段 G：推测解码

**目标：**先 fixed-K 与正确 accept/reject/rollback，再做动态预算。

从 greedy 线性 draft 验证开始理解状态，再实现有理论依据的随机采样。测试 KV、grammar、RNG 与缓存提交位置是否都能回退。只有实际接受率和耗时数据充分时，再引入 cost model。

**验收：**greedy 可与 target-only 对齐；随机路径做分布检验和质量评测；多个 K、EOS/stop、全部接受/首个拒绝/中途拒绝均正确；报告净有效输出，不用 draft 数代替。

### 阶段 H：分布式，再到 P/D

**目标：**先一种并行拓扑，再增加 P/D 与缓存传输。

先单机多卡 TP 或明确的模型并行，建立 rank 一致执行与故障行为；MoE EP、PP、CP 是不同工程问题，不必同时实现。P/D 之前先定义 KV layout 合约与所有权协议。

**验收：**collective 次序一致、单 rank 故障能终止/恢复、跨进程取消不泄漏、传输不会访问已复用页。之后才做动态伸缩或在线专家迁移。

### 开发顺序背后的原则

每一步都保留一个较慢但可信的参考路径。性能优化越深入，越需要它来定位问题。真正能独立开发框架的标志，是能说明一个优化改变了什么状态、依赖哪些假设、怎样被证伪，而不只是能复述优化名称。

<a id="ch16"></a>
## 16. 如何证明正确、如何证明更快

### 16.1 正确性不只有“回答看起来正常”

| 层次 | 验证方法 | 能抓到什么 |
|---|---|---|
| 数值 | 固定 token 轨迹比较 logits/top-k/logprob | mask、位置、KV、量化误差 |
| 生成 | greedy、EOS/stop、max output、grammar/JSON/tool | 状态更新与输出协议错误 |
| 随机 | 固定条件下分布统计，配合质量集 | speculative correction / sampling 偏差 |
| 生命周期 | cancel、重试、超时、OOM、worker crash | 双重释放、泄漏、重复输出 |
| 并发 | batch 重排、插入/结束、异步压力 | metadata 映射、跨轮竞态 |
| 组合 | prefix+spec、hybrid+grammar、PD+cancel 等 | 单功能正确但交互错误 |

两种浮点路径不一定逐 bit 相同；不同采样实现也不必生成相同句子。接受标准要由模型误差容忍、服务合同和确定性需求共同定义，不能只靠肉眼评判。

### 16.2 性能测试的分母与边界

建议固定请求 mix 和 SLO 后，定义：

```text
request goodput = 同时满足成功、TTFT 和指定 ITL 条件的请求数 / 测量时间
SLO token goodput = 上述合格请求的真实输出 token 数 / 测量时间
单位资源 goodput = goodput / 全部相关设备数
```

ITL 条件需要明确是每请求最大值、每请求 p95，还是其他合同；不同定义不能混比。按 token 加权可能偏向长输出，所以请求和 token 两个口径都保留。

P/D/draft 的所有设备都要计入，异构硬件则额外比较实际成本。还要计入 CPU、网络、共享存储和缓存维护资源，不能把它们当免费。

### 16.3 可执行的对比矩阵

1. **固定输入条件：**模型 revision、tokenizer/template、量化与质量、硬件、网络、总资源。
2. **两种配置口径：**双方共同支持配置；以及各自最佳配置但相同资源/SLO。两种结果分开呈现。
3. **三类请求：**短输入低延迟；长输入 Prefill 压力；多轮公共前缀复用。保留输出长度、到达间隔、取消比例。
4. **三种负载区间：**低负载、SLO 拐点、过载后恢复。
5. **两种缓存状态：**冷缓存与稳定热缓存，明确 warmup 是否排除。
6. **完整窗口：**含入队与排空；重复运行给波动范围，保留错误与超时。

优先做 open-loop 到达率扫描观察排队增长；closed-loop 固定并发也有用，但客户端等响应再发下一条，会改变到达过程，可能掩盖过载。二者不是可互换的同一个实验。

### 16.4 哪些图最有信息量

- 横轴 offered request rate，纵轴 goodput，并标出 p95/p99 TTFT、ITL 何时越线。
- 横轴并发/长度，纵轴每 step 耗时，拆 CPU、计算、通信、传输。
- cache hit tokens 与净 TTFT 改善，同时画 transfer/prefetch 等待。
- speculation 的 accepted/emitted 与 verify 时间，按 batch/context 分桶。
- KV active、reserved、evictable、free 与 preemption/retraction 同时观察。

“tokens/s 更高”的单个数字，通常不足以决定线上迁移。

### 16.5 从引擎曲线到容量规划

引擎性能是一个随 batch、长度、缓存、硬件、并行与质量设置变化的函数，不是永久常数。分流/扩缩容改变 batch 形态后，原来的平均服务时间可能失效。

因此把 profile 的适用范围和未知状态保留下来。缺样本、采集失败和过期不能当作零负载；本手册不以历史环境数据推导当前生产容量。这一点与跨集群路由的边界同样重要。

<a id="ch17"></a>
## 17. 按问题阅读源码

不要逐文件从第一行读到最后一行。先定位一个问题，写下预期状态，再找修改这些状态的代码。下表路径均相对各项目根目录，链接见第 20 章。

| 想回答的问题 | vLLM | SGLang | DeepInfer |
|---|---|---|---|
| 一轮如何开始/结束？ | `vllm/v1/engine/core.py` | `python/sglang/srt/managers/scheduler.py` 的 event loops | `deepinfer/engine/llm_engine.py` 的 `step` |
| 哪些 token 进入本轮？ | `vllm/v1/core/sched/scheduler.py` | `get_next_batch_to_run`、`ScheduleBatch` | `csrc/engine/scheduler.hpp` |
| KV 谁分配/释放？ | `kv_cache_manager.py`、`block_pool.py` | `mem_cache/radix_cache.py`、`unified_radix_cache.py` 及 pool | `csrc/engine/block_manager.hpp`、`DeferredRelease` |
| GPU 执行在哪里？ | 两条 `GPUModelRunner` 路径 | `srt/model_executor/model_runner.py` | `deepinfer/worker/worker.py` 及 model executor |
| 验证预算怎么选？ | `worker/gpu/spec_decode/adaptive_verification.py` | `speculative/dspark_components/dspark_planner.py` | `csrc/engine/mtp_varlen.hpp` |
| 服务 P/D 边界在哪里？ | connector 与 `disagg_prefill.md` | P/D event loops、`pd_disaggregation.mdx` | `llm_engine.py` 的 KV 保存/提示发送 hooks |

建议做三张笔记：

- **状态表：**每个字段谁写、谁读、在哪个线程/stream、属于哪一轮。
- **所有权表：**页、请求、传输与缓存分别持有什么引用，谁负责终止。
- **时间线：**CPU 计划、H2D、forward、sample、D2H、输出/释放如何交错。

读完一个机制，试着解释它在“请求恰好取消”和“显存恰好不足”时的行为。如果解释不出来，说明只读懂了正常路径。

<a id="ch18"></a>
## 18. 自测题与参考答案

### 题目

1. 为什么 Prefill 可以处理很多 token，普通 Decode 却不能随意一次生成几十个未来 token？
2. GQA 模型算 KV 显存该用 query heads 还是 KV heads？
3. Prefix hit 100% 是否意味着不需要再做任何模型计算？
4. 一个请求输出了 5 token，它的 KV 是否必定已经包含这 5 个？
5. 把页引用降为零后，为什么还可能不能立刻复用？
6. vLLM hash 与 SGLang radix，是否一方有 prefix cache 而另一方没有？
7. 总 token budget 一样的两个 batch 是否应该耗时一样？
8. 两步 EP 负载 `[200,0]`、`[0,200]` 平均很均衡，为什么仍可能慢？
9. speculation 的接受率提高，吞吐是否一定提高？
10. SGLang adaptive EAGLE 与 DSpark ragged 是否是同一个算法？
11. P/D 分离以后，为什么 D KV 不足可能影响 P 的生命周期？
12. 为什么 engine benchmark 的 tokens/s 不能直接代入 上层路由器 的跨集群容量模型？

### 参考答案

1. prompt 已知，可在因果 mask 下并行处理位置；未来 token 尚未确定，有自回归依赖。speculation 也只是先猜再验证。
2. KV heads。query heads 与 KV heads 可不同；MLA/hybrid 还要另按实际状态计算。
3. 否。可能仍需恢复最后位置的 logits或进行边界计算，随后也要生成新 token；外置命中还有搬运成本。实现常会保留至少一个待算位置。
4. 否。刚采样出的末 token 通常要到下一轮才被作为输入写入 KV；预留槽位也不等于已有效写入。
5. GPU/传输可能仍在读它；还需等待完成事件或解除在途 pin。不同实现中的引用字段未必覆盖所有所有权。
6. 两者都有；它们主要在前缀身份和索引/生命周期组织上不同。
7. 否。context 长度、P/D mix、attention 形态、graph padding、通信与缓存都影响时间。
8. 每一步都受最忙 rank 限制，均值掩盖了同步等待；需看逐 step 的最大值。
9. 不一定。更多 draft/verify、跨更大的图档位、采样与同步可能增加更大的成本。
10. 不是。前者按桶的 accept-length EMA 选择档位；后者可按 confidence 和成本分配 ragged 验证预算。
11. P 需要保留源 KV，等 D 分配和传输完成；晚预留会把部分等待/驻留成本移到 P。
12. benchmark 可能无排队、无真实接受率/缓存/故障开销，且 batch/长度不同；全局容量还受 SLO 与路由改变负载分布影响。

<a id="ch19"></a>
## 19. 术语速查与常见误解

| 术语 | 本手册中的含义 |
|---|---|
| token | tokenizer 输出的基本整数单位 |
| logits | 采样前的词表分数 |
| sequence | 一条需要维护生成历史与状态的 token 序列 |
| KV cache | 保留历史 key/value 或相应模型状态以便增量计算 |
| block / page | KV 的分配单位；具体字节布局依后端 |
| prefix hit | 复用同身份、同前缀的已有计算 |
| continuous batching | 每轮允许请求进入和离开 batch |
| chunked prefill | 把长输入分多轮处理，保持完整语义 |
| scheduler | 选择本轮工作与预算的组件 |
| model runner | 将计划转成张量、设备执行与采样的组件 |
| rank | 分布式执行中的一个进程/参与者编号 |
| collective | 一组 rank 协同执行的通信操作 |
| preemption / retraction | 暂停或移出运行集合；恢复可能重算 |
| offload / prefetch | 向较慢层迁出 / 提前载回数据 |
| speculative decoding | draft 候选经 target 验证后提交的生成方法 |
| survival probability | 到某候选位置为止的整个前缀被接受的概率 |
| EPLB | 专家并行内部的专家负载均衡 |
| P/D | Prefill 与 Decode 分阶段部署/执行 |
| SLO | 服务必须达到的时延、成功率等目标 |
| goodput | 满足目标的有效工作率，口径必须明确 |

**常见误解，读完后应能自己纠正：**

- “Paged KV 会减少模型本来需要存的所有 KV。”——主要减少分配浪费并支持共享/回收，不改变模型本身的状态需求。
- “开 prefix cache 就能让所有多轮请求命中。”——前缀、模型身份、容量、路由和布局都要匹配。
- “动态 batching 等于更大的 batch 永远更好。”——可能恶化 ITL/尾延迟。
- “多机 EP 就是多集群路由。”——一个是同步模型执行，一个是独立服务资源间分流。
- “支持某功能等于支持任意功能组合。”——hybrid state、speculation、P/D、LoRA、图执行常有交互边界。
- “DeepInfer 的公开包等于完整线上 DeepSeek 服务。”——公开范围明确更窄。
- “vLLM 只会 PagedAttention，SGLang 只会 RadixAttention。”——这会漏掉当前两者的大部分服务与执行能力。
- “一个性能冠军图足以决定迁移。”——先核对模型质量、资源、到达过程、缓存、实际输出和 SLO。

<a id="ch20"></a>
## 20. 版本固定的资料索引

以下是本手册项目结论的主要依据。源码固定 SHA；release 页面用于确认版本与发布说明。基础数值例子、接口和开发路线为教学推导/建议，不是项目 benchmark。

### 通用原理

<a id="src-f1"></a>
**[F1]** 经典随机推测采样的算法前提：[Fast Inference from Transformers via Speculative Decoding，原始论文](https://proceedings.mlr.press/v202/leviathan23a.html)。这里只用来解释经典线性 rejection sampling，不声称所有当前实现都使用同一个变体。

### vLLM

<a id="src-v0"></a>
**[V0]** [v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)。

<a id="src-v1"></a>
**[V1]** [README：模型、硬件、服务和功能范围](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/README.md)。

<a id="src-v2"></a>
**[V2]** [EngineCore：step 与在途 batch 队列](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py)；[AsyncLLM](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/async_llm.py)。

<a id="src-v3"></a>
**[V3]** [Scheduler：统一 token 进度、分配与 preemption](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py)。

<a id="src-v4"></a>
**[V4]** [Prefix caching 设计](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/design/prefix_caching.md)；[KVCacheManager](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py)；[BlockPool](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py)。

<a id="src-v5"></a>
**[V5]** [Hybrid KV cache manager 设计](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/design/hybrid_kv_cache_manager.md)。注意该文档开头声明的历史撰写基线；分组/布局细节还应与 [V4] 当前源码交叉阅读。

<a id="src-v6"></a>
**[V6]** [Data parallel deployment：路由、EP 耦合、dummy forwards](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/serving/data_parallel_deployment.md)。

<a id="src-v7"></a>
**[V7]** [Disaggregated prefill：connector 与兼容矩阵](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/features/disagg_prefill.md)。

<a id="src-v8"></a>
**[V8]** [Adaptive verification 文档](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/features/speculative_decoding/adaptive_verification.md)；[实现](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py)；[独立的 Dynamic SD 路径](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/features/speculative_decoding/dynamic_speculative_decoding.md)。

<a id="src-v9"></a>
**[V9]** [Expert parallel deployment：EPLB、DBO、通信后端](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/docs/serving/expert_parallel_deployment.md)。

<a id="src-v10"></a>
**[V10]** [GPUModelRunner 路径](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu_model_runner.py)；[GPU Model Runner V2 路径](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py)。

### SGLang

<a id="src-s0"></a>
**[S0]** [v0.5.22 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)。

<a id="src-s1"></a>
**[S1]** [README：项目范围、模型与硬件](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/README.md)。

<a id="src-s2"></a>
**[S2]** [Scheduler：normal/overlap/P-D event loops](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/scheduler.py)；[Req 与 ScheduleBatch](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/schedule_batch.py)。

<a id="src-s3"></a>
**[S3]** [RadixCache：匹配、插入、淘汰、路径锁](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/radix_cache.py)。

<a id="src-s4"></a>
**[S4]** [UnifiedRadixCache：多组件缓存状态](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/unified_radix_cache.py)。

<a id="src-s5"></a>
**[S5]** [HiCache best practices：L1/L2/L3 与 P/D](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/hicache_best_practices.mdx)。

<a id="src-s6"></a>
**[S6]** [P/D 文档与 prefill_complete 限制](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/pd_disaggregation.mdx)；[仓库内 gateway](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/sgl-model-gateway/README.md)。

<a id="src-s7"></a>
**[S7]** [Speculative decoding](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/speculative_decoding.mdx)；[EAGLE adaptive](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/adaptive_speculative_decoding.mdx)；[DSpark planner](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/speculative/dspark_components/dspark_planner.py)；[SPS cost](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/speculative/dspark_components/dspark_sps.py)。

<a id="src-s8"></a>
**[S8]** [Expert parallelism：EP/EPLB 与 overlap](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/expert_parallelism.mdx)。

<a id="src-s9"></a>
**[S9]** [Deterministic inference：backend 与缓存兼容边界](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/docs/docs/advanced_features/deterministic_inference.mdx)。

<a id="src-s10"></a>
**[S10]** [TokenizerManager](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/tokenizer_manager.py)；[DetokenizerManager](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/detokenizer_manager.py)；[ModelRunner](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/model_executor/model_runner.py)。

### DeepInfer

<a id="src-d1"></a>
**[D1]** [README：开源范围、模型/硬件、依赖与确定性](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md)。

<a id="src-d2"></a>
**[D2]** [LLMEngine：step、abort、DeferredRelease、服务 hooks](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/engine/llm_engine.py)；[C++ Scheduler](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp)；[BlockManager](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/block_manager.hpp)。

<a id="src-d3"></a>
**[D3]** [MTPVarlen：confidence、预算、成本估计与异步历史](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/mtp_varlen.hpp)。

<a id="src-d4"></a>
**[D4]** [Worker：forward / previous post-process / sample](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py)。

<a id="src-d5"></a>
**[D5]** [EPLB policy](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/eplb/policy.py)。

<a id="src-d6"></a>
**[D6]** [YoCo bounded replay 与等价性测试](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/yoco/README.md)。

<a id="src-d7"></a>
**[D7]** [Benchmark 说明](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/README.md)；[benchmark.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/benchmark.py)；[plot_pareto.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/examples/sweep/plot_pareto.py)。

---

**证据与实验边界：**项目比较以固定源码/官方文档为依据，示例数字均注明教学性质。CPU 配套实验单独验证了 toy 实现；本次没有跑正式框架的 GPU/NPU benchmark，也不代表任何特定企业的线上引擎配置。文中的开发路线和优先级是据此提出的工程建议。

<a id="ch21"></a>
## 21. 负载均衡专题：官方部署说明与 DualPath 组内调度

**本章的证据范围。**[DeepSeek 官方部署说明](https://zhuanlan.zhihu.com/p/2095560295930508953)是 DeepSeek 已认证机构号发布的《DeepSeek V4.1 Flash 大规模推理部署实践》，读取时页面标注编辑于 2026-10-20。它说明生产系统中尚未完整开源的负载均衡和缓存服务，不能当成全部已在公开仓库中实现。本章的 DualPath 算法核对自 [arXiv v2 第 6 节](https://arxiv.org/html/2602.21548v2#S6)，该版本页面日期为 2026-02-26。知乎引用的是 SIGCOMM 2026 会议版，标题与 arXiv 版不同；这里不声称核对过会议终稿或生产私有代码。

### 21.1 这篇文章的参考价值在哪里

它最有价值的地方是补齐“引擎之外怎样把系统组织起来”。读的时候，把三种证据分开：开源代码能证明具体实现；官方部署文章能证明作者披露的生产设计；论文能说明算法和实验条件。三者能相互解释，但不能无条件互换。

| 学习问题 | 官方部署说明补充的事实 | 对设计自己系统的启发 |
|---|---|---|
| Prefill 的负载怎么算 | 按输入总长分桶，再按拟合的 attention 计算代价分配请求和 chunk | 请求数、未命中 token 数、真实耗时是三种不同量 |
| Decode 为什么不能只数请求 | 同时平衡请求数和 KV 总长度；引用 DualPath 的组内规则 | 并发决定 batch 的一部分成本，历史长度影响 attention 与显存 |
| 为什么不按前缀亲和路由 | Block KV 放在独立共享存储，所有 Prefill 可读取任意前缀 | 缓存架构决定路由可自由选择的范围 |
| 扩缩容怎样应对长尾 | P/D 分开估算需求，实例下线包含宽限期与剩余请求/KV 转移 | 扩缩容需要与请求生命周期和迁移协议一起设计 |
| 哪些功能能直接复用 | KVCache 和负载均衡代码尚未完整开源 | 设计说明不等于可直接运行的开源组件 |

文章中的大集群吞吐、每卡收入和线上优化百分比是作者特定部署下的报告，不是三个框架公平对测的结果。不能据此计算你的成本优势，也不能把该文中的配置常量作为其他模型的默认值。

### 21.2 “组内”到底是哪一层

先建立这个层次，才能读懂同名的 scheduler：

```text
跨集群路由：新请求发给哪个服务集群？
    ↓
服务集群内：选择 Prefill / Decode 实例或 engine group
    ↓
组内请求分配：这一条请求交给哪个 attention DP rank / engine？
    ↓
单引擎迭代调度：本次 forward 取哪些请求、多少 token？
    ↓
MoE 专家调度：计算经过哪些专家副本，怎样避免某张卡变慢？
```

知乎 Decode Load Balancer 引用的 **DualPath 组内调度是第三层**，对应论文 §6.1 的 **DE Scheduling Phase 2: within a group**。论文 §6.2 的 **Intra-Engine Scheduling** 则是第四层，讲 Prefill 的 compute quota。不要因为中文都可以叫“内部调度”，就把它们当成同一算法。

DualPath 的 group 是降低调度器交互压力的组织单位：同组全为 Prefill engines（PE）或全为 Decode engines（DE），同一节点上的 engines 属于同组，由 rank 0 leader 定期统一拉取任务。论文并未把这里的 group 定义为一个地理区域或独立服务集群，也不能把每个 group 都默认为单机。

论文所称 global scheduler 是这一推理系统范围内的全局协调者。**“global”这个词本身不意味着跨地域、跨数据中心或 上层路由器 级别的流量治理。**

### 21.3 Decode 为什么要同时看两个负载维度

对每个 Decode engine `e`，调度器关心：

| 符号 | 意义 | 影响 |
|---|---|---|
| `seq[e]` | 已分配、尚未完成的请求数 | 并发规模和每步 batch 成本 |
| `tok[e]` | 上述请求的 token 总数 | attention 历史长度、KV 显存以及相关数据搬运压力的代理量 |
| `free_hbm[e]` | 可用于新请求的剩余显存 | 这个请求能否接入的硬约束 |
| `len(r)` | 本次分配为新请求计入的长度 | 请求分配后对 token 负载的增量 |

`len(r)` 不能擅自解释为“精确预测了请求最终会生成多少 token”。论文该段没有给出未来输出长度预测器。自己实现时，需要定义清楚当前已知长度、预留输出空间和运行中增长如何分别记账。原始 token 数也只是在特定模型/硬件下使用的成本代理量，不是各模型通用的时间单位。

例如两张卡都在跑 8 条请求：一张的每条上下文只有 1K，另一张每条都是 64K。least-connections 会认为它们一样忙，实际 attention 和 KV 压力可能相差很大。反过来，只按总 token 选择，也可能把大量短请求堆到一张卡，使其 batch 太大。这是同时看 `seq` 和 `tok` 的动机。

### 21.4 DualPath Decode 组内算法：逐步拆解

下面对应论文 §6.1。它是**显存可行性约束下的双维度贪心分配**；这句话是对算法的概括，不是论文给出的独立算法名称。

**第一步：计算这一轮计划接纳的请求集合 R。**把组内剩余 HBM 加总，从该组私有队列头开始，估算可以放进总剩余显存的前缀请求集合。这里暂时假设显存可以完美拼在一起，所以 `R` 是可接纳集合的上界。真实显存分散在不同卡上，后面还要逐卡验证。

**第二步：计算 token 高水位 Z。**设组内 engines 集合为 `E`：

```text
Z = 1.05 × (Σ tok[e] + Σ len(r)) / |E|
             e ∈ E       r ∈ R
```

先把“已有负载 + 本轮准备新增负载”平均分到各卡，再放宽 5%。因此 `Z` 不是单卡显存容量，也不是固定的 105% GPU 利用率，更不是 goodput 的目标值。它是这次 fetch 的负载分界线。

**第三步：逐条取私有队列的队首请求 r。**

1. 筛出显存足以接下 `r` 的卡。没有任何卡能装下时，本轮 fetch 结束；论文规则不会跳过队首继续挑更小的请求。
2. 在可接入的卡中，筛出 `tok[e] + len(r) ≤ Z` 的集合 `low`。
3. 如果 `low` 非空，在里面选 `seq[e]` 最小的卡：既不突破 token 高水位，又尽量平衡并发数。
4. 如果 `low` 为空，在所有可接入卡中选 `tok[e]` 最小的卡：优先控制 KV/token 压力，即使本次接入会越过 `Z`。
5. 分配后更新请求数、token 计数和显存预留，再处理下一个请求。

教学伪代码如下。`estimate_kv_bytes`、并列候选的确定性选择是你需要补齐的工程接口，并非已公开的 DualPath 源码：

```python
R = queue_prefix_that_fits_aggregate_free_hbm(queue, engines)
Z = 1.05 * (sum(e.tok for e in engines)
            + sum(r.length for r in R)) / len(engines)

assigned = []
while queue:
    r = queue[0]
    feasible = [e for e in engines
                if e.free_hbm >= estimate_kv_bytes(r)]
    if not feasible:
        break

    low = [e for e in feasible if e.tok + r.length <= Z]
    if low:
        chosen = min(low, key=lambda e: e.seq)
    else:
        chosen = min(feasible, key=lambda e: e.tok)

    reserve_and_assign(chosen, r)  # 同步更新 tok / seq / free_hbm
    assigned.append((chosen, queue.pop(0)))
```

论文没有在该段规定并列候选如何打破平局、取消如何回滚预留、状态多久汇报一次，也没有证明该贪心策略全局最优。实现中必须补齐这些细节，不能把伪代码当成生产完整实现。

### 21.5 算一遍：为什么短请求会去 KV 更长的卡

以下数字只用于教学，`K` 统一表示千 token，不涉及真实 GPU 容量。假设三张卡都有足够 HBM，本轮队列里有一个 2K 的短请求和一个 12K 的长请求：

| 卡 | 当前 `tok` | 当前 `seq` |
|---|---:|---:|
| A | 30K | 2 |
| B | 20K | 6 |
| C | 40K | 4 |

计算：`Z = 1.05 × (30 + 20 + 40 + 2 + 12) / 3 = 36.4K`。

先处理 2K 请求：A 接入后 32K、B 接入后 22K，都在阈值内；C 接入后 42K 超过阈值。A/B 中 A 的并发更低，所以选 A。**虽然 B 的 token 更少，也不必选 B。**

然后处理 12K 请求：A 已变成 32K，接入后会是 44K；B 接入后 32K；C 接入后 52K。只有 B 不超过阈值，所以选 B。最后 A/B/C 的 token 为 32K / 32K / 40K，请求数为 3 / 7 / 4。

这正是“短请求补并发、长请求补 token 空间”的来源。不过算法没有固定的“短于多少 K 叫短请求”开关；选择结果由当前负载、此次 `Z`、请求长度和显存可行性共同决定。

再看两个边界：

- **阈值不是拒绝线。**如果可接入的卡接单后全都超过 `Z`，算法仍会选择其中 `tok` 最少的卡。把 `Z` 写成硬拒绝条件会改变算法。
- **组内总空闲显存不是可接入证明。**两卡各剩 6 GiB，总共 12 GiB，队首请求需要单卡 8 GiB；总量看似足够，但没有一卡能接，fetch 停止。这是 `R` 只是上界的原因。

### 21.6 其余三种规则不要与它混淆

**DE 组间分配。**论文还维护一个全局等待队列和每组的私有队列。收到 DE group 拉取时，把全局队列里的请求分到总 `tok` 最少的组，再做前述组内分配。论文明确说 DE 调度不保证全局 FIFO。这仍然是推理系统内部对 engine groups 的分配，不能自动等同于跨服务集群路由。

**PE 请求分配。**论文使用 token 上限 `β` 排除过载 PE；其余 PE 按节点读取队列是否低于 `α` 分为两档，优先在读取队列短的一档中选 `tok` 最小的 PE，以免存储 NIC 缺少后续工作。这里包含 I/O 供给均衡，并不是只追求最小请求队列。论文 Algorithm 1 对分配后的分类刷新没有展开，复现时要显式定义过载判断如何随更新生效。

**KV 读取路径选择。**PE/DE 选定后，比较两端节点的读取队列，在较短的一侧读取 KV。DualPath 允许传统的“存储 → P”以及“存储 → D → 计算网络 → P”路径，以利用 D 侧闲置的存储 NIC。两端同时拆分读取同一个请求是论文提出的后续方向，不应写成已实现默认机制。[论文 §4–§6](https://arxiv.org/html/2602.21548v2#S4)

这套路径依赖存储网络、计算网络、RDMA 和 QoS 等基础条件。普通跨地域网络并不自动满足它的带宽和干扰假设。

### 21.7 Prefill compute quota：从“分 token”走向“分时间”

官方部署文章披露的 Prefill LB 有四个层次：

1. **按输入总长分桶：**`<16K / 16–64K / 64–256K / ≥256K`。各桶引擎数按上一轮 compute-quota 需求占比重分。
2. **DP 组间：**按剩余 compute-quota 拉取请求；由于共享 Block KV 使任意 Prefill 都可命中前缀，不做前缀亲和路由。
3. **DP 组内：**批量请求进入引擎后，逐个分给当前 compute-quota 负载最小的 DP rank。
4. **单卡内：**FIFO 处理，每轮设时间预算，通过拟合的 `cost(q,c)` 反解可计算的最大 chunk；`q` 是待计算 token 数，`c` 是已算前缀长度。

为什么同样的 `q` 也不代表同样的 attention 代价？以普通 causal attention 的教学模型为例，本次计算的 query–key 对数量约为：

```text
pairs(q, c) = q*c + q*(q+1)/2
```

`q=128` 时，`c=1K` 与 `c=100K` 的历史读取和点积规模差异很大。这个公式只帮助建立直觉，**不是 V4.1 生产 cost 模型的已披露精确公式**；SWA、压缩/稀疏 attention、硬件与并行策略都会改变拟合关系。拟合参数需要 profile，不能跨模型复制。

DualPath arXiv §6.2 同样讨论按 attention 时间预算做 FIFO packing，但描述的是用二分查找缩小 chunk。当前知乎文章则描述拟合二次模型后反解。两者共享“按时间对齐各 rank”的思想，不应说成实现细节完全一样。

一个建议的实现练习：建立请求 `(cached_tokens, new_tokens)` 的代价模型，分别运行固定 token chunk 和固定时间 quota，比较同步组里 `max(rank_attention_time) / mean(rank_attention_time)` 与 TTFT。训练和验证 profile 时使用不同长度组合，并保留模型误差和尾延迟，不只报告拟合平均误差。

### 21.8 与 vLLM / SGLang 怎样公平地比较

| 对比对象 | 能直接从本手册固定源码验证什么 | 不能据此推出什么 |
|---|---|---|
| vLLM engine scheduler | token 进度、预算、KV 分配、抢占及分布式执行接口 | 不能只看 scheduler.py 就断言整个部署没有实例负载均衡 |
| SGLang engine scheduler / gateway | 批调度、缓存组织与独立 gateway/P-D 路由层 | 不能把 radix 命中逻辑当作所有路由与调度策略的总称 |
| DeepInfer 公共仓库 | 引擎调度、并行执行、EPLB、推测解码等 | 不能从官方部署文章反推 P/D LB 已完整开源 |
| DeepSeek 官方部署文章 + DualPath 论文 | 生产设计披露与论文级调度规则 | 不能证明所披露规则优于其他框架在你的负载下的实际配置 |

如果当前关心负载均衡，比较对象应是“入口路由 + 缓存体系 + P/D 选择 + engine 调度”的整条链。比较各自一个叫 scheduler 的文件，容易把不同层次的设计放在一起。

### 21.9 对 上层路由器 的启发与可验证的下一步

沿用本文的分层：跨集群路由器 是跨集群分流。DualPath 的逐卡显存约束、组内 `Z` 和读取队列并不是它的直接替换件。最值得借鉴的是三条设计思路：

- **用能解释成本的负载量。**在请求数之外区分历史长度、未命中长度、已知输出进度和缓存读取工作；但哪些量能在入口及时拿到，必须先核实。
- **把硬可行性与优化排序分开。**可达、健康、模型兼容和容量可接纳是前提；其后的候选排序可以组合延迟、预计工作量和成本。不要把缺失指标当成“负载很低”。
- **按资源层级做决策。**上层控制集群间流量，下层控制单卡/同步组内的 batch 和 KV；由下层发布带时间戳、适用模型/硬件和误差范围的服务能力，而不是让上层模拟所有 GPU 状态。

下面是建议的离线研究流程，不代表已经上线，也不代表当前 上层路由器 已实现这些能力：

| 步骤 | 产出 | 验证标准 |
|---|---|---|
| 建立请求特征记录 | 输入长度、可复用长度、未命中长度、输出长度、候选集群、状态时间戳 | 未知/缺失/零值可区分；不把事后已知输出当成入口已知信息 |
| 构建可校准的工作量模型 | Prefill 代价、Decode 活跃序列与历史长度、KV 读取压力 | 在保留出的模型/长度/负载区间评估误差 |
| 比较候选策略 | 原有策略、最少请求、最少 token、双维度贪心等 | 相同请求流、约束、失败/取消语义下比较 |
| 闭环模拟与影子评估 | 各集群队列与时延演化 | 路由改变服务进度，不能直接把旧日志的下一秒队列当成新策略结果 |
| 小范围负载实验 | SLO 内 goodput、TTFT/TBT 分位、失败率、资源开销 | 包含长短混合、缓存冷热、突发、状态陈旧和异构集群 |

尤其不要直接把 `1.05` 调进跨集群权重公式：它来自论文的组内负载分界。对于异构集群，需要先把“token 数”转成可比较的预计服务时间或剩余可用能力。这个转换是新设计，需要自己的证据。

### 21.10 阅读后应能回答的五个问题

1. **为什么队列里是短请求时，算法可能选 token 更多的卡？**因为多张卡接入后都不超过 `Z`，此时比较的是 `seq`。
2. **为什么全组显存总量足够，仍然接不下队首请求？**显存分散在不同卡上，必须有一个候选卡能够容纳。
3. **`Z` 是显存上限吗？**不是；显存可行性单独判断，所有候选都超过 `Z` 时仍可分配。
4. **共享 KV 就一定应该放弃前缀亲和吗？**不是；需要比较各路径的真实延迟、带宽和本地命中收益，官方文章的选择有其存储架构前提。
5. **读懂 DualPath 后能直接替换跨集群路由吗？**不能；可迁移的是负载建模与约束分层的思想，具体对象和状态语义需要重新设计。
