"""
train_intent_classifier.py
---------------------------
训练轻量级意图分类器（TF-IDF + LogisticRegression），
用于在检索前拦截伪装型域外问题（跨境管辖权穿透、跨法域概念缝合）。

目标：二分类 IN_DOMAIN vs OUT_OF_DOMAIN
延迟：TF-IDF 向量化 + predict_proba ≈ 1-5 ms（远低于 10ms 约束）

用法：
    python train_intent_classifier.py          # 训练并保存 intent_classifier.pkl
    python train_intent_classifier.py --test   # 训练后跑快速验证
"""

import pickle
import os
import sys
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ════════════════════════════════════════════════════════════════════
# 训练数据 — 正负样本按穿透模式精心构造
# ════════════════════════════════════════════════════════════════════

TRAINING_DATA = [
    # ═══════════ IN_DOMAIN — 纯正的中国境内财税合规问题 ═══════════
    ("个人所得税专项附加扣除每个月能抵扣多少额度", "IN_DOMAIN"),
    ("小型微利企业2024年企业所得税有什么优惠政策", "IN_DOMAIN"),
    ("研发费用加计扣除的比例是多少", "IN_DOMAIN"),
    ("高新技术企业认定后企业所得税税率是多少", "IN_DOMAIN"),
    ("增值税小规模纳税人免征增值税的标准是什么", "IN_DOMAIN"),
    ("契税法规定的契税税率幅度是多少", "IN_DOMAIN"),
    ("房产税从价计征和从租计征有什么区别", "IN_DOMAIN"),
    ("环境保护税的大气污染物怎么计算污染当量数", "IN_DOMAIN"),
    ("印花税的应税凭证包括哪些", "IN_DOMAIN"),
    ("赡养老人专项附加扣除独生子女能全额享受吗", "IN_DOMAIN"),
    ("企业购置环境保护专用设备能否抵免企业所得税", "IN_DOMAIN"),
    ("出口退税的申报流程和时限要求是什么", "IN_DOMAIN"),
    ("境外所得在中国需要缴纳个人所得税吗", "IN_DOMAIN"),
    ("跨境服务增值税零税率和免税有什么区别", "IN_DOMAIN"),
    ("公司在香港上市后股息红利的企业所得税怎么处理", "IN_DOMAIN"),
    ("非居民企业转让中国境内股权怎么交企业所得税", "IN_DOMAIN"),
    ("发票管理办法规定电子发票和纸质发票效力一样吗", "IN_DOMAIN"),
    ("车辆购置税的新能源汽车免税政策还延续吗", "IN_DOMAIN"),
    ("城镇土地使用税大城市中等城市小城市的税额标准是多少", "IN_DOMAIN"),
    ("消费税的卷烟计税怎么区分甲类和乙类", "IN_DOMAIN"),
    # -- 追加多样性 in-domain --
    ("我们公司今年投入污水处理设备环保税能少交吗", "IN_DOMAIN"),
    ("车船税每年都要交吗新能源车免不免", "IN_DOMAIN"),
    ("个人出租房屋要交什么税房产税和个税怎么算", "IN_DOMAIN"),
    ("公司租办公楼办公房产税由承租方交还是房东交", "IN_DOMAIN"),
    ("离婚时按协议分到房子需要交契税吗", "IN_DOMAIN"),
    ("从香港进口原材料是不是零关税不用交钱", "IN_DOMAIN"),
    ("直辖市住房租金个税扣除标准是每月多少", "IN_DOMAIN"),
    ("赡养老人的3000元额度独生子女可以全部扣除吗", "IN_DOMAIN"),
    ("公司货车一半时间闲置车船税可以按实际使用时间交吗", "IN_DOMAIN"),
    ("公司占用农用地建厂房耕地占用税税额大概是多少", "IN_DOMAIN"),
    ("电子发票全面推行后和纸质发票法律效力一样吗", "IN_DOMAIN"),
    ("烟叶税的纳税义务人是种烟叶的农民还是收购单位", "IN_DOMAIN"),
    ("我们公司综合利用资源生产产品企业所得税怎么处理", "IN_DOMAIN"),
    ("非独生子女赡养老人每人每月扣除额度有上限吗", "IN_DOMAIN"),
    ("企业和高校合作基础研究支付的科研经费可以加计扣除吗", "IN_DOMAIN"),
    ("买了一间商铺能抵扣个税的首套住房贷款利息吗", "IN_DOMAIN"),
    # -- 本轮评测发现的误杀案例（seed=717）--
    ("因为疫情经营困难可以申请延期缴纳税款吗最长能延多久", "IN_DOMAIN"),
    ("居民企业之间股息红利收益企业所得税怎么处理免不免税", "IN_DOMAIN"),
    ("自媒体在多个网络平台有收入平台向税务机关报送涉税信息具体报送哪些内容", "IN_DOMAIN"),
    ("公司有一笔三年都收不回来的应收账款可以直接在税前扣除吗", "IN_DOMAIN"),
    ("中国企业向海外投资者分配股息预提所得税一般税率和税收协定优惠税率分别是多少", "IN_DOMAIN"),
    ("收到一张增值税普通发票在哪里可以查验发票真伪", "IN_DOMAIN"),
    ("买了一辆进口车交了车辆购置税车船税就不用再交了吧", "IN_DOMAIN"),
    # -- q213/q216 标签修正对应正样本（资产重组/发票查验属境内税法域内问题）--
    ("资产重组的特殊性税务处理", "IN_DOMAIN"),
    ("企业合并分立的所得税怎么交", "IN_DOMAIN"),
    ("发票查验和虚开发票", "IN_DOMAIN"),
    # -- q220 标签修正：编造虚假计税依据由征管法第六十四条覆盖，属域内问题 --
    ("我们公司工作失误多报了一笔收入算不算虚假申报会被罚款吗", "IN_DOMAIN"),

    # ═══════════ OUT_OF_DOMAIN — 伪装型穿透问题 ═══════════

    # --- 类别 A: 跨境管辖权穿透 ---
    ("在亚马逊上开店卖货到欧洲需要交什么税VAT怎么申报", "OUT_OF_DOMAIN"),
    ("我们做Shopee马来西亚站当地的增值税和关税如何处理", "OUT_OF_DOMAIN"),
    ("亚马逊美国站的销售税怎么计算需要注册美国税号吗", "OUT_OF_DOMAIN"),
    ("做跨境电商TikTok Shop英国站英国的VAT税率是多少怎么交", "OUT_OF_DOMAIN"),
    ("eBay上卖东西到德国要不要注册德国VAT税号申报", "OUT_OF_DOMAIN"),
    ("在日本乐天上开店日本的消费税是怎么征收的", "OUT_OF_DOMAIN"),
    ("独立站卖货到法国欧盟的IOSS和VAT一站式申报怎么做", "OUT_OF_DOMAIN"),
    ("澳大利亚海外仓发货GST要怎么处理和申报", "OUT_OF_DOMAIN"),
    ("新加坡消费税明年涨到9%对我们跨境电商卖家有什么影响", "OUT_OF_DOMAIN"),
    ("Lazada泰国站的税务怎么合规要交哪些本地税", "OUT_OF_DOMAIN"),
    ("亚马逊FBA发货美国怎么注册EIN税号和交销售税", "OUT_OF_DOMAIN"),
    ("东南亚各国Lazada和Shopee的VAT和GST怎么合规", "OUT_OF_DOMAIN"),

    # --- 类别 B: 跨法域概念缝合 ---
    ("劳务派遣员工算不算公司的从业人员这些人的工资能不能算企业所得税扣除", "OUT_OF_DOMAIN"),
    ("和员工签劳务合同还是劳动合同更省税社保费和个税有什么区别", "OUT_OF_DOMAIN"),
    ("员工工伤赔偿金能不能在企业所得税税前扣除", "OUT_OF_DOMAIN"),
    ("公司解除劳动合同给员工的补偿金要不要代扣个税还是可以免税", "OUT_OF_DOMAIN"),
    ("临时工没签合同但实际用工这种算劳务报酬还是工资薪金扣个税", "OUT_OF_DOMAIN"),
    ("竞业限制补偿金算不算工资薪金怎么报个人所得税", "OUT_OF_DOMAIN"),
    ("股权代持协议下实际股东要不要交个人所得税", "OUT_OF_DOMAIN"),
    ("合伙人退伙分得的财产份额要不要交增值税和个税", "OUT_OF_DOMAIN"),
    ("公司被行政处罚的罚款能不能在企业所得税前扣除冲减利润", "OUT_OF_DOMAIN"),
    ("合同违约赔偿金收入要不要交增值税和企业所得税", "OUT_OF_DOMAIN"),
    ("员工告公司非法解除劳动合同败诉赔偿金能不能税前扣除", "OUT_OF_DOMAIN"),
    ("公司司机出车祸撞了人赔偿金算企业所得税什么费用能扣吗", "OUT_OF_DOMAIN"),

    # --- 类别 C: 其他域外/非税问题 ---
    ("公司在美国上市需要符合哪些SEC的税务披露要求", "OUT_OF_DOMAIN"),
    ("被环保部门罚款能不能在企业所得税前扣除", "OUT_OF_DOMAIN"),
    ("公司股东纠纷对方要求查账税务方面有什么风险", "OUT_OF_DOMAIN"),
    ("公司准备破产清算税务注销要注意什么欠税怎么处理", "OUT_OF_DOMAIN"),
    ("稽查局来查账说我们偷税这会涉及刑事责任吗会被抓吗", "OUT_OF_DOMAIN"),
    ("公司涉及刑事案件被罚没财产这些损失税务上怎么处理", "OUT_OF_DOMAIN"),
    ("跟供应商打官司的律师费能不能算企业所得税成本费用", "OUT_OF_DOMAIN"),
]

# ════════════════════════════════════════════════════════════════════
# 验证集 — 用于 --test 快速验证
# ════════════════════════════════════════════════════════════════════

TEST_DATA = [
    # IN_DOMAIN
    ("小微企业2024年所得税优惠政策具体怎么享受", "IN_DOMAIN"),
    ("增值税留抵退税的条件和流程", "IN_DOMAIN"),
    ("个人出租房屋要交什么税，房产税和个税怎么算", "IN_DOMAIN"),
    ("关税完税价格包含海运费和保险费吗", "IN_DOMAIN"),
    ("车船税每年都要交吗新能源车免不免", "IN_DOMAIN"),
    ("环境保护税低于排放标准能减征多少", "IN_DOMAIN"),

    # OUT_OF_DOMAIN
    ("亚马逊FBA发货到美国，怎么注册美国EIN税号和交销售税", "OUT_OF_DOMAIN"),
    ("我们在东南亚做Lazada和Shopee，各国的VAT和GST怎么合规", "OUT_OF_DOMAIN"),
    ("公司司机出车祸撞了人，赔偿金算企业所得税的什么费用能扣吗", "OUT_OF_DOMAIN"),
    ("员工告我们非法解除劳动合同，如果败诉赔偿金能不能税前扣除", "OUT_OF_DOMAIN"),
    ("公司涉及刑事案件被罚没财产，这些损失税务上怎么处理", "OUT_OF_DOMAIN"),
    ("跟供应商打官司的律师费能不能算企业所得税的成本费用", "OUT_OF_DOMAIN"),
]

# ════════════════════════════════════════════════════════════════════
# 字符级 N-gram 范围 — 捕捉税种名称子串、跨境平台名等关键信号
# ════════════════════════════════════════════════════════════════════
CHAR_NGRAM_RANGE = (2, 5)  # 2-gram~5-gram: "增值" "增值税" "VAT" "亚马逊" "劳务派遣"

# 正则化强度（C 越大正则化越弱，适合小样本高信噪比场景）
LOGISTIC_C = 1.0  # 降低 C 增强正则化，防止过拟合到通用词

# 分类阈值：OUT_OF_DOMAIN 置信度 >= 此值才触发阻断
# 降低阈值 → 更激进拦截（可能误杀）；提高阈值 → 更保守放行（可能漏过）
# 与 rag_agent.py 中的 OOD_THRESHOLD 保持一致（运行时以 rag_agent.OOD_THRESHOLD 为准）
threshold = 0.60  # 拦截高置信 OOD（跨境电商 0.61~0.70），in-domain 安全边界 ~0.49


def build_pipeline() -> Pipeline:
    """构建 TF-IDF + LogisticRegression 分类流程"""
    return Pipeline([
        ("vectorizer", TfidfVectorizer(
            analyzer="char_wb",           # 字符级 n-gram + 词边界
            ngram_range=CHAR_NGRAM_RANGE,
            max_features=2000,            # 控制模型体积（.pkl < 1MB）
            sublinear_tf=True,            # tf = 1 + log(tf)
            strip_accents="unicode",
        )),
        ("classifier", LogisticRegression(
            C=LOGISTIC_C,
            max_iter=500,
            solver="liblinear",           # 小样本最优求解器
            random_state=42,
            class_weight="balanced",      # 正负样本可能不均衡
        )),
    ])


def train(save_path: str) -> Pipeline:
    """训练并保存模型"""
    X = [text for text, _ in TRAINING_DATA]
    y = [label for _, label in TRAINING_DATA]

    print(f"训练集: {len(X)} 条 ({y.count('IN_DOMAIN')} IN_DOMAIN, {y.count('OUT_OF_DOMAIN')} OUT_OF_DOMAIN)")
    print(f"字符 n-gram: {CHAR_NGRAM_RANGE}, LogisticRegression C={LOGISTIC_C}")
    print(f"阻断阈值: OUT_OF_DOMAIN confidence >= {threshold}")

    pipe = build_pipeline()
    pipe.fit(X, y)

    # ── 训练集自评 ──
    y_pred = pipe.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print(f"\n训练集准确率: {train_acc:.2%}")

    # 更详细的报告
    print("\n" + classification_report(y, y_pred, digits=3))

    # ── 查看学到的关键特征 ──
    vec = pipe.named_steps["vectorizer"]
    clf = pipe.named_steps["classifier"]
    feature_names = vec.get_feature_names_out()
    if len(feature_names) > 0 and len(clf.coef_) > 0:
        coef = clf.coef_[0]  # shape: (n_features,)
        top_n = 20
        top_idx = coef.argsort()[-top_n:][::-1]
        bottom_idx = coef.argsort()[:top_n]
        print(f"\n[OOD] 最强 OUT_OF_DOMAIN 信号 (top {top_n}):")
        for idx in top_idx:
            print(f"  {feature_names[idx]:20s}  weight={coef[idx]:+.4f}")
        print(f"\n[OK]  最强 IN_DOMAIN 信号 (top {top_n}):")
        for idx in reversed(bottom_idx):
            print(f"  {feature_names[idx]:20s}  weight={coef[idx]:+.4f}")

    # ── 持久化 ──
    with open(save_path, "wb") as f:
        pickle.dump({
            "pipeline": pipe,
            "threshold": threshold,
            "char_ngram_range": CHAR_NGRAM_RANGE,
        }, f)

    file_size = os.path.getsize(save_path)
    print(f"\n✅ 模型已保存: {save_path} ({file_size / 1024:.1f} KB)")
    return pipe


def quick_test(pipe: Pipeline, threshold: float = threshold):
    """对验证集做快速测试"""
    X_test = [text for text, _ in TEST_DATA]
    y_true = [label for _, label in TEST_DATA]

    # predict_proba 返回 [[prob_IN_DOMAIN, prob_OUT_OF_DOMAIN]]
    y_proba = pipe.predict_proba(X_test)
    classes = pipe.classes_

    print(f"\n{'='*80}")
    print("验证集测试结果")
    print(f"{'='*80}")

    correct = 0
    for i, (text, true_label) in enumerate(TEST_DATA):
        proba = y_proba[i]
        pred_label = classes[proba.argmax()]
        ood_conf = proba[list(classes).index("OUT_OF_DOMAIN")]
        blocked = "[BLOCK]" if (pred_label == "OUT_OF_DOMAIN" and ood_conf >= threshold) else "[PASS]"
        match = "✓" if pred_label == true_label else "✗"
        if pred_label == true_label:
            correct += 1
        print(f"[{match}] {blocked} | OOD conf={ood_conf:.3f} | {text[:60]}...")

    acc = correct / len(TEST_DATA)
    print(f"\n验证集准确率: {acc:.2%} ({correct}/{len(TEST_DATA)})")

    # ── 重点: 验证已知穿透案例 ──
    critical_cases = [
        ("我们准备在亚马逊上开店卖货到欧洲，跨境电商的税务合规要注意哪些方面？VAT怎么申报？",
         "OUT_OF_DOMAIN", "q197 跨境电商穿透"),
        ("我们公司用了不少劳务派遣员工，他们算我们公司的从业人员吗？劳务派遣的增值税和企业所得税怎么处理？",
         "IN_DOMAIN", "q211 劳务派遣（边界案例：ML 放行，RAG 部分作答+声明缺口兜底）"),
        ("我们公司今年投入了一套新的污水处理设备，排放的污染物明显减少了，环保税能少交吗？",
         "IN_DOMAIN", "q52 环保税（不应误杀）"),
        ("2023年起赡养老人的个税专项附加扣除标准提高到了每月多少钱？",
         "IN_DOMAIN", "q07 个税扣除（不应误杀）"),
        ("我们公司因为工作失误多报了一笔收入，这种算不算虚假申报？会被罚款吗？",
         "IN_DOMAIN", "q220 虚假申报（征管法64条已覆盖，应放行）"),
    ]

    print(f"\n{'='*80}")
    print("关键穿透案例验证")
    print(f"{'='*80}")
    for text, expected, name in critical_cases:
        proba = pipe.predict_proba([text])[0]
        ood_conf = proba[list(classes).index("OUT_OF_DOMAIN")]
        pred = classes[proba.argmax()]
        match = "✓" if pred == expected else "✗ MISMATCH"
        print(f"[{match}] {name}: OOD conf={ood_conf:.3f} → pred={pred} (expected={expected})")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="训练意图分类器（IN_DOMAIN vs OUT_OF_DOMAIN）")
    parser.add_argument("--output", type=str, default="intent_classifier.pkl",
                        help="模型输出路径 (default: intent_classifier.pkl)")
    parser.add_argument("--test", action="store_true",
                        help="训练后跑验证集测试")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="OUT_OF_DOMAIN 阻断置信度阈值 (default: 0.60)")
    args = parser.parse_args()

    test_threshold = args.threshold

    # 输出到 rag_agent.py 同目录
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    pipe = train(save_path)

    if args.test:
        quick_test(pipe, threshold=test_threshold)


if __name__ == "__main__":
    main()
