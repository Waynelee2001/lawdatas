const fs = require('fs');
const path = require('path');

const ANNOT_DIR = path.join(__dirname, 'data', 'laws_annotation');
const LAW_INDEX_FILE = path.join(__dirname, 'js', 'law_index.js');
const OUT_FILE = path.join(__dirname, 'js', 'backlinks.js');

// 1. 读取 Law Name To ID 映射
let lawNameToId = {};
const lawIndexContent = fs.readFileSync(LAW_INDEX_FILE, 'utf-8');
const match = lawIndexContent.match(/const\s+lawNameToId\s*=\s*\{([\s\S]*?)\};/);
if (match && match[1]) {
    const lines = match[1].split('\n');
    for (let line of lines) {
        const m = line.match(/"([^"]+)"\s*:\s*"(\d+)"/);
        if (m) {
            lawNameToId[m[1]] = m[2];
        }
    }
} else {
    console.error("无法从 js/law_index.js 中提取 lawNameToId！");
    process.exit(1);
}

// 补充全称前缀
const mapFileContent = fs.readFileSync(path.join(__dirname, 'all_laws_map.json'), 'utf-8');
const lawMap = JSON.parse(mapFileContent);
for (const [lid, fullname] of Object.entries(lawMap)) {
    if (!lawNameToId[fullname] || parseInt(lid) > parseInt(lawNameToId[fullname])) {
        lawNameToId[fullname] = lid;
    }
    if (fullname.startsWith("中华人民共和国")) {
        const alias = fullname.replace("中华人民共和国", "");
        if (!lawNameToId[alias] || parseInt(lid) > parseInt(lawNameToId[alias])) {
            lawNameToId[alias] = lid;
        }
    }
}

// 2. 原封不动地照搬前端的正则和处理函数
const PARAGRAPH_SUFFIX = ''; // 后端同步，不再收取“款/项”作为法条本体
const LAW_REF_PATTERN = new RegExp('(《[^》]+》|民法典|刑法|公司法|合伙企业法|民事诉讼法|刑事诉讼法|行政诉讼法|行政处罚法|行政许可法|行政强制法|行政复议法|国家赔偿法|公务员法|治安管理处罚法|票据法|证券法|保险法|海商法|信托法|企业破产法|劳动法|劳动合同法|消费者权益保护法|反不正当竞争法|反垄断法|产品质量法|商标法|专利法|著作权法|仲裁法|公证法|律师法|法律援助法|人民陪审员法|监察法|社区矫正法|立法法|宪法)(第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?' + PARAGRAPH_SUFFIX + ')', 'g');
const SELF_REF_PATTERN = new RegExp('(本法|本规定|本条例|本解释|本办法|本意见|本决定)(第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?' + PARAGRAPH_SUFFIX + ')', 'g');
const FOLLOW_ARTICLE_PATTERN = new RegExp('([、，,])(第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?' + PARAGRAPH_SUFFIX + ')', 'g');

// 前端原版复刻：根据当前解析的 law 状态
let currentLawId = null;
let currentLawTitle = "";

function processLawReferences(html) {
    if (typeof lawNameToId === 'undefined') return html;

    html = html.replace(LAW_REF_PATTERN, function (match, lawPart, articlePart, offset) {
        var lawName = lawPart.replace(/[《》]/g, '');
        var lawId = lawNameToId[lawName];

        if (lawId) {
            var articleOnly = articlePart.match(/第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?/)[0];
            return '<span class="law-ref" data-law-id="' + lawId +
                '" data-law-name="' + lawName +
                '" data-article="' + articleOnly +
                '" data-full="' + lawName + ' ' + articleOnly + '">' + match + '</span>';
        }
        return match;
    });

    if (currentLawId && currentLawTitle) {
        html = html.replace(SELF_REF_PATTERN, function (match, selfWord, articlePart) {
            var articleOnly = articlePart.match(/第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?/)[0];
            return '<span class="law-ref law-ref-self" data-law-id="' + currentLawId +
                '" data-law-name="' + currentLawTitle +
                '" data-article="' + articleOnly +
                '" data-full="' + currentLawTitle + ' ' + articleOnly + '">' + match + '</span>';
        });
    }

    var changed = true;
    while (changed) {
        changed = false;
        html = html.replace(
            new RegExp('(<span class="law-ref[^"]*"[^>]*data-law-id="([^"]+)"[^>]*data-law-name="([^"]+)"[^>]*>[^<]*<\\/span>)([^<《》。）（()]{0,30}?)([、，,]|或者|或|和|及|以及|至)(第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?' + PARAGRAPH_SUFFIX + ')', 'g'),
            function (match, prevSpan, lawId, lawName, between, separator, articlePart) {
                changed = true;
                var articleOnly = articlePart.match(/第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?/)[0];
                return prevSpan + between + separator +
                    '<span class="law-ref" data-law-id="' + lawId +
                    '" data-law-name="' + lawName +
                    '" data-article="' + articleOnly +
                    '" data-full="' + lawName + ' ' + articleOnly + '">' + articlePart + '</span>';
            }
        );
    }

    var gongsiId = lawNameToId['中华人民共和国公司法'];
    if (gongsiId) {
        // 1. 在后端环境中，读取的还是没有被加过任何 span 的原生纯文本！
        // 当我们看到“旧法条（《新法版》新条号）”的拼接时，直接在供提取的字符串里把旧法部份斩草除根（仅保留新法括注）
        // 以免随后的提取器把旧法条再次收录，形成一次引用两遍的错误记录。
        html = html.replace(
            new RegExp('(?:中华人民共和国)?公司法[\\s\\u3000]*第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?(?:、第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?)*(?:&nbsp;|\\s|\\u3000)*（《公司法2023版》((?:第[一二三四五六七八九十百千零〇○]+条(?:、)?)+)）', 'g'),
            function (m, newArts) {
                var firstArt = newArts.split('、')[0];
                return '（<span class="law-ref" data-law-id="' + gongsiId +
                    '" data-law-name="中华人民共和国公司法" data-article="' + firstArt +
                    '" data-full="中华人民共和国公司法 ' + firstArt + '">《公司法2023版》' +
                    newArts + '</span>）';
            }
        );

        // 2. 还有：因为上面的 processLawReferences 循环会把它变成带 span 的形式
        // 若经过了第一步还是带了 span，这里用前端的一摸一样代码清洗：
        html = html.replace(
            new RegExp('<span class="law-ref"[^>]*data-law-id="' + gongsiId + '"[^>]*>([^<]+)<\\/span>(?:&nbsp;|\\s|\\u3000)*（(?:<span class="law-ref"[^>]*>)?《公司法2023版》((?:第[一二三四五六七八九十百千零〇○]+条(?:、)?)+)(?:<\\/span>)?）', 'g'),
            function (m, originalText, newArts) {
                var firstArt = newArts.split('、')[0];
                return '（<span class="law-ref" data-law-id="' + gongsiId +
                    '" data-law-name="中华人民共和国公司法" data-article="' + firstArt +
                    '" data-full="中华人民共和国公司法 ' + firstArt + '">《公司法2023版》' +
                    newArts + '</span>）';
            }
        );
    }

    var minsuId = lawNameToId['民事诉讼法'];
    if (minsuId) {
        html = html.replace(
            new RegExp('(?:中华人民共和国)?(?:民事诉讼法|民诉法|民诉法解释)[\\s\\u3000]*第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?(?:、第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?)*(?:&nbsp;|\\s|\\u3000)*（《民事诉讼法2023版》(第[一二三四五六七八九十百千零〇○]+条)）', 'g'),
            function (m, newArt) {
                return '（<span class="law-ref" data-law-id="' + minsuId +
                    '" data-law-name="民事诉讼法" data-article="' + newArt +
                    '" data-full="民事诉讼法 ' + newArt + '">《民事诉讼法2023版》' +
                    newArt + '</span>）';
            }
        );
        html = html.replace(
            new RegExp('<span class="law-ref"[^>]*data-law-id="' + minsuId + '"[^>]*>([^<]+)<\\/span>(?:&nbsp;|\\s|\\u3000)*（(?:<span class="law-ref"[^>]*>)?《民事诉讼法2023版》(第[一二三四五六七八九十百千零〇○]+条)(?:<\\/span>)?）', 'g'),
            function (m, originalText, newArt) {
                return '（<span class="law-ref" data-law-id="' + minsuId +
                    '" data-law-name="民事诉讼法" data-article="' + newArt +
                    '" data-full="民事诉讼法 ' + newArt + '">《民事诉讼法2023版》' +
                    newArt + '</span>）';
            }
        );
    }

    var xssId = lawNameToId['刑事诉讼法'];
    if (xssId) {
        // 处理 "刑事诉讼法第X条（《刑事诉讼法2018版》第Y条）" → span指向新条号Y
        html = html.replace(
            new RegExp('(?:中华人民共和国)?刑事诉讼法[\\s\\u3000]*第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?[^（]*（《刑事诉讼法2018版》(第[一二三四五六七八九十百千零〇○]+条)）', 'g'),
            function (m, newArt) {
                return '（<span class="law-ref" data-law-id="' + xssId +
                    '" data-law-name="刑事诉讼法" data-article="' + newArt +
                    '" data-full="刑事诉讼法 ' + newArt + '">《刑事诉讼法2018版》' +
                    newArt + '</span>）';
            }
        );
        // 处理已有span + 括注的情况
        html = html.replace(
            new RegExp('<span class="law-ref"[^>]*data-law-id="' + xssId + '"[^>]*>([^<]+)<\\/span>(?:&nbsp;|\\s|\\u3000)*（(?:<span class="law-ref"[^>]*>)?《刑事诉讼法2018版》(第[一二三四五六七八九十百千零〇○]+条)(?:<\\/span>)?）', 'g'),
            function (m, originalText, newArt) {
                return originalText +
                    '（<span class="law-ref" data-law-id="' + xssId +
                    '" data-law-name="刑事诉讼法" data-article="' + newArt +
                    '" data-full="刑事诉讼法 ' + newArt + '">《刑事诉讼法2018版》' +
                    newArt + '</span>）';
            }
        );
    }

    var xzId = lawNameToId['中华人民共和国行政诉讼法'];
    if (xzId) {
        // 处理 "行政诉讼法第X条（《行政诉讼法2017版》第Y条）" → span指向新条号Y
        html = html.replace(
            new RegExp('(?:中华人民共和国)?行政诉讼法[\\s\\u3000]*第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?[^（]*（《行政诉讼法2017版》(第[一二三四五六七八九十百千零〇○]+条)）', 'g'),
            function (m, newArt) {
                return '（<span class="law-ref" data-law-id="' + xzId +
                    '" data-law-name="中华人民共和国行政诉讼法" data-article="' + newArt +
                    '" data-full="中华人民共和国行政诉讼法 ' + newArt + '">《行政诉讼法2017版》' +
                    newArt + '</span>）';
            }
        );
        // 处理已有span + 括注的情况
        html = html.replace(
            new RegExp('<span class="law-ref"[^>]*data-law-id="' + xzId + '"[^>]*>([^<]+)<\\/span>(?:&nbsp;|\\s|\\u3000)*（(?:<span class="law-ref"[^>]*>)?《行政诉讼法2017版》(第[一二三四五六七八九十百千零〇○]+条)(?:<\\/span>)?）', 'g'),
            function (m, originalText, newArt) {
                return originalText +
                    '（<span class="law-ref" data-law-id="' + xzId +
                    '" data-law-name="中华人民共和国行政诉讼法" data-article="' + newArt +
                    '" data-full="中华人民共和国行政诉讼法 ' + newArt + '">《行政诉讼法2017版》' +
                    newArt + '</span>）';
            }
        );
    }

    var zcId = lawNameToId['中华人民共和国仲裁法'];
    if (zcId) {
        html = html.replace(
            new RegExp('(?:中华人民共和国)?仲裁法[\\s\\u3000]*第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?(?:、第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?)*(?:&nbsp;|\\s|\\u3000)*（《仲裁法2025版》((?:第[一二三四五六七八九十百千零〇○]+条(?:、)?)+)）', 'g'),
            function (m, newArts) {
                var firstArt = newArts.split('、')[0];
                return '（<span class="law-ref" data-law-id="' + zcId +
                    '" data-law-name="中华人民共和国仲裁法" data-article="' + firstArt +
                    '" data-full="中华人民共和国仲裁法 ' + firstArt + '">《仲裁法2025版》' +
                    newArts + '</span>）';
            }
        );
        html = html.replace(
            new RegExp('<span class="law-ref"[^>]*data-law-id="' + zcId + '"[^>]*>([^<]+)<\\/span>(?:&nbsp;|\\s|\\u3000)*（(?:<span class="law-ref"[^>]*>)?《仲裁法2025版》((?:第[一二三四五六七八九十百千零〇○]+条(?:、)?)+)(?:<\\/span>)?）', 'g'),
            function (m, originalText, newArts) {
                var firstArt = newArts.split('、')[0];
                return '（<span class="law-ref" data-law-id="' + zcId +
                    '" data-law-name="中华人民共和国仲裁法" data-article="' + firstArt +
                    '" data-full="中华人民共和国仲裁法 ' + firstArt + '">《仲裁法2025版》' +
                    newArts + '</span>）';
            }
        );
    }
    // 通用校正：将「（现改为第Y条）」「（现为第Y条）」跟在 span 后的情况，把带有新属性的 span 包裹在新条号文字上
    // 允许 span 结束标签 </span> 和全角括号 （ 之间有最多 15 个字符的间隔（如"第一款"）
    html = html.replace(
        /(<span class="law-ref"[^>]*>)([^<]+)(<\/span>)([^（]{0,15})(（现(?:改)?为(?:第[一二三四五六七八九十百千零〇○0-9]+(?:、第[^）]+)?条)+[^）]*）)/g,
        function (m, openTag, innerText, closeSpan, middleText, corrNote) {
            var newArtMatch = corrNote.match(/第([一二三四五六七八九十百千零〇○0-9]+)条/);
            if (!newArtMatch) return m;
            var newArt = '第' + newArtMatch[1] + '条';
            var updatedTag = openTag
                .replace(/(data-article=")[^"]*"/, '$1' + newArt + '"')
                .replace(/(data-full="[^"]*? )第[^"]*条"/, '$1' + newArt + '"');
            var innerCorrNote = corrNote.slice(1, -1);
            return innerText + middleText + '（' + updatedTag + innerCorrNote + '</span>）';
        }
    );
    // 连续引用的通用校正：如“...第七十一条（现改为第八十条）、第七十二条（现改为第八十一条）...”
    // 利用正则循环，让后面的连续法条（无前缀）继承前面的 span 的 data-law-id 和 data-law-name
    var consecutiveCorrRegex = /(<span class="law-ref"[^>]*data-law-id="([^"]+)"[^>]*data-law-name="([^"]+)"[^>]*>[^<]*<\/span>)(）(?:、|和|及|或者?|，))(第[一二三四五六七八九十百千零〇○0-9]+条)（(现(?:改)?为)(第[一二三四五六七八九十百千零〇○0-9]+条[^）]*)）/g;
    var oldHtml;
    do {
        oldHtml = html;
        html = html.replace(consecutiveCorrRegex, function (m, prevSpan, lawId, lawName, sep, oldArt, prefix, newArtNumText) {
            var newArtMatch = newArtNumText.match(/第([一二三四五六七八九十百千零〇○0-9]+)条/);
            if (!newArtMatch) return m;
            var newArt = '第' + newArtMatch[1] + '条';
            var newSpan = '<span class="law-ref" data-law-id="' + lawId + '" data-law-name="' + lawName + '" data-article="' + newArt + '" data-full="' + lawName + ' ' + newArt + '">' + prefix + newArtNumText + '</span>';
            return prevSpan + sep + oldArt + '（' + newSpan + '）';
        });
    } while (oldHtml !== html);

    return html;
}

// 3. 结果容器与上下文抓取
let backlinks = {}; // { [targetLawId]: { [targetArticle]: [ { sourceLawId, sourceLawName, sourceArticle, context } ] } }

function addBacklink(targetId, targetArt, sourceId, sourceName, sourceArt, ctx) {
    if (!targetId || !targetArt) return;
    if (!backlinks[targetId]) backlinks[targetId] = {};
    if (!backlinks[targetId][targetArt]) backlinks[targetId][targetArt] = [];

    // 判重
    const exists = backlinks[targetId][targetArt].some(b => b.sourceLawId === sourceId && b.sourceArticle === sourceArt);
    if (!exists) {
        backlinks[targetId][targetArt].push({
            sourceLawId: sourceId,
            sourceLawName: sourceName,
            sourceArticle: sourceArt,
            context: ctx
        });
    }
}

function getContext(text, matchIndex, matchLength, range = 40) {
    const start = Math.max(0, matchIndex - range);
    const end = Math.min(text.length, matchIndex + matchLength + range);
    let ctx = text.substring(start, end).replace(/[\r\n]/g, '');
    if (start > 0) ctx = "..." + ctx;
    if (end < text.length) ctx = ctx + "...";
    return ctx;
}

// 4. 重头戏：读取所有文件并强力模拟前端
const files = fs.readdirSync(ANNOT_DIR).filter(f => f.endsWith('.json'));
let totalBacklinks = 0;

for (const file of files) {
    const sourceId = file.replace('.json', '');
    const data = JSON.parse(fs.readFileSync(path.join(ANNOT_DIR, file), 'utf-8'));

    currentLawId = sourceId;
    currentLawTitle = data.name || lawMap[sourceId] || "未知";

    const items = data.content || [];
    for (const item of items) {
        const text = item.lawWebContent || "";
        if (!text) continue;

        // 分割行或独立段落，避免大段落上下文混乱，并找到自身是哪一条
        // 从原文（非html）直接提前抽取出所有的条款行号节点，建立索引数组
        const artPositions = [];
        const artRegex = /(?:^|\n)\s*(?:【[^】]+】\s*)*(第[一二三四五六七八九十百千零〇○]+条(?:之[一二三四五六七八九十]+)?)/g;
        let mArt;
        while ((mArt = artRegex.exec(text)) !== null) {
            artPositions.push({ pos: mArt.index, num: mArt[1] });
        }
        const numRegex1 = /(?:^|\n)\s*(\d+)[\.．]/g;
        let mNum1;
        while ((mNum1 = numRegex1.exec(text)) !== null) {
            artPositions.push({ pos: mNum1.index, num: mNum1[1] + '.' });
        }
        const numRegex2 = /(?:^|\n)\s*([一二三四五六七八九十百千]+)、/g;
        let mNum2;
        while ((mNum2 = numRegex2.exec(text)) !== null) {
            artPositions.push({ pos: mNum2.index, num: mNum2[1] + '、' });
        }

        artPositions.sort((a, b) => a.pos - b.pos);

        // 返回某个位置所属的条款号
        function findSourceArticle(matchPos) {
            let best = "无行号";
            for (const item of artPositions) {
                if (item.pos <= matchPos) {
                    best = item.num;
                } else {
                    break;
                }
            }
            return best;
        }

        // --- 核心点：执行前端打链 ---
        const outputHtml = processLawReferences(text);

        // --- 取出所有被打上 law-ref 的实体 ---
        // 这里必须用更严谨的正宗 DOM 思想，但为了性能咱们用一个包含 data-full 和 innerText 的正则抽取
        const refRegex = /<span class="law-ref(?: law-ref-self)?"?[^>]*data-law-id="([^"]+)"?[^>]*data-law-name="([^"]*)"?[^>]*data-article="([^"]+)"?[^>]*data-full="([^"]+)"?[^>]*>(.*?)<\/span>/g;
        let match;

        let lastRawSearchCursor = 0;

        while ((match = refRegex.exec(outputHtml)) !== null) {
            const tgtId = match[1];
            const tgtName = match[2];
            const tgtArt = match[3];
            const tgtFull = match[4];
            const spanHtmlContent = match[5]; // 这是里面的文字，包含可能被替换的各种奇怪内容

            // 重要：为了在原始 `text` 里面找到它的产生位置（因为前端替换可能增加内容如《公司法2023版》）
            // 我们不搜索替换过的内容，而是简单粗暴地搜索 `tgtFull`（即 原“法名+条号”）或者只搜前面的文字部分
            // 可是因为是反引用，有本法引用、连续引用等，并不一定有 tgtFull 在原文中一模一样存在。

            // 最稳妥的方法：利用 match 在 outputHtml 中的位置，换算回原文的无标签近似位置
            const htmlBefore = outputHtml.substring(0, match.index);
            const plainBefore = htmlBefore.replace(/<[^>]+>/g, '');
            const approxPos = plainBefore.length;

            const sArt = findSourceArticle(approxPos);
            // 这里也不要用 text.indexOf 了，直接依靠 approxPos 加前后截取
            const ctxText = getContext(text, approxPos, spanHtmlContent.replace(/<[^>]+>/g, '').length, 50);

            // 过滤掉因为“民事诉讼法 法条 -> 《民法典2023》”替换导致的源法条名被污染为别的的问题。
            // 使用 sourceName，即 file 名对应的原始法律名称
            addBacklink(tgtId, tgtArt, sourceId, currentLawTitle, sArt, ctxText);
            totalBacklinks++;
        }
    }
}

// 统计实际新增引用数量 (去掉同目标下的重复条目)
let uniqueCount = 0;
for (const [tgtId, tgtMap] of Object.entries(backlinks)) {
    for (const [tgtArt, srcList] of Object.entries(tgtMap)) {
        uniqueCount += srcList.length;
    }
}

console.log(`Node提取反引用完成！\n总抽取标签量：${totalBacklinks}\n去重后有效连接数：${uniqueCount}`);

const jsContent = "var globalBacklinks = " + JSON.stringify(backlinks, null, 0) + ";";
fs.writeFileSync(OUT_FILE, jsContent, 'utf-8');
console.log(`结果写入：${OUT_FILE}`);

