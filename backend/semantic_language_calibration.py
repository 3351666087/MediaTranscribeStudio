"""Conservative, text-only language signals for semantic arbitration.

This module is deliberately independent from candidate generation and
composition. It does not decide a language or rewrite text; it only exposes
visible script evidence that an arbitrator can use when deciding whether to
request bounded language and ASR challengers.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .language import normalize_language_tag
from .persistence import validate_strict_json


SEMANTIC_LANGUAGE_CALIBRATION_SCHEMA_VERSION = "1.0.0"
SEMANTIC_LANGUAGE_CALIBRATION_ARTIFACT_TYPE = (
    "semantic-language-calibration"
)

_SCRIPT_KEYS = (
    "latin",
    "han",
    "kana",
    "hangul",
    "cyrillic",
    "arabic",
    "devanagari",
    "greek",
    "hebrew",
    "thai",
    "other",
)
_KNOWN_SCRIPTS = frozenset(_SCRIPT_KEYS) - {"other"}
_LANGUAGE_DOMAINS = ("language-span", "asr-text")
_SCRIPT_MARKERS = (
    ("LATIN", "latin"),
    ("CJK", "han"),
    ("IDEOGRAPH", "han"),
    ("HIRAGANA", "kana"),
    ("KATAKANA", "kana"),
    ("HANGUL", "hangul"),
    ("CYRILLIC", "cyrillic"),
    ("ARABIC", "arabic"),
    ("DEVANAGARI", "devanagari"),
    ("GREEK", "greek"),
    ("HEBREW", "hebrew"),
    ("THAI", "thai"),
)

_LATIN_LANGUAGES = frozenset(
    {
        "af",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "la",
        "lt",
        "lv",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "vi",
    }
)
_HAN_LANGUAGES = frozenset({"cmn", "hak", "nan", "wuu", "yue", "zh"})
_ARABIC_LANGUAGES = frozenset({"ar", "fa", "ps", "ur"})
_CYRILLIC_LANGUAGES = frozenset({"be", "bg", "mk", "ru", "sr", "uk"})
_DEVANAGARI_LANGUAGES = frozenset({"hi", "mr", "ne", "sa"})
_SCRIPT_SUBTAG_EXPECTATIONS = {
    "Arab": frozenset({"arabic"}),
    "Cyrl": frozenset({"cyrillic"}),
    "Deva": frozenset({"devanagari"}),
    "Grek": frozenset({"greek"}),
    "Hang": frozenset({"hangul"}),
    "Hani": frozenset({"han"}),
    "Hans": frozenset({"han"}),
    "Hant": frozenset({"han"}),
    "Hebr": frozenset({"hebrew"}),
    "Hira": frozenset({"kana"}),
    "Jpan": frozenset({"han", "kana"}),
    "Kana": frozenset({"kana"}),
    "Kore": frozenset({"hangul", "han"}),
    "Latn": frozenset({"latin"}),
    "Thai": frozenset({"thai"}),
}

# Technical literals are excluded only from conflict/routing decisions. Raw
# script counts still describe exactly what is visibly present.
_PROTECTED_TOKENS = frozenset(
    {
        "ai",
        "api",
        "asr",
        "ass",
        "cli",
        "cpu",
        "css",
        "csv",
        "gpu",
        "html",
        "http",
        "https",
        "json",
        "llm",
        "pdf",
        "ram",
        "sdk",
        "sql",
        "srt",
        "ui",
        "url",
        "ux",
        "vad",
        "vtt",
        "xml",
        "yaml",
    }
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:[._'’/-][^\W_]+)*", re.UNICODE)
_URL_RE = re.compile(r"(?i)^(?:https?://|www\.)")
_VERSION_RE = re.compile(r"(?i)^(?:v?\d+(?:[._-]\d+)+)$")
_TECHNICAL_LITERAL_RE = re.compile(
    r"(?i)^(?:[a-z]+\d+[a-z0-9._-]*|"
    r"\d+[a-z]+[a-z0-9._-]*)$"
)

# Unambiguous forms used for a conservative Hans/Hant signal. This is a
# detector, not a conversion table; shared Han characters are omitted.
_SIMPLIFIED_HAN = frozenset(
    "万与专业东丝两严个为丽举么义乌乐习乡书买乱争于亏云亚产亲亿仅从们价众优"
    "伙会伟传伤伦体余兴养内册写军农冲决况冻净凉减凤击则刚创别制剂剑剧劝办"
    "务动区医华协单卖卫厂县参双发变叶号后吗听启员团园围图圆场坏块声处备复"
    "头夺奖孙学宁宝实宠审对寻导寿将尔尘尽层岁岛岭岳峡币师带帮干并广庄庆库应"
    "开异弃张强归当忆忧怀态总恋恒恳恶悦悬惊战户扑执扩扫扬扰抚报担拢拥择挡挣"
    "挥损换据掩携摄摆摇摊撑敌数斗断无旧时显晓术机杀杂权条来杨极构枪柜树栋标"
    "样档桥梦检楼欢欧毁毕气汇汉污汤沟沪泪洁洒浇济浓涛灭灯灵灾炉点热炼烂烟烦"
    "烧爱爷牵状犹独献玛环现电画畅监盖盘着矿码砖礼祸离种积称穷窃窝竞笔筑签简"
    "粮紧红纤约级纪纯纳纵纷纸线练组终结绝统继续绩绣绿网罗罚翻耀耻职联聪肃胜"
    "脉脱脑脸腊腾节艰艳艺苏范荣药获莱莲营葱蒋蓝虚虫虽蚁蚀补装见观规视觉计订"
    "认讥讨让训议讯记讲许论设访证评词译试诗诚话诞该详语误说请读课谁调财责败"
    "货质贩贯贱贴贵贷贸费贺资赌购赛赵赶趋跃车轨转轮软轻载较辈输边达迁过迈运"
    "还这进远违连迟选递逻遗邮郁邻郑门闪闭闷闲阅队阳阴阵阶际陆陈险随隐难雾静"
    "顿须领预饱馆马驱骑验鱼鲁鸟鸡鸣黄齐齿龙龟"
) - frozenset({"掩", "翻", "耀"})
_TRADITIONAL_HAN = frozenset(
    "萬與專東絲兩嚴個為麗舉麼義烏樂習鄉書買亂爭於虧雲亞產親億僅從們價眾優"
    "夥會偉傳傷倫體餘興養內冊寫軍農沖決況凍淨涼減鳳擊則剛創別製劑劍劇勸辦"
    "務動區醫華協單賣衛廠縣參雙發變葉號後嗎聽啟員團園圍圖圓場壞塊聲處備復"
    "頭奪獎孫學寧寶實寵審對尋導壽將爾塵盡層歲島嶺嶽峽幣師帶幫幹並廣莊慶庫"
    "應開異棄張強歸當憶憂懷態總戀恆懇惡悅懸驚戰戶撲執擴掃揚擾撫報擔攏擁擇"
    "擋掙揮損換據掩攜攝擺搖攤撐敵數鬥斷無舊時顯曉術機殺雜權條來楊極構槍櫃樹"
    "棟柵標樣檔橋夢檢樓歡歐毀畢氣匯漢汙湯溝滬淚潔灑澆濟濃濤滅燈靈災爐點熱煉"
    "爛煙煩燒愛爺牽狀猶獨獻瑪環現電畫暢監蓋盤著礦碼磚禮禍離種積稱窮竊窩競筆"
    "築簽簡糧緊纖約級紀純納縱紛紙線練組終結絕統繼續績繡綠網羅罰翹翻耀恥職聯"
    "聰肅勝脈脫腦臉臘騰節艱艷藝蘇範榮藥獲萊蓮營蔥蔣藍虛蟲雖蟻蝕補裝見觀規"
    "視覺計訂認譏討讓訓議訊記講許論設訪證評詞譯試詩誠話誕該詳語誤說請讀課誰"
    "調財責敗貨質販貫賤貼貴貸貿費賀資賭購賽趙趕趨躍車軌轉輪軟輕載較輩輸邊達"
    "遷過邁運還這進遠違連遲選遞邏遺郵鬱鄰鄭門閃閉悶閒閘閱閥隊陽陰陣階際陸陳"
    "險隨隱難霧靜頓須領預飽館馬驅騎驗魚魯鳥雞鳴黃齊齒龍龜"
) - frozenset({"掩", "翻", "耀"})


def _script_for_character(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for marker, script in _SCRIPT_MARKERS:
        if marker in name:
            return script
    return "other"


def _empty_profile() -> dict[str, int]:
    return {key: 0 for key in _SCRIPT_KEYS}


def _raw_script_profile(text: str) -> dict[str, int]:
    profile = _empty_profile()
    for character in text:
        script = _script_for_character(character)
        if script is not None:
            profile[script] += 1
    return profile


def _token_is_protected(token: str) -> bool:
    normalized = token.strip("._'’/-").casefold()
    if not normalized:
        return True
    if _URL_RE.match(normalized):
        return True
    if normalized in _PROTECTED_TOKENS:
        return True
    if (
        "." in normalized
        and normalized.isascii()
        and normalized.replace(".", "").isalnum()
    ):
        # Hostnames and dotted API names are technical literals, not language
        # evidence. Version strings are covered separately below.
        return True
    if _VERSION_RE.fullmatch(normalized):
        return True
    if _TECHNICAL_LITERAL_RE.fullmatch(normalized):
        return True
    # A single ASCII letter is usually an initial, variable, or version marker.
    if len(normalized) == 1 and normalized.isascii():
        return True
    return False


def _meaningful_script_profile(text: str) -> tuple[dict[str, int], int]:
    profile = _empty_profile()
    protected_letters = 0
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        token_scripts: list[str] = []
        for character in token:
            script = _script_for_character(character)
            if script is not None:
                token_scripts.append(script)
        if _token_is_protected(token):
            protected_letters += len(token_scripts)
            continue
        for script in token_scripts:
            profile[script] += 1
    return profile, protected_letters


def _language_parts(language: str) -> tuple[str, str | None]:
    subtags = language.split("-")
    script = next(
        (
            subtag.title()
            for subtag in subtags[1:]
            if len(subtag) == 4 and subtag.isalpha()
        ),
        None,
    )
    return subtags[0], script


def _expected_scripts(
    primary: str,
    script: str | None,
) -> frozenset[str] | None:
    if script in _SCRIPT_SUBTAG_EXPECTATIONS:
        return _SCRIPT_SUBTAG_EXPECTATIONS[script]
    if primary in _LATIN_LANGUAGES:
        return frozenset({"latin"})
    if primary in _HAN_LANGUAGES:
        return frozenset({"han"})
    if primary in _ARABIC_LANGUAGES:
        return frozenset({"arabic"})
    if primary in _CYRILLIC_LANGUAGES:
        return frozenset({"cyrillic"})
    if primary in _DEVANAGARI_LANGUAGES:
        return frozenset({"devanagari"})
    if primary == "ja":
        return frozenset({"han", "kana"})
    if primary == "ko":
        return frozenset({"hangul", "han"})
    if primary == "th":
        return frozenset({"thai"})
    return None


def _variant_counts(text: str) -> tuple[int, int]:
    simplified = sum(character in _SIMPLIFIED_HAN for character in text)
    traditional = sum(character in _TRADITIONAL_HAN for character in text)
    return simplified, traditional


def _dominant_scripts(profile: Mapping[str, int]) -> list[str]:
    maximum = max((profile[key] for key in _KNOWN_SCRIPTS), default=0)
    if maximum <= 0:
        return []
    return [
        key
        for key in _SCRIPT_KEYS
        if key in _KNOWN_SCRIPTS and profile[key] == maximum
    ]


def _code_switch_likely(
    profile: Mapping[str, int],
    *,
    primary: str,
) -> bool:
    substantial = {
        key for key in _KNOWN_SCRIPTS if profile[key] >= 2
    }
    if len(substantial) < 2:
        return False
    if primary == "ja" and substantial.issubset({"han", "kana"}):
        return False
    if primary == "ko" and substantial.issubset({"hangul", "han"}):
        return False
    return True


def _variant_conflict(
    *,
    script: str | None,
    simplified_count: int,
    traditional_count: int,
) -> bool:
    if script not in {"Hans", "Hant"}:
        return False
    evidence = simplified_count + traditional_count
    if evidence < 2:
        return False
    if script == "Hant":
        return simplified_count >= 2 and simplified_count >= traditional_count
    return traditional_count >= 2 and traditional_count >= simplified_count


def build_semantic_language_calibration(
    claimed_language: str,
    visible_text: str,
) -> dict[str, Any]:
    """Return deterministic language/script hints without changing the text."""

    normalized_language = normalize_language_tag(
        claimed_language,
        allow_auto=False,
    )
    if not isinstance(visible_text, str):
        raise ValueError("visible_text must be a string")

    raw_profile = _raw_script_profile(visible_text)
    meaningful_profile, protected_letters = _meaningful_script_profile(
        visible_text
    )
    meaningful_letters = sum(meaningful_profile.values())
    primary, script = _language_parts(normalized_language)
    expected = _expected_scripts(primary, script)
    simplified_count, traditional_count = _variant_counts(visible_text)
    variant_conflict = _variant_conflict(
        script=script,
        simplified_count=simplified_count,
        traditional_count=traditional_count,
    )
    code_switch = _code_switch_likely(
        meaningful_profile,
        primary=primary,
    )

    language_conflict = False
    if meaningful_letters and expected is not None:
        expected_count = sum(meaningful_profile[key] for key in expected)
        known_count = sum(
            meaningful_profile[key] for key in _KNOWN_SCRIPTS
        )
        # An expected script plus another substantial script is treated as a
        # possible code switch, not proof that the claimed language is wrong.
        if expected_count == 0 or (
            known_count >= 4 and expected_count / known_count < 0.2
        ):
            language_conflict = not code_switch
        if variant_conflict:
            # An explicit Hans/Hant tag is a concrete script contract.
            language_conflict = True

    recommended_domains = (
        list(_LANGUAGE_DOMAINS)
        if code_switch or language_conflict or variant_conflict
        else []
    )
    profile = {
        **raw_profile,
        "letterCount": sum(raw_profile.values()),
        "meaningfulLetterCount": meaningful_letters,
        "protectedLetterCount": protected_letters,
        "simplifiedHanCount": simplified_count,
        "traditionalHanCount": traditional_count,
        "dominantScripts": _dominant_scripts(raw_profile),
    }
    result = {
        "schemaVersion": SEMANTIC_LANGUAGE_CALIBRATION_SCHEMA_VERSION,
        "artifactType": SEMANTIC_LANGUAGE_CALIBRATION_ARTIFACT_TYPE,
        "claimedLanguage": normalized_language,
        "scriptProfile": profile,
        "codeSwitchLikely": code_switch,
        "claimedLanguageConflict": language_conflict,
        "scriptVariantConflict": variant_conflict,
        "recommendedDomains": recommended_domains,
    }
    validate_strict_json(result)
    return result


def calibrate_visible_language(
    claimed_language: str,
    visible_text: str,
) -> dict[str, Any]:
    """Short alias for build_semantic_language_calibration."""

    return build_semantic_language_calibration(claimed_language, visible_text)


semantic_language_calibration = build_semantic_language_calibration
calibrate_semantic_language = build_semantic_language_calibration


__all__ = [
    "SEMANTIC_LANGUAGE_CALIBRATION_ARTIFACT_TYPE",
    "SEMANTIC_LANGUAGE_CALIBRATION_SCHEMA_VERSION",
    "build_semantic_language_calibration",
    "calibrate_visible_language",
    "calibrate_semantic_language",
    "semantic_language_calibration",
]
