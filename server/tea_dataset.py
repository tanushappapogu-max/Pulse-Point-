"""
Tea Text Dataset for TextCNN Training

Labeled examples covering 8 tea types and 10 flavor profiles.
Each entry: (text, tea_type_label, flavor_labels)

Tea types  (0-7): green, black, white, oolong, pu_erh, herbal, yellow, dark
Flavor labels (multi-hot, 10 dims):
  0=floral  1=earthy  2=grassy  3=smoky  4=sweet
  5=bitter  6=fruity  7=nutty   8=vegetal 9=marine
Quality tiers (0-3): ceremonial, premium, standard, culinary
"""

TEA_TYPES = [
    "green", "black", "white", "oolong",
    "pu_erh", "herbal", "yellow", "dark",
]

FLAVOR_LABELS = [
    "floral", "earthy", "grassy", "smoky", "sweet",
    "bitter", "fruity", "nutty", "vegetal", "marine",
]

QUALITY_TIERS = ["ceremonial", "premium", "standard", "culinary"]

# ---------------------------------------------------------------------------
# Labeled training examples
# Format: (text, tea_type_idx, [flavor_indices], quality_idx)
# ---------------------------------------------------------------------------

RAW_EXAMPLES = [
    # ── GREEN TEA ──────────────────────────────────────────────────────────
    ("gyokuro shade grown umami vegetal marine notes", 0, [8, 9, 2], 0),
    ("sencha fresh grassy green steamed japanese", 0, [2, 8, 4], 1),
    ("matcha ceremonial grade bright green powder frothy", 0, [2, 0, 5], 0),
    ("dragonwell longjing flat needle smooth nutty", 0, [7, 2, 4], 1),
    ("gunpowder rolled pellets smoky grassy robust", 0, [3, 2, 5], 2),
    ("bi luo chun spiral curly floral fruity spring", 0, [0, 6, 2], 1),
    ("hojicha roasted brown earthy woody nutty low caffeine", 0, [1, 7, 4], 2),
    ("genmaicha rice popcorn grassy toasted nutty", 0, [2, 7, 4], 2),
    ("kabusecha covered steamed vegetal sweet umami", 0, [8, 4, 9], 1),
    ("anji bai cha white green mineral crisp floral", 0, [0, 2, 4], 1),
    ("tai ping hou kui orchid floral fresh clean", 0, [0, 2, 4], 1),
    ("lu shan yun wu cloud mist high mountain grassy", 0, [2, 8, 0], 1),
    ("shincha first flush new harvest spring bright", 0, [2, 4, 0], 1),
    ("bancha coarse lower grade grassy everyday", 0, [2, 5, 8], 2),
    ("kukicha twig stem roasted mild low tannin", 0, [7, 4, 2], 2),
    ("tencha ground stone mill ceremonial sweet", 0, [4, 2, 0], 0),
    ("green tea jasmine scented floral sweet light", 0, [0, 4, 2], 2),
    ("pan fired wok roasted green smoky toasty", 0, [3, 7, 2], 2),
    ("green needle thin rolled grassy refreshing", 0, [2, 4, 8], 2),
    ("organic green full leaf whole leaf vegetal", 0, [2, 8, 4], 1),

    # ── BLACK TEA ──────────────────────────────────────────────────────────
    ("darjeeling first flush muscatel floral fruity", 1, [0, 6, 4], 1),
    ("assam bold malty brisk full bodied breakfast", 1, [1, 5, 7], 2),
    ("ceylon bright orange liquor citrus crisp", 1, [6, 4, 5], 1),
    ("lapsang souchong heavily smoked pine campfire", 1, [3, 1, 5], 2),
    ("keemun winy fruity orchid chocolate notes", 1, [0, 6, 7], 1),
    ("english breakfast robust bold brisk malty blend", 1, [5, 1, 7], 2),
    ("earl grey bergamot citrus floral aromatic black", 1, [0, 6, 4], 2),
    ("darjeeling second flush muscatel amber bold", 1, [6, 0, 5], 1),
    ("yunnan golden tip dianhong sweet honey earthy", 1, [4, 1, 7], 1),
    ("nilgiri frost queen bold brisk bright blue mountains", 1, [5, 4, 6], 2),
    ("irish breakfast strong robust bold everyday malt", 1, [5, 1, 7], 2),
    ("russian caravan smoky sweet caramel blend", 1, [3, 4, 7], 2),
    ("golden monkey yunnan golden tips sweet honey", 1, [4, 7, 0], 1),
    ("pu-erh ripe cooked shou earthy dark woody", 4, [1, 3, 7], 2),
    ("pu erh sheng raw aged green floral fruity", 4, [6, 0, 1], 1),
    ("black tea masala chai spiced ginger cardamom", 1, [4, 5, 7], 2),
    ("pu erh vintage aged decades deep earthy camphor", 4, [1, 3, 7], 0),
    ("black orthodox full leaf tippy golden bud", 1, [4, 5, 7], 1),
    ("autumn darjeeling muscatel fruity wine-like amber", 1, [6, 0, 4], 1),
    ("breakfast tea everyday teabag everyday malt robust", 1, [5, 7, 1], 2),

    # ── WHITE TEA ──────────────────────────────────────────────────────────
    ("silver needle bai hao yinzhen white downy buds", 2, [0, 4, 2], 0),
    ("white peony bai mu dan two leaves bud floral", 2, [0, 4, 6], 1),
    ("shou mei aged white fruity earthy honeyed", 2, [6, 1, 4], 1),
    ("white tea light delicate subtle honeydew floral", 2, [0, 4, 6], 1),
    ("aged white tea cake compressed fruity woodsy", 2, [6, 1, 0], 1),
    ("gong mei white coarse grade earthy fruity", 2, [1, 6, 4], 2),
    ("white moonlight bai yue liang yunnan wild", 2, [0, 4, 6], 1),
    ("darjeeling white tips floral muscatel light", 2, [0, 6, 4], 1),
    ("silver tips premium white downy buds delicate", 2, [0, 4, 2], 0),
    ("white tea organic whole leaf minimally processed", 2, [0, 4, 2], 1),

    # ── OOLONG TEA ─────────────────────────────────────────────────────────
    ("tieguanyin iron goddess floral orchid roasted", 3, [0, 7, 4], 1),
    ("dan cong duck shit oolong honey floral", 3, [0, 4, 6], 1),
    ("dong ding roasted oolong taiwan caramel nutty", 3, [7, 4, 3], 1),
    ("ali shan high mountain floral creamy milky", 3, [0, 4, 2], 0),
    ("da hong pao big red robe roasted mineral rock", 3, [3, 1, 7], 0),
    ("oriental beauty champagne oolong honey fruity", 3, [4, 6, 0], 0),
    ("wuyi rock oolong mineral smoky roasted oolong", 3, [3, 1, 7], 1),
    ("phoenix oolong dan cong fruity floral aroma", 3, [0, 6, 4], 1),
    ("milk oolong creamy buttery floral light taiwan", 3, [4, 0, 7], 1),
    ("baozhong pouchong lightly oxidized floral green", 3, [0, 2, 4], 1),
    ("concubine oolong bugbitten insect honey floral", 3, [0, 4, 6], 0),
    ("high mountain oolong alpine floral creamy cold", 3, [0, 4, 2], 0),
    ("roasted oolong dark caramel earthy depth", 3, [3, 7, 1], 2),
    ("oolong medium oxidized fruity floral complex", 3, [6, 0, 4], 1),

    # ── PU-ERH ─────────────────────────────────────────────────────────────
    ("shou pu erh ripe cooked earthy dark smooth", 4, [1, 3, 7], 2),
    ("sheng pu erh raw aged floral fruity evolving", 4, [6, 0, 1], 1),
    ("pu-erh cake tuo cha compressed aged vintage", 4, [1, 3, 7], 0),
    ("yunnan pu erh aged warehouse musty earthy", 4, [1, 3, 5], 2),
    ("aged sheng 30 year fruity apricot honeyed complex", 4, [6, 4, 0], 0),
    ("ripe pu erh smooth dark chocolate earthy", 4, [1, 7, 4], 2),
    ("pu erh mini tuo portable mushroom earthy", 4, [1, 3, 5], 2),
    ("wild arbor ancient tree gushu pu erh complex", 4, [1, 6, 0], 0),
    ("liu bao guangxi dark aged earthy woody betel", 7, [1, 3, 7], 2),

    # ── HERBAL / TISANE ────────────────────────────────────────────────────
    ("chamomile floral apple sweet calming herbal", 5, [0, 4, 6], 2),
    ("peppermint cool refreshing menthol herbal", 5, [4, 2, 8], 2),
    ("hibiscus tart fruity ruby red vitamin c", 5, [6, 4, 5], 2),
    ("rooibos red bush sweet earthy vanilla nutty", 5, [4, 1, 7], 2),
    ("ginger spicy warming root digestive herbal", 5, [4, 5, 8], 2),
    ("lavender floral calming purple herbal relaxing", 5, [0, 4, 2], 2),
    ("echinacea immune herbal earthy root medicinal", 5, [1, 5, 8], 2),
    ("lemon balm citrus bright herbal calming", 5, [6, 4, 0], 2),
    ("rosehip fruity tart vitamin c floral", 5, [6, 0, 4], 2),
    ("turmeric golden milk spiced earthy anti-inflammatory", 5, [4, 1, 8], 2),
    ("valerian root earthy herbal sleep calming", 5, [1, 5, 8], 2),
    ("spearmint mild sweet herbal minty cool", 5, [4, 2, 8], 2),
    ("licorice root sweet anise herbal naturally sweet", 5, [4, 6, 8], 2),
    ("elderflower floral delicate sweet spring", 5, [0, 4, 6], 2),
    ("nettle grassy earthy green vegetal iron rich", 5, [2, 1, 8], 2),

    # ── YELLOW TEA ─────────────────────────────────────────────────────────
    ("jun shan yin zhen yellow tea mellow sweet", 6, [4, 0, 2], 0),
    ("meng ding huang ya yellow tender mellow", 6, [4, 2, 0], 0),
    ("yellow tea mellow soft smooth mellowed green", 6, [4, 2, 0], 1),
    ("huang ya yellow bud smothered mellow sweet", 6, [4, 0, 2], 1),

    # ── DARK TEA ───────────────────────────────────────────────────────────
    ("liu bao dark tea earthy woody aged guangxi", 7, [1, 3, 7], 2),
    ("fu brick dark tea golden flower fermented", 7, [1, 7, 4], 2),
    ("anhua dark tea hunan compressed earthy", 7, [1, 3, 5], 2),
    ("tibetan butter tea dark fermented salty", 7, [1, 3, 5], 2),
    ("dark tea post-fermented aged earthy complex", 7, [1, 3, 7], 2),
]


def build_training_examples():
    """Return list of dicts ready for TextCNN training."""
    examples = []
    for text, type_idx, flavor_idxs, quality_idx in RAW_EXAMPLES:
        flavor_hot = [0] * len(FLAVOR_LABELS)
        for fi in flavor_idxs:
            flavor_hot[fi] = 1
        examples.append({
            "text": text,
            "tea_type": type_idx,
            "flavors": flavor_hot,
            "quality": quality_idx,
        })
    return examples


# ---------------------------------------------------------------------------
# Augmentation helpers — cheap data expansion for small datasets
# ---------------------------------------------------------------------------

import random

_SYNONYMS = {
    "floral": ["flowery", "bloom", "blossom", "aromatic"],
    "earthy": ["soil", "dirt", "musty", "loamy", "humus"],
    "grassy": ["green", "fresh", "vegetal", "herbal"],
    "smoky": ["smoked", "fire", "campfire", "charred", "roasted"],
    "sweet": ["honey", "honeyed", "sugary", "mellow", "syrupy"],
    "bitter": ["astringent", "brisk", "tannic", "sharp"],
    "fruity": ["fruit", "berry", "citrus", "tropical"],
    "nutty": ["nut", "toasted", "roasted", "walnut", "almond"],
    "vegetal": ["vegetable", "spinach", "seaweed", "marine"],
    "marine": ["ocean", "sea", "kelp", "umami", "briny"],
}


def augment_text(text, n=2):
    """Return n augmented variants of a text string."""
    words = text.split()
    variants = []
    for _ in range(n):
        new_words = []
        for w in words:
            lw = w.lower()
            if lw in _SYNONYMS and random.random() < 0.35:
                new_words.append(random.choice(_SYNONYMS[lw]))
            else:
                new_words.append(w)
        variants.append(" ".join(new_words))
    return variants


def build_augmented_dataset(seed=42):
    """Build training set with augmentation (3× size)."""
    random.seed(seed)
    base = build_training_examples()
    augmented = []
    for ex in base:
        augmented.append(ex)
        for aug_text in augment_text(ex["text"], n=2):
            augmented.append({**ex, "text": aug_text})
    return augmented
