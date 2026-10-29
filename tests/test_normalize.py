from app.ingestion.normalize import clean_text, detect_language, search_form


def test_clean_text_keeps_wording_but_removes_artifacts():
    raw = "الفصــل ٥.-‏  لا ينعقد الزواج\n\n\n\nإلا برضا"
    out = clean_text(raw)
    assert "ـ" not in out and "‏" not in out
    assert "5" in out                      # Arabic-Indic digits -> ASCII
    assert "\n\n\n" not in out
    assert "لا ينعقد الزواج" in out


def test_clean_text_rejoins_french_hyphenation():
    assert clean_text("le ma-\nriage") == "le mariage"


def test_search_form_folds_arabic_variants_and_accents():
    assert search_form("الزوجة") == search_form("الزوجه")
    assert search_form("إذن") == search_form("اذن")
    assert search_form("Époux") == "epoux"
    assert search_form("مُهْر") == "مهر"


def test_detect_language():
    assert detect_language("ما هو السن القانوني للزواج؟") == "ar"
    assert detect_language("Quelles sont les conditions du divorce en Tunisie ?") == "fr"
    assert detect_language("What does Tunisian law say about child custody?") == "en"


def test_fix_lam_alef_ligature_order():
    from app.ingestion.normalize import fix_lam_alef
    assert fix_lam_alef("ال يحكم بالطالق إال بعد") == "لا يحكم بالطالق إلا بعد"
    assert fix_lam_alef("في صياغته األصلية") == "في صياغته الأصلية"
    assert fix_lam_alef("الزواج") == "الزواج"
