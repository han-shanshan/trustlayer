from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re


class Translator:
    _instance = None

    @staticmethod
    def get_instance():
        if Translator._instance is None:
            Translator._instance = Translator()
        return Translator._instance

    def __init__(self):
        """
        translator: mbart-large-50-many-to-one-mmt
        Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE),
        Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX),
        Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP),
        Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN),
        Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL),
        Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN),
        Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE),
        Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK),
        Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

        language detector: papluca/xlm-roberta-base-language-detection
        arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr),
        hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru),
        swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)
        """
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-one-mmt", use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        self.language_mapping_dict, self.language_abbr_full_name_mapping = self.get_language_mapping_dict()

    @staticmethod
    def get_language_mapping_dict():  # multi language??
        language_detector_languages = "arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh)"
        translator_languages = "Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)"
        pattern = r'\b(\w+)\s\((\w+)\)'
        detector_languages = dict(re.findall(pattern, language_detector_languages))
        mapping = {}
        translator_languages = dict(re.findall(pattern, translator_languages))
        for k in translator_languages.keys():
            if k.lower() in detector_languages:
                mapping[detector_languages[k.lower()]] = translator_languages[k]

        return mapping, {value: key for key, value in detector_languages.items()}

    def language_unification(self, text):
        original_language = self.language_detector(text, top_k=1, truncation=True)[0]['label']
        if original_language == "en":
            return original_language, text
        self.tokenizer.src_lang = self.language_mapping_dict[original_language]
        encoded_hi = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_hi)
        return original_language, self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]