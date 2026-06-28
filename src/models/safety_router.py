from __future__ import annotations

import re
from typing import Any


CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bself[-\s]?harm\b",
    r"\bhurt myself\b",
    r"\bi do not want to live\b",
    r"\bi don't want to live\b",
    r"\bلا اريد ان اعيش\b",
    r"\bمش عايز اعيش\b",
    r"\bمش عايزة اعيش\b",
    r"\bمش عاوز اعيش\b",
    r"\bعايز اموت\b",
    r"\bعايزة اموت\b",
    r"\bعاوز اموت\b",
    r"\bنفسي اموت\b",
    r"\bنفسي أموت\b",
    r"\bهموت نفسي\b",
    r"\bهقتل نفسي\b",
    r"\bاقتل نفسي\b",
    r"\bانتحر\b",
    r"\bانتحار\b",
    r"\bأنتحر\b",
    r"\bأقتل نفسي\b",
]


CRISIS_RESPONSES = {
    "en": (
        "I'm really sorry you're feeling this much pain. If you might hurt yourself or feel in immediate danger, "
        "please call your local emergency number now or go to the nearest emergency department. If you are in the "
        "US or Canada, call or text 988 for immediate crisis support. If you can, stay near another person and "
        "move away from anything you could use to hurt yourself."
    ),
    "fr": (
        "Je suis vraiment désolé que tu ressentes autant de douleur. Si tu risques de te faire du mal ou si tu es "
        "en danger immédiat, appelle maintenant les urgences locales ou va au service d'urgence le plus proche. "
        "Si tu es aux États-Unis ou au Canada, appelle ou envoie un message au 988. Si possible, reste près d'une "
        "autre personne et éloigne-toi de tout objet dangereux."
    ),
    "ar": (
        "أنا آسف جدًا لأنك تمر بهذا الألم. إذا كنت قد تؤذي نفسك أو تشعر أنك في خطر فوري، اتصل برقم الطوارئ المحلي "
        "الآن أو اذهب إلى أقرب قسم طوارئ. إذا كنت في الولايات المتحدة أو كندا يمكنك الاتصال أو إرسال رسالة إلى 988. "
        "حاول أن تبقى بالقرب من شخص تثق به وابتعد عن أي شيء قد تستخدمه لإيذاء نفسك."
    ),
}


def detect_crisis(text: str) -> dict[str, Any]:
    clean_text = text.lower()
    matched = [pattern for pattern in CRISIS_PATTERNS if re.search(pattern, clean_text)]
    return {"is_crisis": bool(matched), "matched_patterns": matched}


def crisis_reply(language_code: str) -> str:
    return CRISIS_RESPONSES.get(language_code, CRISIS_RESPONSES["en"])
