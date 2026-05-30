"""LanguageTool wrapper used as a text-normalisation pre-pass.

Mirrors the shape of :mod:`DataCurator.GoogleTranslate.GoogleTranslate`:
a plain class that operates on raw strings, with no datatrove coupling.
A thin datatrove ``PipelineStep`` wrapper can be built on top by callers
that need it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import language_tool_python
from loguru import logger
from tenacity import Retrying, RetryError, stop_after_attempt


DEFAULT_ALLOWED_FIXES: Tuple[str, ...] = (
    "BRAK_PRZECINKA_",
    "ZBEDNA_WIELKA_LITERA",
    "DOUBLE_",
    "PREP_CASUS",
    "BRAK_SPACJI",
    "PARENTHESIS_WHITESPACE",
    "ZGODNIE_INST",
    "PODMIOT_ORZECZENIE",
    "IDENTYCZNY_JAK",
    "PL_WORD_COHERENCY",
    "PL_SIMPLE_REPLACE",
    "NIEZGODNO_PRZYPADKW_",
    "JAK_I",
    "DYWIZ",
    "POSIADAC_MIEC",
    "UPPERCASE_SENTENCE_START",
    "POZA_TYM",
    "TAK_NA_PRAWDE",
    "W_CHWILI_OBECNEJ",
    "W_WE",
    "PRZECINEK_ANI",
    "JEDNE_DZIECKO",
    "WORD_REPEAT_RULE",
    "COFANIE_PRZECINKA",
    "ZE_Z_SPOL",
    "NADTO_PRZECINEK",
    "BOJE_BOJĘ",
    "NIE_RZECZOWNIK",
    "ITP_ITD",
    "ZWROTNE_BEZ_SIE",
    "BYC_ZNAJDOWAC_SIE_W_POSIADANIU",
    "STOPIEN_WYZSZY",
    "WOLACZ_BEZ_PRZECINKA",
    "PYTANIE_CO",
    "CZASOWNIK_BY",
    "SKROTY_Z",
    "PRZECINEK_"
    "ROWNIE_JAK",
    "JEDNOSTKA_LICZBA",
    "MULTISPOJNIKI",
    "BRAK_",
    "Z_TYM_ZE",
    "NA_DZIEN_DZISIEJSZY",
    "ZE_ZE",
    "DWU_PRZYMIOTNIK",
    "WOJ_MAZOWIECKIE",
    "W_OPARCIU",
    "I_LUB",
    "POD_RZAD",
    "NIEZGODNOSC_KONCOWEK",
    "PRZECINEK_MYSLNIK",
    "Z_ZE",
    "BEDZIE_POTRAFIC",
    "SPACJA_W_SKROCIE",
    "WROCIC_Z_POWROTEM",
    "NAPRZECIWKO",
    "SENTENCE_WHITESPACE",
    "WYDAWAC_SIE_BYC",
    "Z_POSROD",
    "TEGO_OKRESY_TEGO_OKRESU",
    "RULE_NIEMNIEJ",
    "CO_WIECEJ",
    "TYM_BARDZIEJ_ZE",
    "PRZECINEK_POROWNANIE",
    "X-LATEK_MYSLNIK",
    "MNOZENIE",
    "PRZECINEK_TAK_JAK",
    "WIEDZIEC_CO",
    "ZNAJDUJE",
    "ZAROWNO_BEZ_JAK_I",
    "EFEKT_KONCOWY",
    "DOBRA_RENOMA",
    "PL_COMPOUNDS",
    "WRAZ_TYM",
    "DODATKOWO_CO_WIECEJ"
    "WG",
    "POKI_CO",
    "UZNAC_JAKO",
    "WIADACY",
    "ZBEDNA_SPACJA_PRZED",
    "JASNO_NIEBIESKI",
    "TZW_CUDZYSLOW",
    "ZAIMKI_DLUZSZE",
    "SKLADNIA_LICZEBNIKA",
    "PRZESTRZEGAC_CO",
    "WYSOKA_FORMA",
    "ROZCHODZI_SIE",
    "SPACJA_PROCENT",
    "WIEC_ZATEM_PRZECINEK",
    "NIE_JAKIKOLWIEK",
    "JEZUITA",
    "ZAGRANICA-ZA_GRANICA",
    "BRAK_PRZECINKA_",
    "WYDAWAC_SIE_BYC",
    "JEDNOSTKA_LICZBA",
    "ZAROWNO_I",
    "W_SKUTEK",
    "IMIONA_Z_APOSTROFAMI",
    "COMMA_PARENTHESIS_WHITESPACE",
    "UPPERCASE_SENTENCE_START",
    "SENTENCE_WHITESPACE",
    "SKROTOWCE_BEZ_DYWIZU",
    "JEDNOSTKA_LICZBA",
    "SPACJA_W_SKROCIE",
    "WIADACY",
    "MULTISPOJNIKI",
    "PRZECINEK_ANI",
    "COFAC_SIE_DO_TYLU",
    "BRAK_PRZECINKA_SPOJNIK_PROSTY",
    "IM_TYM",
    "DATA_KROPKA_PO_DNIU",
    "TYS",
    # "KONCEPT", <- może powodować błąd w zdaniu "To bardzo fajny koncept" -> "To bardzo fajny pojęcie"
    "PONAD_TO",
    "SKROTY_BEZ_KROPKI",
    "BRAK_PRZECINKA_PRZED_IMIESLOWEM_PRZYSLOWKOWYM",
    "PO_SRODKU",
    "W_OPARCIU",
    "DUZO_ZDROWSZY",
    # "PRZECINEK_PREP", <- tego nie aktywować, może powodować błędy
    "COFANIE_PRZECINKA",
    "BRAK_PRZECINKA_KTORY",
    "DOUBLE_PUNCTUATION",
    "PL_SIMPLE_REPLACE",
    "MUSZE_MUSZE",
    "PRZECINEK_ANI",
    "WIEDZIEC_CO",
    "TEZ_TEZ",
    "BRAK_PRZECINKA_ZE",
    "BRAK_PRZECINKA_SPOJNIK_PROSTY",
    "WORD_REPEAT_RULE",
    "PRZEDE_CZYM",
    "POZA_TYM",
    "PIERWSZY_STYCZEN_1993",
    "PELNIC_ROLE",
    "A_LA_CARTE",
)


class LanguageToolChecker:
    def __init__(
        self,
        language: str = "pl-PL",
        remote_server: str | None = "http://languagetool.loc",
        allowed_fixes: List[str] | Tuple[str, ...] | None = None,
        retries: int = 5,
        max_passes: int = 3,
    ) -> None:
        """Configure the LanguageTool client and the fix allow-list."""
        self.language = language
        self.remote_server = remote_server
        self.allowed_fixes = list(allowed_fixes) if allowed_fixes is not None else list(DEFAULT_ALLOWED_FIXES)
        self.retries = retries
        self.max_passes = max_passes

        if remote_server:
            self.tool = language_tool_python.LanguageTool(language, remote_server=remote_server)
        else:
            self.tool = language_tool_python.LanguageTool(language)

    def fix_and_get_fixes(self, text: str) -> Tuple[str, str]:
        """Run one LanguageTool pass; return ``(fixed_text, remaining_issues_str)``."""
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries)):
                with attempt:
                    checked_text = self.tool.check(text)

                    fixes_to_do = [
                        match for match in checked_text
                        if any(match.rule_id.startswith(allowed_fix) for allowed_fix in self.allowed_fixes)
                    ]

                    fixed = language_tool_python.utils.correct(text, fixes_to_do)
                    no_fixes = [m for m in checked_text if m.rule_id not in self.allowed_fixes]

                    final_fixes = "\n".join(str(issue) for issue in no_fixes)
                    return fixed, final_fixes
        except RetryError as e:
            last_exc = e.last_attempt.exception() if e.last_attempt else e
            exc_type = type(last_exc).__name__
            preview = text[:200].replace("\n", " ")
            logger.opt(exception=last_exc).error(
                f"LanguageTool failed after retries "
                f"[{exc_type}: {last_exc}] "
                f"(text_len={len(text)}, preview={preview!r})"
            )
            return text, f"Błąd przetwarzania [{exc_type}: {last_exc}]"

    def format(self, text: str) -> Tuple[str, str]:
        """Iterate :meth:`fix_and_get_fixes` until the text stabilises."""
        attempts = 0
        while (out := self.fix_and_get_fixes(text))[0] != text:
            if attempts >= self.max_passes:
                break
            text = out[0]
            attempts += 1
        return out

    def fix_and_get_issues(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Like :meth:`fix_and_get_fixes` but returns structured unresolved issues.

        Each issue is a dict with ``message``, ``suggestions``,
        ``error_text`` (the exact offending span pulled out of ``text``)
        and ``context`` (a short window of surrounding text so a
        downstream LLM can locate the issue). Rule IDs and raw offsets
        are still omitted on purpose.
        """
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries)):
                with attempt:
                    checked_text = self.tool.check(text)

                    fixes_to_do = [
                        match for match in checked_text
                        if any(match.rule_id.startswith(allowed_fix) for allowed_fix in self.allowed_fixes)
                    ]
                    fixed = language_tool_python.utils.correct(text, fixes_to_do)
                    no_fixes = [m for m in checked_text if m.rule_id not in self.allowed_fixes]

                    issues = [self._build_issue(text, m) for m in no_fixes]
                    return fixed, issues
        except RetryError as e:
            last_exc = e.last_attempt.exception() if e.last_attempt else e
            exc_type = type(last_exc).__name__
            preview = text[:200].replace("\n", " ")
            logger.opt(exception=last_exc).error(
                f"LanguageTool failed after retries "
                f"[{exc_type}: {last_exc}] "
                f"(text_len={len(text)}, preview={preview!r})"
            )
            return text, [{"message": f"Błąd przetwarzania [{exc_type}: {last_exc}]", "suggestions": []}]

    def format_issues(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Iterate :meth:`fix_and_get_issues` until the text stabilises.

        Returns the final fixed text plus the unresolved-issues list from
        the last pass (so the caller sees the issues that remain *after*
        all auto-fixes were applied).
        """
        attempts = 0
        out = self.fix_and_get_issues(text)
        while out[0] != text:
            if attempts >= self.max_passes:
                break
            text = out[0]
            out = self.fix_and_get_issues(text)
            attempts += 1
        return out

    @staticmethod
    def _build_issue(text: str, match: Any) -> Dict[str, Any]:
        """Extract a JSON-friendly issue dict from a LanguageTool match.

        Pulls the exact offending substring out of ``text`` and a short
        surrounding window so the downstream LLM can see *where* in the
        target the rule fired — not just what the rule said.
        """
        offset = int(getattr(match, "offset", 0) or 0)
        # language_tool_python exposes the span length as `error_length`
        # and the substring as `matched_text`. Older/forked versions may
        # use camelCase — accept both rather than relying on one spelling.
        length = int(
            getattr(match, "error_length", None)
            or getattr(match, "errorLength", None)
            or getattr(match, "length", 0)
            or 0
        )
        end = offset + length
        error_text = (
            getattr(match, "matched_text", None)
            or getattr(match, "matchedText", None)
            or (text[offset:end] if length > 0 else "")
        )
        ctx_pad = 40
        ctx_start = max(0, offset - ctx_pad)
        ctx_end = min(len(text), end + ctx_pad)
        prefix = text[ctx_start:offset]
        suffix = text[end:ctx_end]
        context = f"{prefix}«{error_text}»{suffix}" if error_text else text[ctx_start:ctx_end]
        return {
            "message": match.message,
            "suggestions": list(getattr(match, "replacements", []) or []),
            "error_text": error_text,
            "context": context,
        }
