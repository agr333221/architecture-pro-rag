"""
Задание 7: автоматическое тестирование бота на "золотом наборе" вопросов
и анализ покрытия базы знаний.

Симулирует пробелы в документации, исключая из поиска чанки трех сущностей
(GAP_SOURCES). Исключение делается на уровне поиска
(retriever.search(..., exclude_sources=...)), а не физическим удалением
файлов - чтобы не ломать остальные демонстрации.

ВЫБОР СУЩНОСТЕЙ - история двух ошибок, обе найдены код-ревью:

1. Изначально по аналогии с примером задания (Void Core / Xarn Velgor /
   Synth Flux) исключались tech_voyd_yadro.md / personage_ksarn_velgor.md /
   concept_synt_potok.md. Эти три концепции - "сквозные" для всей саги и
   упоминаются в 26 из 34 документов, поэтому их исключение не создавало
   настоящего пробела (0 из 5 честных "не знаю" в первом прогоне).
2. Затем были выбраны "Коррат Венн / Барон Илиан Кросс / Вэлторн" как
   сущности, которые ЯКОБЫ упоминаются только в собственном документе -
   но проверка велась поиском ТОЧНОЙ словоформы ("Барон Илиан Кросс"),
   которая не находит склоненные формы русского языка ("Барона Илиана
   Кросса", "Илиану Кроссу"...). После правильной проверки по основе
   слова без падежных окончаний (см. scripts/check_isolation.py)
   выяснилось, что Барон Илиан Кросс реально упоминается еще в
   ship_bluzhdayuschaya_kometa.md и docs/personage_aria_kess.md, а
   Коррат Венн - в personage_ilian_kross.md.

Из полного скана 18 крупных сущностей (scripts/check_isolation.py)
единственная действительно изолированная (0 посторонних упоминаний в любой
форме) - Вэлторн. К ней добавлены Виндрал и Коррат Венн - у них всего по
1 постороннему упоминанию каждый, поэтому GAP_SOURCES исключает не только
их собственные документы, но и эти 1-2 документа-упоминания, создавая
настоящий, проверенный пробел без побочных эффектов на остальные вопросы
golden-набора (проверено: ни один answerable-вопрос не полагается на
исключенные файлы).

Запуск:
    python scripts/evaluate.py
    python scripts/check_isolation.py   # проверка изоляции сущностей
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline import answer_query
from safety import is_decline as _is_decline

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "golden_questions.txt"
RESULTS_PATH = ROOT / "logs" / "evaluate_results.jsonl"

# Сущности, намеренно исключаемые из поиска для проверки честного "не знаю".
# Для каждой из трех проверяемых сущностей исключен и ее собственный документ,
# и все документы, где она упоминается в любой словоформе (проверено
# scripts/check_isolation.py - 0 посторонних упоминаний после исключения).
GAP_SOURCES = {
    # Вэлторн - 0 посторонних упоминаний, исключен только собственный файл.
    "knowledge_base/planet_veltorn.md",
    # Виндрал - 1 постороннее упоминание (org_orden_hranitelei_potoka.md).
    "knowledge_base/personage_vindral.md",
    "knowledge_base/org_orden_hranitelei_potoka.md",
    # Коррат Венн - 1 постороннее упоминание (personage_ilian_kross.md).
    "knowledge_base/personage_korrat_venn.md",
    "knowledge_base/personage_ilian_kross.md",
}


def parse_golden_questions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("---") if "QUESTION:" in b]

    questions = []
    for block in blocks:
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("QUESTION:"):
                fields["question"] = line[len("QUESTION:"):].strip()
            elif line.startswith("EXPECTED_STATUS:"):
                fields["expected_status"] = line[len("EXPECTED_STATUS:"):].strip()
            elif line.startswith("KEY_FACTS:"):
                raw = line[len("KEY_FACTS:"):].strip()
                fields["key_facts"] = [] if raw == "-" else [k.strip() for k in raw.split(",")]
        if "question" in fields:
            questions.append(fields)
    return questions


def evaluate_answer(item: dict, result: dict) -> dict:
    answer = result["answer"]
    # Находка код-ревью: сравнение по точному совпадению со строкой I_DONT_KNOW
    # не ловит случаи, когда LLM сама пишет отказ своими словами (например,
    # "...поэтому ответить нельзя. Ответ: Я не знаю."). is_decline() из
    # safety.py ищет подстроку "не зна(ю|ем)" независимо от формулировки.
    declined = _is_decline(answer)
    # Ответ считается небезопасным (например, содержит утечку секрета) -
    # такой ответ не может засчитываться как "корректный", даже если
    # формально не является отказом и содержит нужные KEY_FACTS.
    unsafe = not result.get("safe_output", True)

    if item["expected_status"] == "gap":
        correct = declined
        completeness = None
    else:
        key_facts = item.get("key_facts", [])
        hits = [k for k in key_facts if k.lower() in answer.lower()]
        completeness = len(hits) / len(key_facts) if key_facts else None
        correct = (not declined) and (not unsafe) and (completeness is None or completeness >= 0.5)

    return {
        "question": item["question"],
        "expected_status": item["expected_status"],
        "declined": declined,
        "unsafe": unsafe,
        "completeness": completeness,
        "correct": correct,
        "answer_preview": answer[:200],
        "sources": result["sources"],
    }


def main() -> None:
    questions = parse_golden_questions(GOLDEN_PATH)
    print(f"Загружено golden-вопросов: {len(questions)}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = []

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for item in questions:
            result = answer_query(item["question"], exclude_sources=GAP_SOURCES)
            evaluation = evaluate_answer(item, result)
            evaluation["timestamp"] = datetime.now(timezone.utc).isoformat()
            results.append(evaluation)
            f.write(json.dumps(evaluation, ensure_ascii=False) + "\n")

            status = "OK" if evaluation["correct"] else "FAIL"
            print(f"[{status}] ({item['expected_status']}) {item['question']}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    answerable = [r for r in results if r["expected_status"] == "answerable"]
    gaps = [r for r in results if r["expected_status"] == "gap"]

    print("\n--- Итоги ---")
    print(f"Всего вопросов: {total}, корректно обработано: {correct} ({correct / total:.0%})")
    if answerable:
        avg_completeness = sum(r["completeness"] or 0 for r in answerable) / len(answerable)
        print(f"Средняя полнота ответов на известные темы: {avg_completeness:.0%}")
    if gaps:
        correctly_declined = sum(1 for r in gaps if r["declined"])
        print(f"Честно сказал 'не знаю' на пробелах: {correctly_declined}/{len(gaps)}")

    print(f"\nПодробные результаты сохранены в {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
