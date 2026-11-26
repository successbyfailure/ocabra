"""Repite casos de BFCL y los puntúa con el checker AST oficial.

Se ejecuta en el entorno oficial de BFCL (``uv run --project
berkeley-function-call-leaderboard --with soundfile``). Las tools se compilan
con ``convert_to_tool`` de BFCL, como su handler OpenAI, y el modelo se
registra con ``underscore_to_dot=True``: las APIs OpenAI no admiten puntos en
los nombres de función, así que BFCL los cambia por ``_`` al mandar las tools y
hay que deshacerlo al puntuar.
"""

import importlib.util
import json
import sys
from pathlib import Path

_common_spec = importlib.util.spec_from_file_location(
    "_replay_common", Path(__file__).with_name("_replay_common.py")
)
common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(common)

# Sin "_": el checker busca la config con model_name.replace("_", "/").
CHECKER_MODEL = "ocabra-diagnosis"


def main():
    args = common.arguments()
    sys.path.insert(0, args.source_root)
    from bfcl_eval.constants.enums import Language, ModelStyle
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.utils import convert_to_tool

    MODEL_CONFIG_MAPPING[CHECKER_MODEL] = ModelConfig(
        model_name=CHECKER_MODEL,
        display_name=CHECKER_MODEL,
        url="",
        org="oCabra",
        license="local",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    def language(category):
        if "javascript" in category:
            return Language.JAVASCRIPT
        if "java" in category:
            return Language.JAVA
        return Language.PYTHON

    def score(base_url, case, setting):
        entry = case["payload"]
        category = entry["test_category"]
        functions = entry["function"]
        question = entry["question"]
        messages = question[0] if question and isinstance(question[0], list) else question
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
        response = common.chat(base_url, setting["model"], messages, setting["params"], tools=tools)
        message, stats = common.summarize(response)
        calls = message.get("tool_calls") or []
        decoded = []
        for call in calls:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                arguments = {}
            decoded.append({function.get("name", ""): arguments})
        raw = [
            {c.get("function", {}).get("name", ""): c.get("function", {}).get("arguments")}
            for c in calls
        ]
        if "irrelevance" in category:
            # Acierta si NO llama a ninguna función.
            return {"passed": not decoded, "response": raw or message.get("content"), **stats}
        if not decoded:
            return {
                "passed": False,
                "error_type": "no_tool_call",
                "response": (message.get("content") or "")[:2000],
                **stats,
            }
        result = ast_checker(
            functions,
            decoded,
            entry["possible_answer"],
            language(category),
            category,
            CHECKER_MODEL,
        )
        return {
            "passed": bool(result.get("valid")),
            "error_type": result.get("error_type"),
            "response": raw,
            **stats,
        }

    spec = common.load(args.input)
    if spec.get("mode") == "rescore":
        # Sin GPU: vuelve a puntuar la respuesta que dio en el benchmark, ahora
        # con underscore_to_dot. Si así es válida, el fallo era del arnés.
        with open(args.output, "a", encoding="utf-8") as out:
            for case in spec["cases"]:
                entry = case["payload"]
                category = entry["test_category"]
                decoded = entry.get("model_result_decoded") or []
                row = {"case_id": case["case_id"], "setting": "rescore", "rep": 0}
                try:
                    if "irrelevance" in category or not isinstance(decoded, list):
                        row.update({"passed": None})
                    else:
                        result = ast_checker(
                            entry["function"],
                            decoded,
                            entry["possible_answer"],
                            language(category),
                            category,
                            CHECKER_MODEL,
                        )
                        row.update(
                            {
                                "passed": bool(result.get("valid")),
                                "error_type": result.get("error_type"),
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    row.update({"passed": None, "error": f"{type(exc).__name__}: {exc}"[:300]})
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    common.run(args.input, args.output, score)


if __name__ == "__main__":
    main()
