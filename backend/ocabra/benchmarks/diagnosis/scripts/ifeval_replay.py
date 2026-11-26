"""Repite casos de IFEval y los puntúa con el checker oficial (modo strict).

Se ejecuta dentro del checkout de google-research con las dependencias del
scorer (``uv run --with absl-py --with langdetect --with nltk --with
immutabledict``), igual que el runner de IFEval de oCabra.
"""

import importlib.util
import sys
from pathlib import Path

_common_spec = importlib.util.spec_from_file_location(
    "_replay_common", Path(__file__).with_name("_replay_common.py")
)
common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(common)


def main():
    args = common.arguments()
    sys.path.insert(0, args.source_root)
    from instruction_following_eval import evaluation_lib

    def score(base_url, case, setting):
        source = case["payload"]["input"]
        prompt = source["prompt"]
        response = common.chat(
            base_url, setting["model"], [{"role": "user", "content": prompt}], setting["params"]
        )
        message, stats = common.summarize(response)
        text = message.get("content") or ""
        example = evaluation_lib.InputExample(
            key=source["key"],
            instruction_id_list=source["instruction_id_list"],
            prompt=prompt,
            kwargs=source["kwargs"],
        )
        result = evaluation_lib.test_instruction_following_strict(example, {prompt: text})
        return {
            "passed": bool(result.follow_all_instructions),
            "followed": [bool(item) for item in result.follow_instruction_list],
            "response": text[:4000],
            **stats,
        }

    common.run(args.input, args.output, score)


if __name__ == "__main__":
    main()
