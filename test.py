##

# 경로 잘 확인하고 코드 돌리기! (최종 저장되는 파일명 수정 안하면 덮어씌워지니 그 부분 주의해서 돌리기!)

##
from __future__ import annotations

import ast
from argparse import ArgumentParser
from pathlib import Path
import sys


def print_log(msg: str) -> None:
    print(msg)

## congif 파일에서 load_from = 'work_dirs/grcnn_r50_dota_pretrained_sar_wavelet/best_coco_bbox_mAP_epoch_12.pth' 이렇게 있어야 본인 pth 파일이 load 됨. config 파일도 아래 pth 파일에 맞는 cofig 파일로 수정해서 돌려야함.

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MSFA_ROOT = BASE_DIR / "SARDet_100K-main" / "SARDet_100K-main" / "MSFA"
DEFAULT_RUN_INPUT = BASE_DIR / "image_pile" ### input 이미지나 폴더 경로 ###
DEFAULT_RUN_CONFIG = BASE_DIR / "config.py" ### config 파일도 아래 pth 파일에 맞는 cofig 파일로 수정해서 돌려야함 ### 
DEFAULT_RUN_WEIGHTS = BASE_DIR / "best_coco_bbox_mAP_epoch_12.pth"   ### pth 파일 ###
DEFAULT_RUN_OUT_DIR = BASE_DIR / "outputs"  ##### 최종 저장되는 파일명 (수정 안하면 덮어씌워지니 주의해서 돌리기!) #####


def ensure_msfa_import(msfa_root: Path) -> None:
    """Import msfa, adding project root to sys.path when needed."""
    try:
        import msfa  # noqa: F401
        return
    except ModuleNotFoundError:
        if msfa_root.exists() and str(msfa_root) not in sys.path:
            sys.path.insert(0, str(msfa_root))
        import msfa  # noqa: F401


def parse_args():
    parser = ArgumentParser(description="Simple bbox inference runner.")

    parser.add_argument("inputs", nargs="?", default="", help="Input image or folder path.")
    parser.add_argument("model", nargs="?", default="", help="Config path or model alias.")

    parser.add_argument("--input", dest="input_opt", type=str, default="", help="Input path.")
    parser.add_argument("--config", dest="config_opt", type=str, default="", help="Config path.")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint file path.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")
    parser.add_argument("--pred-score-thr", type=float, default=0.3, help="BBox score threshold.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--show", action="store_true", help="Show popup window.")
    parser.add_argument("--no-save-vis", action="store_true", help="Do not save visualizations.")
    parser.add_argument("--no-save-pred", action="store_true", help="Do not save prediction json.")
    parser.add_argument("--print-result", action="store_true", help="Print prediction details.")
    parser.add_argument(
        "--palette",
        default="none",
        choices=["coco", "voc", "citys", "random", "none"],
        help="Palette used for visualization.",
    )
    parser.add_argument("--texts", type=str, default=None, help="Optional text prompt.")
    parser.add_argument("--custom-entities", "-c", action="store_true", help="Custom text entities.")
    parser.add_argument("--chunked-size", "-s", type=int, default=-1, help="Chunk size for many classes.")
    parser.add_argument("--tokens-positive", "-p", type=str, default=None, help="Grounding token positions.")
    parser.add_argument(
        "--msfa-root",
        type=str,
        default=str(DEFAULT_MSFA_ROOT),
        help="MSFA project root path.",
    )

    args = parser.parse_args()

    inputs = args.input_opt or args.inputs
    model = args.config_opt or args.model

    use_default_run = len(sys.argv) == 1 and not inputs and not model and not args.weights
    if use_default_run:
        inputs = str(DEFAULT_RUN_INPUT)
        model = str(DEFAULT_RUN_CONFIG)
        args.weights = str(DEFAULT_RUN_WEIGHTS)
        args.out_dir = str(DEFAULT_RUN_OUT_DIR)
        print_log("No CLI args detected. Using default test paths.")

    if not inputs:
        raise SystemExit("Input path is required. Use positional <inputs> or --input.")
    if not model and not args.weights:
        raise SystemExit("Config/model is required unless --weights is provided.")

    call_args = {
        "inputs": inputs,
        "out_dir": "" if (args.no_save_vis and args.no_save_pred) else args.out_dir,
        "pred_score_thr": args.pred_score_thr,
        "batch_size": args.batch_size,
        "show": args.show,
        "no_save_vis": args.no_save_vis,
        "no_save_pred": args.no_save_pred,
        "print_result": args.print_result,
        "texts": args.texts,
        "custom_entities": args.custom_entities,
    }

    if args.tokens_positive is not None:
        call_args["tokens_positive"] = ast.literal_eval(args.tokens_positive)
    # Keep compatibility with older mmdet versions.
    call_args.pop("tokens_positive", None)

    init_args = {
        "model": model,
        "weights": args.weights,
        "device": args.device,
        "palette": args.palette,
    }

    if init_args["model"] and str(init_args["model"]).endswith(".pth"):
        print_log("The model argument is a weights file; assigning it to --weights.")
        init_args["weights"] = init_args["model"]
        init_args["model"] = None

    return args, init_args, call_args


def main() -> None:
    args, init_args, call_args = parse_args()

    input_path = Path(call_args["inputs"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if init_args["model"]:
        model_path = Path(str(init_args["model"]))
        if model_path.suffix in {".py", ".json", ".yaml", ".yml"} and not model_path.exists():
            raise FileNotFoundError(f"Model/config not found: {model_path}")

    if init_args["weights"]:
        weights_path = Path(str(init_args["weights"]))
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    ensure_msfa_import(Path(args.msfa_root))

    try:
        from mmdet.apis import DetInferencer
    except ModuleNotFoundError as e:
        msg = str(e)
        if "mmcv._ext" in msg:
            raise SystemExit(
                "mmcv full package with compiled ops is required (missing mmcv._ext)."
            ) from e
        if "mmdet" in msg:
            raise SystemExit("mmdet is not installed. Install with: pip install mmdet") from e
        raise

    inferencer = DetInferencer(**init_args)

    chunked_size = args.chunked_size
    if hasattr(inferencer, "model") and hasattr(inferencer.model, "test_cfg"):
        inferencer.model.test_cfg.chunked_size = chunked_size

    inferencer(**call_args)

    print_log(f"Python: {sys.executable}")
    print_log(f"Input: {input_path}")
    if init_args["model"]:
        print_log(f"Config/Model: {init_args['model']}")
    if init_args["weights"]:
        print_log(f"Weights: {init_args['weights']}")
    if call_args["out_dir"] != "" and not (call_args["no_save_vis"] and call_args["no_save_pred"]):
        print_log(f"results have been saved at {call_args['out_dir']}")


if __name__ == "__main__":
    main()
