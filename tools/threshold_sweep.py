import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import cal_threshold_metrics, get_image_pairs


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for saved MixerCSeg results")
    parser.add_argument("--results_dir", required=True, help="Directory containing *_lab.png and *_pre.png files")
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated threshold list",
    )
    parser.add_argument("--suffix_gt", default="lab", help="Ground-truth filename suffix")
    parser.add_argument("--suffix_pred", default="pre", help="Prediction filename suffix")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    pred_list, gt_list, _, _ = get_image_pairs(args.results_dir, args.suffix_gt, args.suffix_pred)
    metrics = cal_threshold_metrics(pred_list, gt_list, thresholds)

    print("Threshold Sweep")
    print(f"results_dir: {args.results_dir}")
    print("threshold\tmIoU\tF1\tPrecision\tRecall")
    for item in metrics:
        print(
            f"{item['threshold']:.2f}\t"
            f"{item['mIoU']:.4f}\t"
            f"{item['F1']:.4f}\t"
            f"{item['Precision']:.4f}\t"
            f"{item['Recall']:.4f}"
        )


if __name__ == "__main__":
    main()
